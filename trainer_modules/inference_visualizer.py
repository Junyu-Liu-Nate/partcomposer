import random
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import torch
import wandb
import re
import modules.ptp_utils as ptp_utils

from utils.dataset import PartComposerSynthDataset

class InferenceVisualizer:
    def __init__(self, trainer):
        self.trainer = trainer

    ### Inference and visualization
    @torch.no_grad()
    def perform_full_inference(self, instance_prompt, path=None, guidance_scale=7.5, seed=None):
        # Backup the original attention processors
        original_attn_processors = self.trainer.unet.attn_processors

        # Reset the attention controller
        self.trainer.controller.attention_store = {}
        self.trainer.controller.cur_step = 0

        # Attach the attention controller to the unet
        # self.trainer.register_attention_control(self.trainer.controller)

        self.trainer.unet.eval()
        if self.trainer.args.train_text_encoder:
            self.trainer.text_encoder.eval()

        ### Currently using seed to generate noise
        torch.manual_seed(seed)
        latents = torch.randn((1, 4, 64, 64), device=self.trainer.accelerator.device)
        uncond_input = self.trainer.tokenizer(
            [""],
            padding="max_length",
            max_length=self.trainer.tokenizer.model_max_length,
            return_tensors="pt",
        ).to(self.trainer.accelerator.device)

        input_ids = self.trainer.tokenizer(
            [instance_prompt],
            padding="max_length",
            truncation=True,
            max_length=self.trainer.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.to(self.trainer.accelerator.device)
        cond_embeddings = self.trainer.text_encoder(input_ids)[0]
        uncond_embeddings = self.trainer.text_encoder(uncond_input.input_ids)[0]
        text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])

        for t in self.trainer.validation_scheduler.timesteps:
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.trainer.validation_scheduler.scale_model_input(
                latent_model_input, timestep=t
            )

            pred = self.trainer.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            )
            noise_pred = pred.sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            latents = self.trainer.validation_scheduler.step(noise_pred, t, latents).prev_sample
        latents = 1 / 0.18215 * latents

        images = self.trainer.vae.decode(latents.to(self.trainer.weight_dtype)).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).round().astype("uint8")

        self.trainer.unet.train()
        if self.trainer.args.train_text_encoder:
            self.trainer.text_encoder.train()

        # # Restore the original attention processors
        self.trainer.unet.set_attn_processor(original_attn_processors)

        # Image.fromarray(images[0]).save(path)
        img = images[0]
        return img

    def extract_concept_labels(self, instance_prompt: str, out_dim: int) -> torch.Tensor:
        """
        Parses the instance prompt for <assetX> tokens and returns a [1, out_dim] multi-hot tensor.
        
        Args:
            instance_prompt (str): e.g. "a photo of a chair with <asset0> and <asset1> and <asset6> and <asset7>".
            out_dim (int): The total number of possible concept tokens (e.g. 8).

        Returns:
            torch.Tensor: Shape (1, out_dim). Each position is 1 if <assetX> was found in the prompt, else 0.
        """
        # 1. Find all matches of the form <assetX> in the prompt
        pattern = r"<asset(\d+)>"
        matches = re.findall(pattern, instance_prompt)
        # matches will be a list of digit strings, e.g. ["0", "1", "6", "7"]

        # 2. Create a zero vector of shape [1, out_dim]
        concept_vector = torch.zeros((1, out_dim), dtype=torch.float32)

        # 3. Convert each match to int and set that index to 1
        for match_str in matches:
            concept_idx = int(match_str)
            if 0 <= concept_idx < out_dim:
                concept_vector[0, concept_idx] = 1.0

        return concept_vector

    def perform_concept_guidance_inference(
        self,
        instance_prompt,
        path=None,
        guidance_scale=7.5,
        concept_guidance_scale=1.0,
        seed=None
    ):
        """
        A variant of concept guidance that:
        - uses no grad on the UNet
        - only backprop on x_0_est through the concept predictor
        - forcibly plugs updated x_0_est into a custom step function

        This should reduce memory usage drastically, but produces an "off-chain" sample.
        """

        ###############################
        # 1) Setup (same as full_inference)
        ###############################
        original_attn_processors = self.trainer.unet.attn_processors
        self.trainer.controller.attention_store = {}
        self.trainer.controller.cur_step = 0

        self.trainer.unet.eval()
        if self.trainer.args.train_text_encoder:
            self.trainer.text_encoder.eval()

        if seed is not None:
            torch.manual_seed(seed)

        latents = torch.randn((1, 4, 64, 64), device=self.trainer.accelerator.device)

        # Prepare text embeddings for CFG
        uncond_input = self.trainer.tokenizer([""], padding="max_length",
                                    max_length=self.trainer.tokenizer.model_max_length,
                                    return_tensors="pt").to(self.trainer.accelerator.device)
        input_ids = self.trainer.tokenizer(
            [instance_prompt],
            padding="max_length",
            truncation=True,
            max_length=self.trainer.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.to(self.trainer.accelerator.device)

        cond_embeddings = self.trainer.text_encoder(input_ids)[0]
        uncond_embeddings = self.trainer.text_encoder(uncond_input.input_ids)[0]
        text_embeddings = torch.cat([uncond_embeddings, cond_embeddings], dim=0)

        timesteps = self.trainer.validation_scheduler.timesteps
        total_steps = len(timesteps)
        concept_start_step = max(0, total_steps - 10)  # do concept x0 guidance in last 10 steps

        # Setup concept predictor in eval + half precision
        self.trainer.concept_predictor.eval()
        # self.trainer.concept_predictor.to(self.trainer.accelerator.device, dtype=torch.float16)
        self.trainer.concept_predictor.to(self.trainer.accelerator.device)

        # Build concept labels
        concept_labels = self.extract_concept_labels(
            instance_prompt, out_dim=self.trainer.total_num_of_assets
        ).to(self.trainer.accelerator.device)

        ###############################
        # 2) Diffusion sampling
        ###############################
        for i, t in enumerate(timesteps):
            # Scale latents for this step
            with torch.no_grad():
                latent_model_input = torch.cat([latents, latents], dim=0)
                latent_model_input = self.trainer.validation_scheduler.scale_model_input(latent_model_input, t)

                # UNet forward => predicted noise for uncond & cond
                pred = self.trainer.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)
                noise_pred = pred.sample

                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                # Classifier-free guidance
                noise_pred_final = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # Standard formula: x_0_est = ( x_t - sqrt(1-alpha_t)* noise ) / sqrt(alpha_t)
                alpha_t = self.trainer.validation_scheduler.alphas_cumprod[t].to(latents.device)
                alpha_t_sqrt = alpha_t.sqrt()
                one_minus_alpha_t_sqrt = (1 - alpha_t).sqrt()

                x_0_est = (latents - one_minus_alpha_t_sqrt * noise_pred_final) / alpha_t_sqrt

            # If not in concept guidance zone, do normal step
            do_concept_guidance = (i >= concept_start_step)

            if not do_concept_guidance:
                # Just run standard step
                with torch.no_grad():
                    latents = self.custom_ddim_step(x_0_est, latents, t)
                continue

            ###############################
            # 3) Concept guidance on x_0_est only
            ###############################
            with torch.enable_grad():
                #  "x_0_est" is a float tensor => let's make a clone so can do backprop on x_0_est
                x_0_est_guide = x_0_est.clone().detach().requires_grad_(True)

                # if x_0_est_guide.dim() == 5:
                #     x_0_est_guide = x_0_est_guide.unsqueeze(0)
                # print(f'x_0_est_guide shape: {x_0_est_guide.shape}')
                # x_0_est_guide = x_0_est_guide.half()
                x_0_est_guide.retain_grad()

                # Pass x_0_est_guide -> concept predictor
                if self.trainer.args.predictor_type == "classifier":
                    logits = self.trainer.concept_predictor(x_0_est_guide)  # shape [1, out_dim]
                elif self.trainer.args.predictor_type == "classifier_seg":
                    logits, _ = self.trainer.concept_predictor(x_0_est_guide)  # shape [1, out_dim]
                elif self.trainer.args.predictor_type == "classifier_time":
                    logits = self.trainer.concept_predictor(x_0_est_guide, t.to(x_0_est_guide.device))
                elif self.trainer.args.predictor_type in ["classifier_seg_time", "classifier_seg_time_film"]:
                    logits, _ = self.trainer.concept_predictor(x_0_est_guide, t.to(x_0_est_guide.device))
                else:
                    print("Invalid concept predictor type.")
                concept_loss = F.binary_cross_entropy_with_logits(logits, concept_labels)

                concept_loss.backward()
                # print("=== Printing concept predictor gradients ===")
                # for name, param in self.trainer.concept_predictor.named_parameters():
                #     if param.grad is not None:
                #         print(f"Param {name}: grad shape = {param.grad.shape}, "
                #             f"grad dtype = {param.grad.dtype}, "
                #             f"grad mean = {param.grad.mean().item():.6f}")
                #     else:
                #         print(f"Param {name}: grad is None")
                # print(f"[DEBUG] concept_loss = {concept_loss.item()}")
                # # Check grad
                # if x_0_est_guide.grad is None:
                #     print("[DEBUG] x_0_est_guide.grad is STILL None => Possibly overshadowed by no_grad!")
                # else:
                #     print(f"[DEBUG] x_0_est_guide.grad shape: {x_0_est_guide.grad.shape}, mean={x_0_est_guide.grad.mean().item()}")

                x_0_est_updated = x_0_est_guide - concept_guidance_scale * x_0_est_guide.grad

            # Then do a custom step that uses x_0_est_updated => x_{t-1}
            with torch.no_grad():
                latents = self.custom_ddim_step(x_0_est_updated, latents, t)

        ###############################
        # 4) Decode final latents => image
        ###############################
        latents = latents / 0.18215
        images = self.trainer.vae.decode(latents.to(self.trainer.weight_dtype)).sample
        images = (images * 0.5 + 0.5).clamp(0, 1)
        images = images.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).round().astype("uint8")

        # restore
        self.trainer.unet.train()
        if self.trainer.args.train_text_encoder:
            self.trainer.text_encoder.train()
        self.trainer.unet.set_attn_processor(original_attn_processors)

        # print("========== Finish inference with concept guidance ==========")

        return images[0]
    
    def custom_ddim_step(self, x0: torch.Tensor, sample: torch.Tensor, t: int) -> torch.Tensor:
        """
        Use the updated x0 to produce x_{t-1}, ignoring the model's original noise in a standard formula.
        sample => x_t
        x0 => updated x_0
        """
        # figure out next_t or prev_t
        timesteps = self.trainer.validation_scheduler.timesteps
        idx = (timesteps == t).nonzero(as_tuple=True)[0].item()
        if idx+1 < len(timesteps):
            next_t = timesteps[idx+1]  # t_{k+1} in the array
        else:
            next_t = 0

        alpha_t = self.trainer.validation_scheduler.alphas_cumprod[t].to(sample.device)
        alpha_t_1 = self.trainer.validation_scheduler.alphas_cumprod[next_t].to(sample.device) if next_t >= 0 else 0

        # compute eps = (x_t - sqrt(alpha_t)* x0 ) / sqrt(1 - alpha_t)
        eps = (sample - alpha_t.sqrt() * x0) / (1 - alpha_t).sqrt()

        # x_{t-1} = sqrt(alpha_{t-1})* x0 + sqrt(1-alpha_{t-1})* eps
        return alpha_t_1.sqrt() * x0 + (1 - alpha_t_1).sqrt() * eps

    def extract_asset_tokens(self, prompt):
        pattern = r"<asset\d+>"
        tokens = re.findall(pattern, prompt)
        if len(tokens) != 4:
            raise ValueError("The number of asset tokens in the prompt must be exactly 4.")
        return tokens

    def pre_porcess_infer_img(self, prompt):
        placeholder_tokens = [['<asset0>', '<asset1>', '<asset2>', '<asset3>'], ['<asset4>', '<asset5>', '<asset6>', '<asset7>']]

        test_dataset = PartComposerSynthDataset(
            instance_data_root=self.trainer.args.instance_data_dir,
            placeholder_tokens=placeholder_tokens,
            tokenizer=self.trainer.tokenizer,
            randomize_unused_mask_areas = True,
            set_bg_white = False,
            synth_type = 0,
            use_all_sythn = False,
            subject_name = 'chair',
        )
        
        tokens = self.extract_asset_tokens(prompt)
        random.shuffle(tokens)
        synth_image = test_dataset.synthesize_test_img(tokens)

        return synth_image

    def log_classifier_segmenter_outputs(
            self,
            logs,
            synth_data,  # Tuple: (cls_labels, logits_cls, gt_mask, logits_mask) for synthetic image
            inst_data,   # Tuple: (cls_labels, logits_cls, gt_mask, logits_mask) for instance image
            concept_names=None
        ):
        """
        Logs two images (synthetic and instance) where each image shows:
        - Row 1: GT classification labels (as text)
        - Row 2: Predicted classification scores (sigmoid applied, as text)
        - Row 3: GT segmentation masks (grid)
        - Row 4: Predicted segmentation masks (grid)
        The resulting images are stored in the logs dict.
        """
        from PIL import Image, ImageDraw, ImageFont
        import torchvision
        import io
        import wandb

        def text_from_labels(cls_labels):
            active_idx = (cls_labels > 0.5).nonzero(as_tuple=True)[0]
            if concept_names is not None:
                return ", ".join([concept_names[i] for i in active_idx])
            else:
                return " ".join([f"<asset{i}>" for i in active_idx])
        
        def text_from_logits(logits_cls):
            scores = torch.sigmoid(logits_cls).detach().cpu().numpy()
            text_scores = []
            for i, s in enumerate(scores.flatten()):
                if concept_names is not None:
                    text_scores.append(f"{concept_names[i]}={s:.2f}")
                else:
                    text_scores.append(f"<asset{i}>={s:.2f}")
            return ", ".join(text_scores)
        
        def create_mask_grid(gt_mask, logits_mask):
            gt_mask_grid = torchvision.utils.make_grid(gt_mask.unsqueeze(1),
                                                    nrow=gt_mask.size(0), padding=2, normalize=True)
            pred_mask_grid = torchvision.utils.make_grid(torch.sigmoid(logits_mask).unsqueeze(1),
                                                        nrow=logits_mask.size(0), padding=2, normalize=True)
            return gt_mask_grid, pred_mask_grid
        
        def create_text_image(text, width, height=50):
            img = Image.new("RGB", (width, height), color=(255, 255, 255))
            draw = ImageDraw.Draw(img)
            font = ImageFont.load_default()
            draw.text((10, 10), text, fill=(0, 0, 0), font=font)
            return img

        def create_final_image(cls_labels, logits_cls, gt_mask, logits_mask):
            gt_text = "GT Cls: " + text_from_labels(cls_labels)
            pred_text = "Pred Cls: " + text_from_logits(logits_cls)
            text_img_gt = create_text_image(gt_text, 600)
            text_img_pred = create_text_image(pred_text, 600)
            gt_mask_grid, pred_mask_grid = create_mask_grid(gt_mask, logits_mask)
            gt_mask_pil = torchvision.transforms.ToPILImage()(gt_mask_grid)
            pred_mask_pil = torchvision.transforms.ToPILImage()(pred_mask_grid)
            w = max(text_img_gt.width, text_img_pred.width, gt_mask_pil.width, pred_mask_pil.width)
            h_total = text_img_gt.height + text_img_pred.height + gt_mask_pil.height + pred_mask_pil.height
            out_img = Image.new("RGB", (w, h_total), (255, 255, 255))
            offset = 0
            for im in [text_img_gt, text_img_pred, gt_mask_pil, pred_mask_pil]:
                out_img.paste(im, (0, offset))
                offset += im.height

            # --- Set DPI to 300 ---
            # Save the image to a BytesIO buffer with 300 dpi and reload it.
            buf = io.BytesIO()
            out_img.save(buf, format="PNG", dpi=(300, 300))
            buf.seek(0)
            out_img_with_dpi = Image.open(buf)
            return out_img_with_dpi

        # Unpack data for synthetic and instance images
        synth_img = create_final_image(*synth_data)
        inst_img = create_final_image(*inst_data)
        logs["concept_pred_seg_synth"] = wandb.Image(synth_img)
        logs["concept_pred_seg_inst"] = wandb.Image(inst_img)

    @torch.no_grad()
    def perform_MuDI_inference(self, instance_prompt, path=None, guidance_scale=7.5, seed=None, gamma=1.0):
        # Backup the original attention processors
        original_attn_processors = self.trainer.unet.attn_processors

        # Reset the attention controller
        self.trainer.controller.attention_store = {}
        self.trainer.controller.cur_step = 0

        # # Attach the attention controller to the unet
        # self.trainer.register_attention_control(self.trainer.controller)

        self.trainer.unet.eval()
        if self.trainer.args.train_text_encoder:
            self.trainer.text_encoder.eval()

        ### Currently using seed to generate noise
        torch.manual_seed(seed)
        # latents = torch.randn((1, 4, 64, 64), device=self.trainer.accelerator.device)
        # --- New Steps for MuDI initialization ---
        # Extract tokens and prepare synthesized test image and masks
        tokens = self.extract_asset_tokens(instance_prompt)
        synth_image, synth_masks = self.pre_porcess_infer_img(instance_prompt)
        # synth_image: [3,H,W], normalized to [-1,1]
        # synth_masks: [N,1,H,W], each in [0,1]

        device = self.trainer.accelerator.device
        synth_image = synth_image.unsqueeze(0).to(device)  # [1,3,H,W]
        # Save and visualize the synth_image
        synth_image_np = (synth_image.squeeze(0).permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5).clip(0, 1)
        synth_image_pil = Image.fromarray((synth_image_np * 255).astype(np.uint8))
        # synth_image_pil.save("synth_image.png")

        # Merge all masks into a single composite mask M_init
        M_init = (synth_masks.sum(dim=0) > 0).float().to(device)  # [1,H,W]
        M_init = M_init.unsqueeze(0)  # [1,1,H,W]

        # Encode x_init with VAE encoder E(x_init)
        synth_image = synth_image.to(self.trainer.weight_dtype)
        init_latent_dist = self.trainer.vae.encode(synth_image)
        init_latent = init_latent_dist.latent_dist.sample() * 0.18215  # Scale latents
        # # Save and visualize the init_latent
        # init_latent_np = (init_latent.squeeze(0).permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5).clip(0, 1)
        # init_latent_pil = Image.fromarray((init_latent_np * 255).astype(np.uint8))
        # init_latent_pil.save("init_latent.png")

        # Resize M_init to latent resolution
        latent_h, latent_w = init_latent.shape[-2], init_latent.shape[-1]
        M_init_resized = torch.nn.functional.interpolate(M_init, size=(latent_h, latent_w), mode='nearest')

        # Create random Gaussian noise epsilon
        epsilon = torch.randn_like(init_latent)

        # Compute z_T = (E(x_init)*M_init)*γ + ε
        latents = init_latent * M_init_resized * gamma + epsilon
        # # Save and visualize the latents
        # latents_np = (latents.squeeze(0).permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5).clip(0, 1)
        # latents_pil = Image.fromarray((latents_np * 255).astype(np.uint8))
        # latents_pil.save("latents.png")
        # --- End of new MuDI-specific initialization steps ---
        
        uncond_input = self.trainer.tokenizer(
            [""],
            padding="max_length",
            max_length=self.trainer.tokenizer.model_max_length,
            return_tensors="pt",
        ).to(self.trainer.accelerator.device)

        input_ids = self.trainer.tokenizer(
            [instance_prompt],
            padding="max_length",
            truncation=True,
            max_length=self.trainer.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.to(self.trainer.accelerator.device)
        cond_embeddings = self.trainer.text_encoder(input_ids)[0]
        uncond_embeddings = self.trainer.text_encoder(uncond_input.input_ids)[0]
        text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])

        for t in self.trainer.validation_scheduler.timesteps:
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.trainer.validation_scheduler.scale_model_input(
                latent_model_input, timestep=t
            )

            pred = self.trainer.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            )
            noise_pred = pred.sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            latents = self.trainer.validation_scheduler.step(noise_pred, t, latents).prev_sample
        latents = 1 / 0.18215 * latents

        images = self.trainer.vae.decode(latents.to(self.trainer.weight_dtype)).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).round().astype("uint8")

        self.trainer.unet.train()
        if self.trainer.args.train_text_encoder:
            self.trainer.text_encoder.train()

        # # Restore the original attention processors
        self.trainer.unet.set_attn_processor(original_attn_processors)

        # Image.fromarray(images[0]).save(path)
        img = images[0]

        combined_width = 512
        combined_height = 512 * 2

        combined_image = Image.new("RGB", (combined_width, combined_height))
        combined_image.paste(synth_image_pil.resize((512, 512)), (0, 0))
        bottom_image = Image.fromarray(img).resize((512, 512))
        combined_image.paste(bottom_image, (0, 512))
        
        return combined_image

    @torch.no_grad()
    def save_cross_attention_vis(self, prompt, attention_maps, path=None):
        '''
        Save per-step attention maps for a given prompt.
        '''
        tokens = self.trainer.tokenizer.encode(prompt)
        images = []
        for i in range(len(tokens)):
            # print(f"tokens[i]: {tokens[i]}")
            image = attention_maps[:, :, i]
            image = 255 * image / image.max()
            image = image.unsqueeze(-1).expand(*image.shape, 3)
            image = image.numpy().astype(np.uint8)
            image = np.array(Image.fromarray(image).resize((256, 256)))
            image = ptp_utils.text_under_image(
                image, self.trainer.tokenizer.decode(int(tokens[i]))
            )
            images.append(image)
        vis = ptp_utils.view_images(np.stack(images, axis=0))
        # vis.save(path)
        return vis

    def val_inference(self, batch, agg_attn, global_step, logs):
        ### Plot the last step attention
        if self.trainer.args.log_last_step_atten and self.trainer.args.lambda_attention != 0:
            self.trainer.controller.cur_step = 1
            last_sentence = batch["input_ids"][1]
            last_sentence = last_sentence[
                (last_sentence != 0)
                & (last_sentence != 49406)
                & (last_sentence != 49407)
            ]
            last_sentence = self.trainer.tokenizer.decode(last_sentence)

            attention_image = self.save_cross_attention_vis(
                last_sentence,
                attention_maps=agg_attn.detach().cpu()
            )
            logs["last_step_attention"] = wandb.Image(attention_image, caption=f"Step {global_step} Attention")
        self.trainer.controller.cur_step = 0
        self.trainer.controller.attention_store = {}

        ### Perform inference on all training data and plot attention                        
        generated_images = []
        attention_images = []
        for prompt_id, instant_prompt in enumerate(self.trainer.args.instance_prompts):
            for i in range(5):
                if not self.trainer.args.use_MuDI_inference:
                    if self.trainer.args.concept_predictor_guide_inference:
                        img = self.perform_concept_guidance_inference(instant_prompt, seed=i)
                    else:
                        img = self.perform_full_inference(instant_prompt, seed=i)
                else:
                    img = self.perform_MuDI_inference(instant_prompt, seed=i)
                generated_images.append(wandb.Image(img, caption=f"{instant_prompt}_sample_{i}"))
                
                if self.trainer.args.log_inference_atten:
                    full_agg_attn = self.trainer.aggregate_attention_batch(
                        res=16, from_where=("up", "down"), is_cross=True, select=1, batch_size=2
                    )
                    attn_image = self.save_cross_attention_vis(
                        instant_prompt,
                        attention_maps=full_agg_attn.detach().cpu()
                    )
                    attention_images.append(wandb.Image(attn_image, caption=f"{instant_prompt}_sample_{i}"))

        logs["Val construct original"] = generated_images
        if self.trainer.args.log_inference_atten:
            logs["Val construct original attention"] = attention_images

        ### Perform inference on all validation prompts and plot attention 
        val_generated_images = []
        val_attention_images = []
        for prompt_id, val_prompt in enumerate(self.trainer.args.val_mix_prompts):
            for i in range(5):
                if not self.trainer.args.use_MuDI_inference:
                    if self.trainer.args.concept_predictor_guide_inference:
                        img = self.perform_concept_guidance_inference(val_prompt, seed=i)
                    else:
                        img = self.perform_full_inference(val_prompt, seed=i)
                else:
                    img = self.perform_MuDI_inference(val_prompt, seed=i)
                val_generated_images.append(wandb.Image(img, caption=f"{val_prompt}_sample_{i}"))
                
                if self.trainer.args.log_inference_atten:
                    full_agg_attn = self.trainer.aggregate_attention_batch(
                        res=16, from_where=("up", "down"), is_cross=True, select=1, batch_size=2
                    )
                    attn_image = self.save_cross_attention_vis(
                        val_prompt,
                        attention_maps=full_agg_attn.detach().cpu()
                    )
                    val_attention_images.append(wandb.Image(attn_image, caption=f"{val_prompt}_sample_{i}"))
        
        logs["Val part mixing"] = val_generated_images
        if self.trainer.args.log_inference_atten:
            logs["Val part mixing attention"] = val_attention_images

        return logs

    def final_inference(self, logs):
        inference_generated_images = []
        inference_attention_images = []
        for prompt_id, instant_prompt in enumerate(self.trainer.args.final_inference_prompts):
            for i in range(5):
                if not self.trainer.args.use_MuDI_inference:
                    img = self.perform_full_inference(instant_prompt, seed=i)
                else:
                    img = self.perform_MuDI_inference(instant_prompt, seed=i)
                inference_generated_images.append(wandb.Image(img, caption=f"{instant_prompt}_sample_{i}"))
                
                if self.trainer.args.log_inference_atten:
                    full_agg_attn = self.trainer.aggregate_attention_batch(
                        res=16, from_where=("up", "down"), is_cross=True, select=1, batch_size=2
                    )

                    attn_image = self.save_cross_attention_vis(
                        instant_prompt,
                        attention_maps=full_agg_attn.detach().cpu()
                    )

                    inference_attention_images.append(wandb.Image(attn_image, caption=f"{instant_prompt}_sample_{i}"))
        logs["Final Inference Generated Images"] = inference_generated_images
        if self.trainer.args.log_inference_atten:
            logs["Final Inference Attention Images"] = inference_attention_images

        return logs