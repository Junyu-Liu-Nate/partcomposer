import torch.nn.functional as F
import torch

from modules.grad_helper import GradNormFunction
from utils.visualization import visualize_combined_masks_64x64, visualize_individual_masks_64x64, visualize_mask_atten_refined

class LossCalculator:
    def __init__(self, trainer):
        self.trainer = trainer

    ### Loss functions
    def mask_diffusion_loss(self, loss, model_pred, target, batch, global_step, logs):
        ##### Mask diffusion loss #####
        # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
        model_pred_synth, model_pred = torch.chunk(model_pred, 2, dim=0)
        target_synth, target = torch.chunk(target, 2, dim=0)

        if self.trainer.args.apply_masked_loss and not self.trainer.args.apply_weighted_masked_loss:
            ### Not weighting the masked loss according to the mask area
            synth_masks = torch.max(
                batch["synth_masks"], axis=1
            ).values
            max_masks = torch.max(
                batch["instance_masks"], axis=1
            ).values
            downsampled_synth_mask = F.interpolate(
                input=synth_masks, size=(64, 64), mode='nearest'
            )
            downsampled_mask = F.interpolate(
                input=max_masks, size=(64, 64), mode='nearest'
            )
            downsampled_synth_mask = (downsampled_synth_mask > 0.1).float()
            downsampled_mask = (downsampled_mask > 0.1).float()

            if self.trainer.args.apply_bg_loss:
                ### Invert the masks to compute unmasked area
                unmasked_synth_mask = 1.0 - downsampled_synth_mask
                unmasked_mask = 1.0 - downsampled_mask

            ### Save the downsampled masks for visualization
            if self.trainer.args.log_masks and self.trainer.args.log_checkpoints and global_step % self.trainer.args.img_log_steps == 0:
                logs = visualize_combined_masks_64x64(downsampled_synth_mask, downsampled_mask, logs)

            # Create per-pixel weighting for the synthetic image
            # If apply_bg_loss is True => 1.0 on the mask, self.trainer.args.bg_loss_weight on background
            # If apply_bg_loss is False => 1.0 on the mask, 0.0 on background
            if self.trainer.args.apply_bg_loss:
                syn_bg_weight = downsampled_synth_mask + (1.0 - downsampled_synth_mask) * self.trainer.args.bg_loss_weight
                inst_bg_weight = downsampled_mask + (1.0 - downsampled_mask) * self.trainer.args.bg_loss_weight
            else:
                syn_bg_weight = downsampled_synth_mask
                inst_bg_weight = downsampled_mask

            # Expand to match [B, C, 64, 64]
            syn_bg_weight = syn_bg_weight.unsqueeze(1)  # [B,1,64,64]
            inst_bg_weight = inst_bg_weight.unsqueeze(1)  # [B,1,64,64]

            # Elementwise MSE
            mse_synth = (model_pred_synth - target_synth) ** 2
            mse_inst = (model_pred - target) ** 2

            # Apply the mask/background weighting
            weighted_mse_synth = mse_synth * syn_bg_weight
            weighted_mse_inst = mse_inst * inst_bg_weight

            # Single combined mean
            diffusion_loss = weighted_mse_synth.mean() + weighted_mse_inst.mean()
            
            logs["diff_loss"] = diffusion_loss.detach().item()
            loss += diffusion_loss
        else:
            print('Invalid loss configuration. Exiting...')
            exit()

        return loss, logs
    
    def mask_attention_loss(self, loss, batch, global_step, logs):
        attn_loss = 0
        batch_size = batch["input_ids"].shape[0]  # Should be 2 after concatenation
        
        batch_mask_attn_visulization = []
        for batch_idx in range(batch_size):
            # Aggregate attention for the current image
            agg_attn = self.trainer.attn_controller.aggregate_attention_batch(
                res=16,
                from_where=("up", "down"),
                is_cross=True,
                select=batch_idx,
                batch_size=batch_size,
            )
            # print(f'agg_attn size: {agg_attn.size()}')
            input_ids = batch["input_ids"][batch_idx]

            # Determine if the current image is synthetic or instance
            if batch_idx == 0:
                # Synthetic image
                if self.trainer.args.apply_max_pooling_on_atten_masks:
                    pooling_factor = 512 // 16
                    GT_masks = F.max_pool2d(
                        input=batch["synth_masks"][0], kernel_size=pooling_factor, stride=pooling_factor
                    )
                else:
                    GT_masks = F.interpolate(
                        input=batch["synth_masks"][0], size=(16, 16), mode='nearest'
                    )
                    GT_masks = (GT_masks > 0.1).float()
                token_ids = batch["synth_token_ids"][0]
            elif batch_idx == 1:
                if batch["is_prior"][0]:  # assuming batch["is_prior"][0] → True means PRIOR image
                    continue  # skip this image entirely
                # Instance image
                if self.trainer.args.apply_max_pooling_on_atten_masks:
                    pooling_factor = 512 // 16
                    GT_masks = F.max_pool2d(
                        input=batch["instance_masks"][0], kernel_size=pooling_factor, stride=pooling_factor
                    )
                else:
                    GT_masks = F.interpolate(
                        input=batch["instance_masks"][0], size=(16, 16), mode='nearest'
                    )
                    GT_masks = (GT_masks > 0.1).float()
                token_ids = batch["token_ids"][0]
            else:
                # Handle additional images if any
                continue

            # Compute attention loss for each mask/token
            mask_attn_visulization = []
            for mask_id in range(len(GT_masks)):
                curr_placeholder_token_id = self.trainer.placeholder_token_ids[
                    token_ids[mask_id]
                ]

                # Find the index of the placeholder token in input_ids
                token_indices = (input_ids == curr_placeholder_token_id).nonzero(as_tuple=True)[0]
                if len(token_indices) == 0:
                    continue  # Token not found in input_ids
                token_index = token_indices.item()

                # Extract the attention map for the token
                asset_attn_mask = agg_attn[..., token_index]  # Shape: [res, res]
                asset_attn_mask = asset_attn_mask / asset_attn_mask.max()

                mask_attn_visulization.append([GT_masks[mask_id, 0], asset_attn_mask])

                # Compute the attention loss
                attn_loss += F.mse_loss(
                    GT_masks[mask_id, 0].float(),
                    asset_attn_mask.float(),
                    reduction="mean",
                )
            batch_mask_attn_visulization.append(mask_attn_visulization)

        ### Visualize the masks and attention maps
        if self.trainer.args.log_atten_maps and self.trainer.args.log_checkpoints and global_step % self.trainer.args.img_log_steps == 0:
            logs = visualize_mask_atten_refined(batch_mask_attn_visulization, logs)

        # Average the attention loss over the batch
        attn_loss = self.trainer.args.lambda_attention * (attn_loss / batch_size)
        logs["attn_loss"] = attn_loss.detach().item()
        loss += attn_loss

        return loss, logs, agg_attn

    def calculate_concept_prediction_loss_classifier(self, loss, batch, noisy_latents, model_pred, timesteps, logs):
        """
        Convert token_ids_batch (e.g. shape [B, num_used_tokens]) into a multi-hot
        label of shape (B, out_dim) and compute BCEWithLogitsLoss.
        
        Returns:
            loss (scalar)
        """
        synth_token_ids = torch.stack(batch["synth_token_ids"])  # Stack the list into a tensor - the batch["synth_token_ids"] is tensor([[]])
        train_token_ids = batch["token_ids"]
        token_ids_list = [synth_token_ids, train_token_ids]

        # The latents feed to Q can be `noisy_latents` or `model_pred`
        if self.trainer.noise_scheduler.config.prediction_type == "epsilon":
            alpha_t = self.trainer.noise_scheduler.alphas_cumprod[timesteps]  # shape [B]
            alpha_t = alpha_t.view(-1, 1, 1, 1)  # expand for (B,1,1,1)

            # True denoised latents x0
            denoised_latents = (noisy_latents - torch.sqrt(1 - alpha_t) * model_pred) / torch.sqrt(alpha_t)
        else:
            raise ValueError("Currently only supporting 'epsilon' prediction type")

        # print(f"denoised_latents: {denoised_latents.shape}")
        
        total_info_loss = 0
        for i, token_ids in enumerate(token_ids_list):
            # if i == 1:
            #     break  # Skip the train image for now to test the performance
            # Add batch dimension for individual latent slices
            current_latent = denoised_latents[i].unsqueeze(0)  # Shape becomes (1, latent_channels, latent_size, latent_size)
            # print(f"current_latent shape: {current_latent.shape}")
            
            # Predict logits
            if self.trainer.args.predictor_type == "classifier":
                predicted_logits = self.trainer.concept_predictor(current_latent)
            elif self.trainer.args.predictor_type == "classifier_time":
                predicted_logits = self.trainer.concept_predictor(current_latent, timesteps[i])

            # Calculate loss
            B = token_ids.size(0)
            labels = torch.zeros(B, self.trainer.total_num_of_assets, device=predicted_logits.device)

            for i in range(B):
                for t_id in token_ids[i]:
                    labels[i, t_id.item()] = 1.0

            # print(f"predicted_logits: {predicted_logits}")
            # print(f"labels: {labels}")

            info_loss = F.binary_cross_entropy_with_logits(predicted_logits, labels)
            total_info_loss += self.trainer.args.concept_pred_weight * info_loss

        logs["info_loss"] = total_info_loss.item()
        loss = loss + total_info_loss

        return loss, logs

    def calculate_concept_prediction_loss_classifier_segmenter(
        self, loss, batch, noisy_latents, model_pred, timesteps, logs, global_step
    ):
        """
        1) BCE multi-label classification for which concepts appear.
        2) BCE segmentation for each concept's location (mask).

        The ConceptClassifierSegmenter returns (logits_cls, logits_mask):
        logits_cls:  (B, out_dim) classification
        logits_mask: (B, out_dim, 64, 64) segmentation
        """
        # -------------------------------------------------
        # 1) Denoise latents x0 (assuming "epsilon" type)
        # -------------------------------------------------
        if self.trainer.noise_scheduler.config.prediction_type == "epsilon":
            alpha_t = self.trainer.noise_scheduler.alphas_cumprod[timesteps]  # shape [2]
            alpha_t = alpha_t.view(-1, 1, 1, 1)                       # => [2,1,1,1]
            denoised_latents = (noisy_latents - torch.sqrt(1 - alpha_t) * model_pred) / torch.sqrt(alpha_t)
        else:
            raise ValueError("Currently only supporting 'epsilon' prediction type")

        # -------------------------------------------------
        # 2) Two images: synthetic [0], instance [1]
        # -------------------------------------------------
        synth_token_ids = torch.stack(batch["synth_token_ids"])  # shape [1, #synthTokens]
        train_token_ids = batch["token_ids"]                     # shape [1, #instanceTokens]
        token_ids_list = [synth_token_ids, train_token_ids]

        # -------------------------------------------------
        # 3) Ground-truth masks
        # -------------------------------------------------
        synth_masks = batch["synth_masks"]  # e.g. shape [1,4,H,W] if 4 concepts used
        instance_masks = batch["instance_masks"]
        # print(f"synth_masks shape (raw): {synth_masks.shape}")
        # print(f"instance_masks shape (raw): {instance_masks.shape}")

        # (A) If 5D => squeeze out dimension 2 => e.g. [1,4,1,H,W] -> [1,4,H,W]
        if synth_masks.ndim == 5 and synth_masks.shape[2] == 1:
            synth_masks = synth_masks.squeeze(2)
        if instance_masks.ndim == 5 and instance_masks.shape[2] == 1:
            instance_masks = instance_masks.squeeze(2)

        # (B) Downsample to 64×64 if needed
        if synth_masks.shape[-1] != 64:
            synth_masks = F.interpolate(synth_masks, size=(64, 64), mode='nearest')
        if instance_masks.shape[-1] != 64:
            instance_masks = F.interpolate(instance_masks, size=(64, 64), mode='nearest')

        # (C) Binarize => now shape [1, #usedChannels, 64, 64]
        synth_masks = (synth_masks > 0.1).float()
        instance_masks = (instance_masks > 0.1).float()

        # (D) Do the alignment in the loop, below (no simple pad).

        if self.trainer.args.predictor_type == "classifier_seg_time_film":
            out_dim = self.trainer.concept_predictor.cls_out.out_features
        else:
            out_dim = self.trainer.concept_predictor.out.out_features
        # out_dim = self.trainer.concept_predictor.out.out_features  # e.g. 8
        # out_dim = self.trainer.concept_predictor.cls_out.out_features

        # -------------------------------------------------
        # 4) Classification + segmentation
        # -------------------------------------------------
        total_info_loss = 0.0

        # Process [0] => synthetic, [1] => instance
        # Each half has its own token_ids + partial GT mask
        for i, token_ids in enumerate(token_ids_list):
            # (A) The latents for this image => shape [1,4,64,64]
            current_latent = denoised_latents[i].unsqueeze(0)

            # Predict classification + segmentation
            if self.trainer.args.predictor_type == "classifier_seg":
                logits_cls, logits_mask = self.trainer.concept_predictor(current_latent)
            elif self.trainer.args.predictor_type in ["classifier_seg_time", "classifier_seg_time_film"]:
                logits_cls, logits_mask = self.trainer.concept_predictor(current_latent, timesteps[i])
            
            # logits_cls => [1, out_dim], logits_mask => [1, out_dim, 64, 64]

            # ----- Classification Loss -----
            B = token_ids.size(0)  # typically 1
            cls_labels = torch.zeros(B, out_dim, device=logits_cls.device)
            for b_i in range(B):
                for t_id in token_ids[b_i]:
                    cls_labels[b_i, t_id.item()] = 1.0
            cls_loss = F.binary_cross_entropy_with_logits(logits_cls, cls_labels)

            # ----- Segmentation GT -----
            if i == 0:
                # shape => [#synthUsed, 64,64]
                partial_mask = synth_masks[0]
            else:
                partial_mask = instance_masks[0]

            # print(f'token_ids: {token_ids}')
            # partial_mask shape => [nUsedChannels, 64, 64]
            # token_ids[i] => which concept indices e.g. [0,2,4,7]
            used_concepts = token_ids[0]  # shape [nUsedChannels], e.g. [0,2,4,7]
            # print(f'used_concepts: {used_concepts}')

            # Create an empty 8-channel GT => shape (out_dim, 64, 64)
            gt_mask_8ch = torch.zeros(
                (out_dim, partial_mask.shape[-2], partial_mask.shape[-1]),
                device=partial_mask.device,
                dtype=partial_mask.dtype,
            )

            # Place each partial channel into the correct concept index
            #   partial_mask[ch_idx] => concept used_concepts[ch_idx]
            for ch_idx, concept_idx in enumerate(used_concepts):
                gt_mask_8ch[concept_idx.item()] = partial_mask[ch_idx]

            # Add a batch dimension => [1, out_dim, 64, 64]
            gt_mask_8ch = gt_mask_8ch.unsqueeze(0)

            seg_loss = F.binary_cross_entropy_with_logits(logits_mask, gt_mask_8ch)
            # print(f'cls_loss: {cls_loss}, seg_loss: {seg_loss}')

            if self.trainer.args.apply_grad_norm:
                cls_loss = GradNormFunction.apply(cls_loss)
                seg_loss = GradNormFunction.apply(seg_loss)
                combined_loss = cls_loss + seg_loss
            else:
                combined_loss = cls_loss + self.trainer.args.concept_pred_seg_scale * seg_loss
            
            ### Apply time-based weighting:
            if self.trainer.args.use_time_weight:
                # alpha_t shape => [B,1,1,1], so alpha_t[i,0,0,0] is a scalar
                weight_i = torch.sqrt(alpha_t[i,0,0,0]).detach()
                weighted_loss = combined_loss * weight_i
                # logs[f"time_weight_{i}"] = weight_i.item()  # optional logging
            else:
                weighted_loss = combined_loss
            
            # total_info_loss += self.trainer.args.concept_pred_weight * combined_loss
            total_info_loss += self.trainer.args.concept_pred_weight * weighted_loss

            if self.trainer.args.log_concept_predictor and global_step % self.trainer.args.img_log_steps == 0:
                # Save outputs for logging
                if i == 0:
                    synth_logging = (cls_labels[0], logits_cls[0], gt_mask_8ch[0], logits_mask[0])
                else:
                    inst_logging = (cls_labels[0], logits_cls[0], gt_mask_8ch[0], logits_mask[0])
            
        # -------------------------------------------------
        # 5) Accumulate + log
        # -------------------------------------------------
        logs["info_loss"] = total_info_loss.item()
        loss = loss + total_info_loss

        # If logging data is available, call the logging function.
        if self.trainer.args.log_concept_predictor and global_step % self.trainer.args.img_log_steps == 0:
            if synth_logging is not None and inst_logging is not None:
                # Pass self.trainer.concept_names if it exists; otherwise, use None.
                concept_names = getattr(self, "concept_names", None)
                self.trainer.inference_visualizer.log_classifier_segmenter_outputs(logs, synth_logging, inst_logging, concept_names)

        return loss, logs