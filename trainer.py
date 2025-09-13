"""
This script is adapted from the Break-a-Scene codebase. We follow a similar
fintuning structure but with our heavily customized modules.
We keep their original license below.
"""
"""
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import itertools
import logging
import math
import os
from pathlib import Path
import torch
import math
import torch
import numpy as np
from datetime import datetime
from tqdm.auto import tqdm
import torch.nn.functional as F

import datasets
import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version

from modules.concept_predictor import ConceptClassifier, ConceptClassifierSegmenter, ConceptClassifierWithTime, ConceptClassifierSegmenterWithTime, ConceptClassifierSegmenterWithTimeFiLM, ConceptRegressor
from modules.grad_helper import GradNormFunction
import modules.ptp_utils as ptp_utils
from modules.ptp_utils import AttentionStore
from modules.cross_attn_processor import P2PCrossAttnProcessor, P2PCrossAttnProcessorWithLoRA

from utils.load_model import import_model_class_from_model_name_or_path
from utils.dataset import PartComposerSynthDataset
from utils.arg_parser import parse_args
from utils.visualization import visualize_combined_masks_64x64, visualize_individual_masks_64x64, visualize_mask_atten_refined

from trainer_modules.setup import Setup
from trainer_modules.attn_controller import AttnController
from trainer_modules.loss_calculator import LossCalculator
from trainer_modules.model_saver import ModelSaver
from trainer_modules.inference_visualizer import InferenceVisualizer
from trainer_modules.monitor import Monitor

check_min_version("0.12.0")

logger = get_logger(__name__)

class PartComposer:
    def __init__(self):
        self.args = parse_args()
        
        self.setup = Setup(self, logger)
        self.attn_controller = AttnController(self)
        self.loss_calculator = LossCalculator(self)
        self.model_saver = ModelSaver(self, logger)
        self.inference_visualizer = InferenceVisualizer(self)
        self.monitor = Monitor(self)

        self.main()

    def main(self):
        date_time_string = self.setup.set_logging_dir()

        ### Setup
        self.setup.load_pretrained_models()
        self.setup.add_initialize_new_tokens()
        if self.args.use_bg_tokens:
            self.setup.add_initialize_bg_tokens()
        
        ### Specify the prompts used to inference the reconstruction of training images
        self.args.instance_prompts = []
        for placeholder_tokens in self.placeholder_tokens:
            self.args.instance_prompts.append(
                "a photo of a " + self.args.subject_name + " with " + " and ".join(placeholder_tokens)
            )
            if self.args.use_bg_tokens:
                self.args.instance_prompts[-1] += ", on a <bg0> background"
            elif self.args.set_bg_white:
                self.args.instance_prompts[-1] += ", on a simple white background"
        logger.info(f"Validation prompts to reconstruct the input images: {self.args.instance_prompts}")

        ### Initialize the concept predictor
        if self.args.train_concept_predictor:
            self.setup.init_concept_predictor()

        ### Stage 1 training: start by only optimizing the embeddings + concept predictor
        if self.args.train_concept_predictor:
            params_to_optimize = list(self.text_encoder.get_input_embeddings().parameters()) + list(self.concept_predictor.parameters())
        else:
            params_to_optimize = self.text_encoder.get_input_embeddings().parameters()
        optimizer_class = self.setup.set_optimize_info()
        optimizer = optimizer_class(
            params_to_optimize,
            lr=self.args.initial_learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            weight_decay=self.args.adam_weight_decay,
            eps=self.args.adam_epsilon,
        )

        ### Create dataset and dataloaders
        train_dataset = PartComposerSynthDataset(
            instance_data_root=self.args.instance_data_dir,
            placeholder_tokens=self.placeholder_tokens,
            use_bg_tokens=self.args.use_bg_tokens,
            bg_data_root=self.args.bg_data_dir,
            bg_placeholder_tokens=self.bg_placeholder_tokens if self.args.use_bg_tokens else None,
            use_prior_data=self.args.use_prior_data,
            prior_data_root=self.args.prior_data_dir,
            prior_prob=self.args.prior_prob,
            tokenizer=self.tokenizer,
            size=self.args.resolution,
            center_crop=self.args.center_crop,
            randomize_unused_mask_areas = self.args.randomize_unused_mask_areas,
            set_bg_white = self.args.set_bg_white,
            sample_type = self.args.sample_type,
            synth_type = self.args.synth_type,
            use_all_sythn = self.args.use_all_synth_imgs,
            use_all_instance = self.args.use_all_instance,
            subject_name = self.args.subject_name,
            train_detailed_prompt = self.args.train_detailed_prompt,
            sythn_detailed_prompt = self.args.sythn_detailed_prompt,
        )
        
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=True,
            collate_fn=lambda examples: train_dataset.collate_fn(examples),
            num_workers=self.args.dataloader_num_workers,
        )

        ### Set scheduler info
        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / self.args.gradient_accumulation_steps
        )
        if self.args.max_train_steps is None:
            self.args.max_train_steps = (
                self.args.num_train_epochs * num_update_steps_per_epoch
            )
            overrode_max_train_steps = True

        lr_scheduler = get_scheduler(
            self.args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.args.lr_warmup_steps
            * self.args.gradient_accumulation_steps,
            num_training_steps=self.args.max_train_steps
            * self.args.gradient_accumulation_steps,
            num_cycles=self.args.lr_num_cycles,
            power=self.args.lr_power,
        )

        (
            self.unet,
            self.text_encoder,
            optimizer,
            train_dataloader,
            lr_scheduler,
        ) = self.accelerator.prepare(
            self.unet, self.text_encoder, optimizer, train_dataloader, lr_scheduler
        )

        # For mixed precision training cast the text_encoder and vae weights to half-precision
        # as these models are only used for inference, keeping weights in full precision is not required.
        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

        # Move vae and text_encoder to device and cast to weight_dtype
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)

        low_precision_error_string = (
            "Please make sure to always have all model weights in full float32 precision when starting training - even if"
            " doing mixed precision training. copy of the weights should still be float32."
        )

        if self.accelerator.unwrap_model(self.unet).dtype != torch.float32:
            raise ValueError(
                f"Unet loaded as datatype {self.accelerator.unwrap_model(self.unet).dtype}. {low_precision_error_string}"
            )

        if (
            self.args.train_text_encoder
            and self.accelerator.unwrap_model(self.text_encoder).dtype != torch.float32
        ):
            raise ValueError(
                f"Text encoder loaded as datatype {self.accelerator.unwrap_model(self.text_encoder).dtype}."
                f" {low_precision_error_string}"
            )

        # Need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / self.args.gradient_accumulation_steps
        )
        if overrode_max_train_steps:
            self.args.max_train_steps = (
                self.args.num_train_epochs * num_update_steps_per_epoch
            )
        # Afterwards recalculate the number of training epochs
        self.args.num_train_epochs = math.ceil(
            self.args.max_train_steps / num_update_steps_per_epoch
        )

        # Initialize the trackers to use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if self.accelerator.is_main_process:
            init_kwargs = {}
            if self.args.report_to == "wandb":
                init_kwargs["wandb"] = {
                    "name": self.args.wandb_run_name + '_' + date_time_string
                }

            self.accelerator.init_trackers('break-a-scene', config=vars(self.args), init_kwargs=init_kwargs)

        ### Train
        total_batch_size = (
            self.args.train_batch_size
            * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
        logger.info(f"  Num Epochs = {self.args.num_train_epochs}")
        logger.info(
            f"  Instantaneous batch size per device = {self.args.train_batch_size}"
        )
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
        )
        logger.info(
            f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}"
        )
        logger.info(f"  Total optimization steps = {self.args.max_train_steps}")
        global_step = 0
        first_epoch = 0

        ### Potentially load in the weights and states from a previous save
        if self.args.resume_from_checkpoint:
            first_epoch, resume_step = self.setup.load_trained_weights(first_epoch, num_update_steps_per_epoch)

        ### Only show the progress bar once on each machine.
        progress_bar = tqdm(
            range(global_step, self.args.max_train_steps),
            disable=not self.accelerator.is_local_main_process,
        )
        progress_bar.set_description("Steps")

        ### keep original embeddings as reference
        orig_embeds_params = (
            self.accelerator.unwrap_model(self.text_encoder)
            .get_input_embeddings()
            .weight.data.clone()
        )

        ### Create attention controller
        self.controller = AttentionStore()
        self.attn_controller.register_attention_control(self.controller)

        for epoch in range(first_epoch, self.args.num_train_epochs):
            self.unet.train()
            if self.args.train_text_encoder:
                self.text_encoder.train()
            for step, batch in enumerate(train_dataloader):
                if self.args.phase1_train_steps == global_step:
                    ### Start stage 2 training, add and update LoRA layers
                    self.setup.set_lora_layers()

                    ### Set optimizer to update LoRA + token embeddings + concept predictor
                    ### The official implementation updates the whole text_encoder when setting train_text_encoder to True, which is the default setting
                    if self.args.train_concept_predictor:
                        params_to_optimize = (
                            itertools.chain(self.lora_layers.parameters(), self.text_encoder.parameters(), self.concept_predictor.parameters())
                            if self.args.train_text_encoder
                            else itertools.chain(
                                self.lora_layers.parameters(),
                                self.text_encoder.get_input_embeddings().parameters(),
                                self.concept_predictor.parameters()
                            )
                        )
                    else:
                        params_to_optimize = (
                            itertools.chain(self.lora_layers.parameters(), self.text_encoder.parameters())
                            if self.args.train_text_encoder
                            else itertools.chain(
                                self.lora_layers.parameters(),
                                self.text_encoder.get_input_embeddings().parameters(),
                            )
                        )

                    del optimizer
                    optimizer = optimizer_class(
                        params_to_optimize,
                        lr=self.args.learning_rate,
                        betas=(self.args.adam_beta1, self.args.adam_beta2),
                        weight_decay=self.args.adam_weight_decay,
                        eps=self.args.adam_epsilon,
                    )
                    del lr_scheduler
                    lr_scheduler = get_scheduler(
                        self.args.lr_scheduler,
                        optimizer=optimizer,
                        num_warmup_steps=self.args.lr_warmup_steps
                        * self.args.gradient_accumulation_steps,
                        num_training_steps=self.args.max_train_steps
                        * self.args.gradient_accumulation_steps,
                        num_cycles=self.args.lr_num_cycles,
                        power=self.args.lr_power,
                    )

                    self.lora_layers, optimizer, lr_scheduler = self.accelerator.prepare(
                        self.lora_layers, optimizer, lr_scheduler
                    )

                logs = {}

                ### Skip steps until reaching the resumed step
                if (
                    self.args.resume_from_checkpoint
                    and epoch == first_epoch
                    and step < resume_step
                ):
                    if step % self.args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                    continue
                
                ### Key trainings step
                with self.accelerator.accumulate(self.unet):
                    ##### Diffusion Denoise Process #####
                    # Convert images to latent space
                    latents = self.vae.encode(
                        batch["pixel_values"].to(dtype=self.weight_dtype)
                    ).latent_dist.sample()
                    latents = latents * 0.18215

                    # Sample noise that to add to the latents
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(
                        0,
                        self.noise_scheduler.config.num_train_timesteps,
                        (bsz,),
                        device=latents.device,
                    )
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = self.noise_scheduler.add_noise(
                        latents, noise, timesteps
                    )

                    # Get the text embedding for conditioning
                    encoder_hidden_states = self.text_encoder(batch["input_ids"])[0]

                    # Predict the noise residual
                    model_pred = self.unet(
                        noisy_latents, timesteps, encoder_hidden_states
                    ).sample

                    # Get the target for loss depending on the prediction type
                    if self.noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif self.noise_scheduler.config.prediction_type == "v_prediction":
                        target = self.noise_scheduler.get_velocity(
                            latents, noise, timesteps
                        )
                    else:
                        raise ValueError(
                            f"Unknown prediction type {self.noise_scheduler.config.prediction_type}"
                        )

                    loss = 0
                    ###### Mask diffusion loss ######
                    loss, logs = self.loss_calculator.mask_diffusion_loss(loss, model_pred, target, batch, global_step, logs)

                    ###### Attention loss ######
                    if self.args.lambda_attention != 0:
                        loss, logs, agg_attn = self.loss_calculator.mask_attention_loss(loss, batch, global_step, logs)

                    ###### Concept Prediction Loss ######
                    if self.args.train_concept_predictor:
                        if self.args.predictor_type in ["classifier", "classifier_time"]:
                            loss, logs = self.loss_calculator.calculate_concept_prediction_loss_classifier(loss, batch, noisy_latents, model_pred, timesteps, logs)
                        elif self.args.predictor_type in ["classifier_seg", "classifier_seg_time", "classifier_seg_time_film"]:
                            loss, logs = self.loss_calculator.calculate_concept_prediction_loss_classifier_segmenter(loss, batch, noisy_latents, model_pred, timesteps, logs, global_step)
                        elif self.args.predictor_type == "regressor":
                            loss, logs = self.loss_calculator.calculate_concept_prediction_loss_regressor(loss, batch, noisy_latents, model_pred, timesteps, logs)
                        else:
                            raise ValueError(f"Unknown predictor type {self.args.predictor_type}")
                    
                    ###### Prior Preservation Loss ######
                    is_prior = batch["is_prior"][0].item()
                    if is_prior:
                        model_pred_synth, model_pred_inst = torch.chunk(model_pred, 2, dim=0)
                        target_synth, target_inst = torch.chunk(target, 2, dim=0)

                        prior_mse = F.mse_loss(
                            model_pred_inst.float(), target_inst.float(), reduction="mean"
                        )
                        loss = loss + self.args.lambda_prior_loss * prior_mse
                        logs["prior_mse"] = (self.args.lambda_prior_loss * prior_mse.detach().item())

                    self.accelerator.backward(loss)

                    ### No need to keep the attention store
                    self.controller.attention_store = {}
                    self.controller.cur_step = 0

                    ### Gradient clipping
                    if self.accelerator.sync_gradients:
                        params_to_clip = (
                            itertools.chain(
                                self.unet.parameters(), self.text_encoder.parameters()
                            )
                            if self.args.train_text_encoder
                            else self.unet.parameters()
                        )
                        self.accelerator.clip_grad_norm_(
                            params_to_clip, self.args.max_grad_norm
                        )

                    ###### Monitor the learned embeddings and LoRA weights ######
                    logs = self.monitor.monitor_embeddings(logs)
                    if self.args.phase1_train_steps < global_step + 1:
                        logs = self.monitor.monitor_lora(logs)
                    if self.args.train_concept_predictor:
                        logs = self.monitor.monitor_concept_predictor(logs)
                    
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=self.args.set_grads_to_none)

                    ### Make sure don't updating any embedding weights besides the newly added token
                    if self.args.train_text_encoder:
                        if global_step < self.args.phase1_train_steps:
                            with torch.no_grad():
                                self.accelerator.unwrap_model(
                                    self.text_encoder
                                ).get_input_embeddings().weight[: -self.total_num_of_assets] = orig_embeds_params[: -self.total_num_of_assets]
                    else:
                        with torch.no_grad():
                            self.accelerator.unwrap_model(
                                self.text_encoder
                            ).get_input_embeddings().weight[: -self.total_num_of_assets] = orig_embeds_params[: -self.total_num_of_assets]

                ### Save checkpoints and log the results
                ### Checks if the accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    ### Save global step checkpoints
                    # if global_step % self.args.checkpointing_steps == 0:
                    #     if self.accelerator.is_main_process:
                    #         save_path = os.path.join(
                    #             self.args.output_dir, f"checkpoint-{global_step}"
                    #         )
                    #         self.accelerator.save_state(save_path)
                    #         logger.info(f"Saved state to {save_path}")

                    ### Save img_log_steps checkpoints
                    if (
                        self.args.log_checkpoints
                        and global_step % self.args.img_log_steps == 0
                        and global_step > self.args.phase1_train_steps
                    ):
                        ### Save the pipeline (trained weights)
                        if global_step > self.args.phase1_train_steps:
                            ckpts_path = os.path.join(
                                self.args.output_dir, "checkpoints", f"{global_step:05}"
                            )
                            os.makedirs(ckpts_path, exist_ok=True)
                            # self.save_pipeline(ckpts_path)
                            self.model_saver.save_pipeline_lora(ckpts_path)

                        logs = self.inference_visualizer.val_inference(batch, agg_attn, global_step, logs)
                        
                        self.accelerator.log(logs, step=global_step)
                        self.controller.cur_step = 0
                        self.controller.attention_store = {}

                logs["loss"] = loss.detach().item()
                logs["lr"] = lr_scheduler.get_last_lr()[0]
                progress_bar.set_postfix(**logs)
                self.accelerator.log(logs, step=global_step)

                if global_step >= self.args.max_train_steps:
                    break
        
        ### Final inference
        logger.info("Performing final inference...")
        logs = self.inference_visualizer.final_inference(logs)

        self.accelerator.log(logs, step=global_step)
        logger.info("Final inference done.")

        self.model_saver.save_pipeline_lora(self.args.output_dir)

        self.accelerator.end_training()

if __name__ == "__main__":
    PartComposer()