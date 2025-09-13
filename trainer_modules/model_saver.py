import os
import torch

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    UNet2DConditionModel,
    DDIMScheduler,
)

class ModelSaver:
    def __init__(self, trainer, logger):
        self.trainer = trainer
        self.logger = logger

    ### Save models
    def save_pipeline(self, path):
        self.trainer.accelerator.wait_for_everyone()
        if self.trainer.accelerator.is_main_process:
            pipeline = DiffusionPipeline.from_pretrained(
                self.trainer.args.pretrained_model_name_or_path,
                unet=self.trainer.accelerator.unwrap_model(self.trainer.unet),
                text_encoder=self.trainer.accelerator.unwrap_model(self.trainer.text_encoder),
                tokenizer=self.trainer.tokenizer,
                revision=self.trainer.args.revision,
            )
            pipeline.save_pretrained(path)

    def save_pipeline_lora(self, path):
        '''
        Save the LoRA weights, the text_encoder, and the tokenizer without affecting training.
        '''
        self.trainer.accelerator.wait_for_everyone()
        if self.trainer.accelerator.is_main_process:
            # # Backup the original attention processors and device
            original_attn_processors = self.trainer.unet.attn_processors
            original_device = next(self.trainer.unet.parameters()).device

            # # Reset attention processors to standard LoRACrossAttnProcessor
            # self.trainer.reset_attention_processors(self.trainer.unet)
            
            # Move unet to CPU for saving
            self.trainer.unet.to('cpu')
            self.trainer.unet.to(torch.float32)

            # Save the LoRA weights
            unet = self.trainer.accelerator.unwrap_model(self.trainer.unet)
            unet.save_attn_procs(path)
            self.logger.info(f"Saved LoRA parameters to {path}")

            # Save the text_encoder
            text_encoder = self.trainer.accelerator.unwrap_model(self.trainer.text_encoder)
            text_encoder.to(torch.float32)
            text_encoder.save_pretrained(path)
            self.logger.info(f"Saved text_encoder to {path}")

            # Save the tokenizer
            self.trainer.tokenizer.save_pretrained(path)
            self.logger.info(f"Saved tokenizer to {path}")

            # Save concept predictor weights
            if self.trainer.args.train_concept_predictor:
                concept_predictor = self.trainer.accelerator.unwrap_model(self.trainer.concept_predictor)
                concept_predictor.to(torch.float32)
                torch.save(
                    concept_predictor.state_dict(),
                    os.path.join(path, "pytorch_concept_predictor.bin")
                )
                self.logger.info(f"Saved concept_predictor to {path}")

            # # Restore the original attention processors and move unet back to original device
            self.trainer.unet.set_attn_processor(original_attn_processors)
            self.trainer.unet.to(original_device)