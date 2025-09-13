import argparse
import os
import re
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from diffusers import (
    StableDiffusionPipeline,
    DiffusionPipeline,
    DDIMScheduler,
    DDPMScheduler,
    UNet2DConditionModel,
    AutoencoderKL,
    DDIMScheduler,
)
from diffusers.models.cross_attention import LoRACrossAttnProcessor
from diffusers.utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer

from utils.dataset import PartComposerSynthDataset
from modules.cross_attn_processor import P2PCrossAttnProcessor, P2PCrossAttnProcessorWithLoRA
from modules.concept_predictor import ConceptClassifier, ConceptClassifierSegmenter, ConceptClassifierWithTime, ConceptClassifierSegmenterWithTime, ConceptClassifierSegmenterWithTimeFiLM

class ModelLoader:
    def __init__(self, inferencer):
        self.inferencer = inferencer

    def _load_pipeline_lora(self):
        """
        Loads the LoRA weights from self.inferencer.args.model_path and sets up
        UNet, VAE, text_encoder, tokenizer, pipeline, plus a ddim scheduler.
        """
        # Load base model components
        self.inferencer.unet = UNet2DConditionModel.from_pretrained(
            self.inferencer.args.pretrained_model_name_or_path,
            subfolder="unet",
            revision=self.inferencer.args.revision,
            torch_dtype=torch.float16,
        )
        # self.inferencer.unet.enable_gradient_checkpointing()
        # if is_xformers_available():
        #     self.inferencer.unet.enable_xformers_memory_efficient_attention()
        #     print("Using xFormers memory-efficient attention.")

        self.inferencer.vae = AutoencoderKL.from_pretrained(
            self.inferencer.args.pretrained_model_name_or_path,
            subfolder="vae",
            revision=self.inferencer.args.revision,
            torch_dtype=torch.float16,
        )

        self.inferencer.text_encoder = CLIPTextModel.from_pretrained(
            self.inferencer.args.model_path,
            revision=self.inferencer.args.revision,
            torch_dtype=torch.float16,
        )

        self.inferencer.tokenizer = CLIPTokenizer.from_pretrained(
            self.inferencer.args.model_path,
            revision=self.inferencer.args.revision,
        )

        # Resize token embeddings in the text_encoder
        self.inferencer.text_encoder.resize_token_embeddings(len(self.inferencer.tokenizer))

        # Create the pipeline using StableDiffusionPipeline
        self.inferencer.pipeline = StableDiffusionPipeline(
            unet=self.inferencer.unet,
            vae=self.inferencer.vae,
            text_encoder=self.inferencer.text_encoder,
            tokenizer=self.inferencer.tokenizer,
            scheduler=DDIMScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                clip_sample=False,
                set_alpha_to_one=False,
            ),
            safety_checker=None,
            feature_extractor=None,
        )
        # self.inferencer.pipeline = StableDiffusionPipeline(
        #     unet=self.inferencer.unet,
        #     vae=self.inferencer.vae,
        #     text_encoder=self.inferencer.text_encoder,
        #     tokenizer=self.inferencer.tokenizer,
        #     scheduler=DDPMScheduler.from_pretrained(
        #         self.inferencer.args.pretrained_model_name_or_path, subfolder="scheduler"
        #     ),
        #     safety_checker=None,
        #     feature_extractor=None,
        # )
        self.inferencer.pipeline.to(self.inferencer.args.device)

        # Manually load the LoRA
        self.load_custom_lora_processors(self.inferencer.unet, self.inferencer.args.model_path)

        # Validation scheduler (which we might override in concept guidance)
        self.inferencer.validation_scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        self.inferencer.validation_scheduler.set_timesteps(50)
        # self.inferencer.validation_scheduler = DDPMScheduler.from_pretrained(
        #     self.inferencer.args.pretrained_model_name_or_path, subfolder="scheduler"
        # )
        # self.inferencer.validation_scheduler.set_timesteps(1000)

    def load_custom_lora_processors(self, unet, model_path):
        """
        A helper that manually loads LoRA from model_path/pytorch_lora_weights.bin
        and sets them in unet via the custom cross-attn processor.
        """
        lora_path = os.path.join(model_path, "pytorch_lora_weights.bin")
        if not os.path.exists(lora_path):
            raise ValueError(f"Cannot find LoRA weights at {lora_path}")

        lora_state_dict = torch.load(lora_path, map_location="cpu")
        new_attn_procs = {}

        for name, processor in unet.attn_processors.items():
            # E.g. "down_blocks.0.attentions.0.transformer_blocks.0.attn1.processor"
            if name.endswith(".processor"):
                fixed_name = name.replace(".processor", ".processor.lora_processor")
            else:
                fixed_name = f"{name}.lora_processor"

            processor_sd = {
                k[len(fixed_name)+1:] : v
                for k, v in lora_state_dict.items()
                if k.startswith(fixed_name)
            }

            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim

            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks."):].split(".")[0])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks."):].split(".")[0])
                hidden_size = unet.config.block_out_channels[block_id]
            else:
                continue

            base_lora = LoRACrossAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                rank=4,
            )

            base_lora.load_state_dict(processor_sd, strict=True)

            custom_processor = P2PCrossAttnProcessorWithLoRA(
                controller=None,
                place_in_unet=("down" if "down_blocks" in name else
                                "up" if "up_blocks" in name else "mid"),
                original_processor=base_lora,
                log_attn_maps=False,
            )
            new_attn_procs[name] = custom_processor

        unet.set_attn_processor(new_attn_procs)
        unet.to(self.inferencer.args.device)

    def _load_concept_predictor(self):
        """
        Loads concept predictor from "pytorch_concept_predictor.bin" in self.inferencer.args.model_path,
        and moves it to self.inferencer.args.device. We assume the out_dim=8 or so, but you can adapt as needed.
        """
        ckpt_path = os.path.join(self.inferencer.args.model_path, "pytorch_concept_predictor.bin")
        if not os.path.exists(ckpt_path):
            raise ValueError(f"No concept predictor found at {ckpt_path}")

        # Adjust out_dim to match however many concept tokens you actually have
        # If you used <asset0>.. <asset7>, then out_dim=8
        out_dim = 8
        if self.inferencer.args.predictor_type == "classifier":
            self.inferencer.concept_predictor = ConceptClassifier(
                latent_channels=4,
                latent_size=64,
                out_dim=out_dim,
                hidden_dim=256
            )
        elif self.inferencer.args.predictor_type == "classifier_seg":
            self.inferencer.concept_predictor = ConceptClassifierSegmenter(
                latent_channels=4,
                latent_size=64,
                out_dim=out_dim,
                hidden_dim=256
            )
        elif self.inferencer.args.predictor_type == "classifier_time":
            self.inferencer.concept_predictor = ConceptClassifierWithTime(
                latent_channels=4,
                latent_size=64,
                out_dim=out_dim,
                hidden_dim=256
            )
        elif self.inferencer.args.predictor_type == "classifier_seg_time":
            self.inferencer.concept_predictor = ConceptClassifierSegmenterWithTime(
                latent_channels=4,
                latent_size=64,
                out_dim=out_dim,
                hidden_dim=256
            )
        elif self.inferencer.args.predictor_type == "classifier_seg_time_film":
            self.inferencer.concept_predictor = ConceptClassifierSegmenterWithTimeFiLM(
                latent_channels=4,
                latent_size=64,
                out_dim=out_dim,
                hidden_dim=256
            )
        else:
            raise ValueError(f"Unsupported predictor_type {self.inferencer.args.predictor_type}")

        self.inferencer.concept_predictor.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        self.inferencer.concept_predictor.eval()
        self.inferencer.concept_predictor.to(self.inferencer.args.device, dtype=torch.float16)
        print(f"[INFO] Loaded concept predictor from {ckpt_path} with out_dim={out_dim}")
        
        if self.inferencer.args.predictor_type in ["classifier_seg_time", "classifier_seg_time_film"]:
            original_time_emb_forward = self.inferencer.concept_predictor.time_emb.forward
            def patched_time_emb_forward(timesteps):
                out = original_time_emb_forward(timesteps)
                return out.to(torch.float16)
            self.inferencer.concept_predictor.time_emb.forward = patched_time_emb_forward