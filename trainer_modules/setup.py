import os
import torch
import math
import torch
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

import datasets
import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    UNet2DConditionModel,
    DDIMScheduler,
)
from transformers import AutoTokenizer, PretrainedConfig
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from diffusers.loaders import AttnProcsLayers
from diffusers.models.cross_attention import LoRACrossAttnProcessor

from modules.concept_predictor import ConceptClassifier, ConceptClassifierSegmenter
from modules.ptp_utils import AttentionStore
from utils.load_model import import_model_class_from_model_name_or_path

class Setup:
    def __init__(self, trainer, logger):
        self.trainer = trainer
        self.logger = logger
    
    def set_logging_dir(self):
        ### Get the training start time
        now = datetime.now()
        date_time_string = now.strftime("%Y-%m-%d-%H-%M")
        self.trainer.args.output_dir = self.trainer.args.output_dir + '_' + date_time_string

        logging_dir = Path(self.trainer.args.output_dir, self.trainer.args.logging_dir)

        self.trainer.accelerator = Accelerator(
            gradient_accumulation_steps=self.trainer.args.gradient_accumulation_steps,
            mixed_precision=self.trainer.args.mixed_precision,
            log_with=self.trainer.args.report_to,
            logging_dir=logging_dir,
        )

        if (
            self.trainer.args.train_text_encoder
            and self.trainer.args.gradient_accumulation_steps > 1
            and self.trainer.accelerator.num_processes > 1
        ):
            raise ValueError(
                "Gradient accumulation is not supported when training the text encoder in distributed training. "
                "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
            )

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        self.logger.info(self.trainer.accelerator.state, main_process_only=False)
        if self.trainer.accelerator.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        # If passed along, set the training seed now.
        if self.trainer.args.seed is not None:
            set_seed(self.trainer.args.seed)

        # Handle the repository creation
        if self.trainer.accelerator.is_main_process:
            os.makedirs(self.trainer.args.output_dir, exist_ok=True)

        return date_time_string

    ### Training setup
    def load_pretrained_models(self):
        # import correct text encoder class
        text_encoder_cls = import_model_class_from_model_name_or_path(
            self.trainer.args.pretrained_model_name_or_path, self.trainer.args.revision
        )

        ### Load scheduler and models
        self.trainer.noise_scheduler = DDPMScheduler.from_pretrained(
            self.trainer.args.pretrained_model_name_or_path, subfolder="scheduler"
        )
        # self.logger.info("Loaded noise scheduler config:", self.trainer.noise_scheduler.config)
        # self.trainer.noise_scheduler.config.prediction_type = "epsilon"
        # self.logger.info("Overriding scheduler's prediction_type to epsilon.")

        # self.logger.info(f'Start loading text encoder from {self.trainer.args.pretrained_model_name_or_path} with revision {self.trainer.args.revision}')
        self.trainer.text_encoder = text_encoder_cls.from_pretrained(
            self.trainer.args.pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=self.trainer.args.revision,
        )
        # self.logger.info("Loaded text encoder config:", self.trainer.text_encoder.config)
        # self.logger.info(f'Finish loading text encoder from {self.trainer.args.pretrained_model_name_or_path} with revision {self.trainer.args.revision}')
        
        self.trainer.vae = AutoencoderKL.from_pretrained(
            self.trainer.args.pretrained_model_name_or_path,
            subfolder="vae",
            revision=self.trainer.args.revision,
        )
        # self.logger.info("Loaded VAE config:", self.trainer.vae.config)

        self.trainer.unet = UNet2DConditionModel.from_pretrained(
            self.trainer.args.pretrained_model_name_or_path,
            subfolder="unet",
            revision=self.trainer.args.revision,
        )
        # self.logger.info("Loaded UNet config:", self.trainer.unet.config)

        ### Load the tokenizer
        # self.logger.info(f'Start loading tokenizer from {self.trainer.args.tokenizer_name} with revision {self.trainer.args.revision}')
        if self.trainer.args.tokenizer_name:
            self.trainer.tokenizer = AutoTokenizer.from_pretrained(
                self.trainer.args.tokenizer_name, revision=self.trainer.args.revision, use_fast=False
            )
            # self.logger.info("Loaded tokenizer config (custom name):", self.trainer.tokenizer.init_kwargs)
        elif self.trainer.args.pretrained_model_name_or_path:
            self.trainer.tokenizer = AutoTokenizer.from_pretrained(
                self.trainer.args.pretrained_model_name_or_path,
                subfolder="tokenizer",
                revision=self.trainer.args.revision,
                use_fast=False,
            )
            # self.logger.info("Loaded tokenizer config (from model path):", self.trainer.tokenizer.init_kwargs)
        # self.logger.info(f'Finish loading tokenizer from {self.trainer.args.tokenizer_name} with revision {self.trainer.args.revision}')
        # exit()

    def add_initialize_new_tokens(self):
        '''
            Add new part tokens to tokenizer and initialize their embeddings.
        '''
        ### Add assets tokens to tokenizer
        total_assets_indices = []
        for assets_indices in self.trainer.args.assets_indices_lists:
            for assets_idx in assets_indices:
                total_assets_indices.append(assets_idx)
        self.trainer.total_num_of_assets = len(total_assets_indices)
        self.logger.info(f"Total number of part assets: {self.trainer.total_num_of_assets}")

        self.trainer.all_placeholder_tokens = [
            self.trainer.args.placeholder_token.replace(">", f"{idx}>")
            for idx in total_assets_indices
        ]
        self.trainer.placeholder_tokens = []
        for idx, assets_indices in enumerate(self.trainer.args.assets_indices_lists):
            self.trainer.placeholder_tokens.append([])
            for assets_idx in assets_indices:
                self.trainer.placeholder_tokens[idx].append(self.trainer.args.placeholder_token.replace(">", f"{assets_idx}>"))
        self.logger.info(f"All placeholder tokens: {self.trainer.all_placeholder_tokens}")
        self.logger.info(f"Placeholder tokens: {self.trainer.placeholder_tokens}")

        num_added_tokens = self.trainer.tokenizer.add_tokens(self.trainer.all_placeholder_tokens)
        self.logger.info(f'After adding part tokens, total number of tokens: {len(self.trainer.tokenizer)}')
        # assert num_added_tokens == self.trainer.args.num_of_assets
        # assert num_added_tokens == total_num_of_assets
        self.trainer.placeholder_token_ids = self.trainer.tokenizer.convert_tokens_to_ids(
            self.trainer.all_placeholder_tokens
        )
        self.trainer.text_encoder.resize_token_embeddings(len(self.trainer.tokenizer))

        ### Initialize token embeddings
        all_initializer_tokens = []
        for initializer_tokens in self.trainer.args.initializer_tokens_list:
            for initializer_token in initializer_tokens:
                all_initializer_tokens.append(initializer_token)
        if len(all_initializer_tokens) > 0:
            ### Use initializer tokens
            token_embeds = self.trainer.text_encoder.get_input_embeddings().weight.data
            for tkn_idx, initializer_token in enumerate(all_initializer_tokens):
                curr_token_ids = self.trainer.tokenizer.encode(
                    initializer_token, add_special_tokens=False
                )
                # assert (len(curr_token_ids)) == 1
                token_embeds[self.trainer.placeholder_token_ids[tkn_idx]] = token_embeds[
                    curr_token_ids[0]
                ]
        else:
            ### Initialize new tokens randomly
            token_embeds = self.trainer.text_encoder.get_input_embeddings().weight.data
            token_embeds[-self.trainer.total_num_of_assets :] = token_embeds[
                -3 * self.trainer.total_num_of_assets : -2 * self.trainer.total_num_of_assets
            ]

    def add_initialize_bg_tokens(self):
        '''
            Add new bg tokens to tokenizer and initialize their embeddings.
        '''
        bg_indices = []
        for idx in self.trainer.args.bg_indices:
            bg_indices.append(idx)

        self.trainer.total_num_of_bgs = len(bg_indices)
        self.trainer.bg_placeholder_tokens = [
            self.trainer.args.bg_placeholder_token.replace(">", f"{i}>") for i in bg_indices
        ]
        self.logger.info(f'placeholder tokens: {self.trainer.bg_placeholder_tokens}')

        added_bg_tokens = self.trainer.tokenizer.add_tokens(self.trainer.bg_placeholder_tokens)
        self.logger.info(f'added_bg_tokens: {added_bg_tokens}')
        self.logger.info(f'After adding bg tokens, total number of tokens: {len(self.trainer.tokenizer)}')
        bg_token_ids = self.trainer.tokenizer.convert_tokens_to_ids(self.trainer.bg_placeholder_tokens)
        self.trainer.text_encoder.resize_token_embeddings(len(self.trainer.tokenizer))

        init_bg_embeds = self.trainer.text_encoder.get_input_embeddings().weight.data
        self.logger.info(f'bg_initializer_tokens: {self.trainer.args.bg_initializer_tokens}')
        if self.trainer.args.bg_initializer_tokens:
            for idx, init_token in enumerate(self.trainer.args.bg_initializer_tokens):
                self.logger.info(f'idex: {idx}, init_token: {init_token}')
                init_id = self.trainer.tokenizer.encode(init_token, add_special_tokens=False)[0]
                init_bg_embeds[bg_token_ids[idx]] = init_bg_embeds[init_id]
        else:
            init_bg_embeds[-self.trainer.total_num_of_bgs:] = init_bg_embeds[
                -(2 * self.trainer.total_num_of_bgs):-self.trainer.total_num_of_bgs
            ]

    def set_optimize_info(self):
        '''
            Set optimizer and scheduler for training stage 1.
        '''
        ### Set validation scheduler for logging
        self.trainer.validation_scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        self.trainer.validation_scheduler.set_timesteps(50)

        ### Start by only optimizing the embeddings
        self.trainer.vae.requires_grad_(False)
        self.trainer.unet.requires_grad_(False)
        # Freeze all parameters except for the token embeddings in text encoder
        self.trainer.text_encoder.text_model.encoder.requires_grad_(False)
        self.trainer.text_encoder.text_model.final_layer_norm.requires_grad_(False)
        self.trainer.text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)

        if self.trainer.args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                self.trainer.unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError(
                    "xformers is not available. Make sure it is installed correctly"
                )

        if self.trainer.args.gradient_checkpointing:
            self.trainer.unet.enable_gradient_checkpointing()
            if self.trainer.args.train_text_encoder:
                self.trainer.text_encoder.gradient_checkpointing_enable()

        if self.trainer.args.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        if self.trainer.args.scale_lr:
            self.trainer.args.learning_rate = (
                self.trainer.args.learning_rate
                * self.trainer.args.gradient_accumulation_steps
                * self.trainer.args.train_batch_size
                * self.trainer.accelerator.num_processes
            )

        if self.trainer.args.use_8bit_adam:
            try:
                import bitsandbytes as bnb

            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        return optimizer_class

    def set_scheduler_info(self, train_dataloader, optimizer):
        '''
            Currently disgarded. The logic is in the main trainer function
        '''
        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / self.trainer.args.gradient_accumulation_steps
        )
        if self.trainer.args.max_train_steps is None:
            self.trainer.args.max_train_steps = (
                self.trainer.args.num_train_epochs * num_update_steps_per_epoch
            )
            overrode_max_train_steps = True

        lr_scheduler = get_scheduler(
            self.trainer.args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.trainer.args.lr_warmup_steps
            * self.trainer.args.gradient_accumulation_steps,
            num_training_steps=self.trainer.args.max_train_steps
            * self.trainer.args.gradient_accumulation_steps,
            num_cycles=self.trainer.args.lr_num_cycles,
            power=self.trainer.args.lr_power,
        )

        (
            self.trainer.unet,
            self.trainer.text_encoder,
            optimizer,
            train_dataloader,
            lr_scheduler,
        ) = self.trainer.accelerator.prepare(
            self.trainer.unet, self.trainer.text_encoder, optimizer, train_dataloader, lr_scheduler
        )

        # For mixed precision training cast the text_encoder and vae weights to half-precision
        # as these models are only used for inference, keeping weights in full precision is not required.
        self.trainer.weight_dtype = torch.float32
        if self.trainer.accelerator.mixed_precision == "fp16":
            self.trainer.weight_dtype = torch.float16
        elif self.trainer.accelerator.mixed_precision == "bf16":
            self.trainer.weight_dtype = torch.bfloat16

        # Move vae and text_encoder to device and cast to weight_dtype
        self.trainer.vae.to(self.trainer.accelerator.device, dtype=self.trainer.weight_dtype)

        low_precision_error_string = (
            "Please make sure to always have all model weights in full float32 precision when starting training - even if"
            " doing mixed precision training. copy of the weights should still be float32."
        )

        if self.trainer.accelerator.unwrap_model(self.trainer.unet).dtype != torch.float32:
            raise ValueError(
                f"Unet loaded as datatype {self.trainer.accelerator.unwrap_model(self.trainer.unet).dtype}. {low_precision_error_string}"
            )

        if (
            self.trainer.args.train_text_encoder
            and self.trainer.accelerator.unwrap_model(self.trainer.text_encoder).dtype != torch.float32
        ):
            raise ValueError(
                f"Text encoder loaded as datatype {self.trainer.accelerator.unwrap_model(self.trainer.text_encoder).dtype}."
                f" {low_precision_error_string}"
            )

        # Need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / self.trainer.args.gradient_accumulation_steps
        )
        if overrode_max_train_steps:
            self.trainer.args.max_train_steps = (
                self.trainer.args.num_train_epochs * num_update_steps_per_epoch
            )
        # Afterwards recalculate our number of training epochs
        self.trainer.args.num_train_epochs = math.ceil(
            self.trainer.args.max_train_steps / num_update_steps_per_epoch
        )

        # Need to initialize the trackers to use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if self.trainer.accelerator.is_main_process:
            init_kwargs = {}
            # if self.trainer.args.report_to == "wandb":
            #     init_kwargs["wandb"] = {
            #         "name": self.trainer.args.wandb_run_name + '_' + date_time_string
            #     }

            self.trainer.accelerator.init_trackers('break-a-scene', config=vars(self.trainer.args), init_kwargs=init_kwargs)

        return num_update_steps_per_epoch

    def load_trained_weights(self, first_epoch, num_update_steps_per_epoch):
        if self.trainer.args.resume_from_checkpoint != "latest":
            path = os.path.basename(self.trainer.args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(self.trainer.args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            self.trainer.accelerator.self.logger.info(
                f"Checkpoint '{self.trainer.args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            self.trainer.args.resume_from_checkpoint = None
        else:
            self.trainer.accelerator.self.logger.info(f"Resuming from checkpoint {path}")
            self.trainer.accelerator.load_state(os.path.join(self.trainer.args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * self.trainer.args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch * self.trainer.args.gradient_accumulation_steps
            )

        return first_epoch, resume_step

    def set_lora_layers(self):
        ''' 
            Initialize LoRA layers here
        '''
        ### Note: These are already set in set_optimize_info function
        # self.trainer.vae.requires_grad_(False)
        # self.trainer.text_encoder.requires_grad_(False)
        # self.trainer.unet.requires_grad_(False)

        lora_attn_procs = {}
        for name in self.trainer.unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.trainer.unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.trainer.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.trainer.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.trainer.unet.config.block_out_channels[block_id]

            lora_attn_procs[name] = LoRACrossAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=self.trainer.args.lora_rank
            )

        self.trainer.unet.set_attn_processor(lora_attn_procs)
        self.trainer.lora_layers = AttnProcsLayers(self.trainer.unet.attn_processors)
        self.trainer.accelerator.register_for_checkpointing(self.trainer.lora_layers)

        # Unfreeze LoRA layers
        self.trainer.lora_layers.requires_grad_(True)
        if self.trainer.args.train_text_encoder:
            self.trainer.text_encoder.requires_grad_(True)

        # Re-register attention control after initializing LoRA layers
        # self.trainer.controller = AttentionStore()
        # self.trainer.register_attention_control(self.trainer.controller)
        self.trainer.controller = AttentionStore()
        self.trainer.attn_controller.register_attention_control_with_lora(self.trainer.controller)
        # self.logger.info("LoRA layers and attention control registered")

        # self.logger.info(f"Attention processor keys after setting LoRA layers: {self.trainer.unet.attn_processors.keys()}")

    def init_concept_predictor(self):
        """
        Instantiates the Q network for concept prediction.
        """
        # Example hyperparams - adjust accordingly or load from self.trainer.args
        latent_channels = 4       # Standard SD latent channels
        latent_size = 64          # Typically the UNet latents are 64x64
        out_dim = self.trainer.total_num_of_assets  # number of concept tokens e.g. 8
        hidden_dim = 256

        if self.trainer.args.predictor_type == "classifier":
            self.trainer.concept_predictor = ConceptClassifier(
                latent_channels=latent_channels,
                latent_size=latent_size,
                out_dim=out_dim,
                hidden_dim=hidden_dim
            ).to(self.trainer.accelerator.device)
        elif self.trainer.args.predictor_type == "classifier_seg":
            self.trainer.concept_predictor = ConceptClassifierSegmenter(
                latent_channels=latent_channels,
                latent_size=latent_size,
                out_dim=out_dim,
                hidden_dim=hidden_dim
            ).to(self.trainer.accelerator.device)
        elif self.trainer.args.predictor_type == "classifier_time":
            self.trainer.concept_predictor = ConceptClassifierWithTime(
                latent_channels=latent_channels,
                latent_size=latent_size,
                out_dim=out_dim,
                hidden_dim=hidden_dim
            ).to(self.trainer.accelerator.device)
        elif self.trainer.args.predictor_type == "classifier_seg_time":
            self.trainer.concept_predictor = ConceptClassifierSegmenterWithTime(
                latent_channels=latent_channels,
                latent_size=latent_size,
                out_dim=out_dim,
                hidden_dim=hidden_dim
            ).to(self.trainer.accelerator.device)
        elif self.trainer.args.predictor_type == "classifier_seg_time_film":
            self.trainer.concept_predictor = ConceptClassifierSegmenterWithTimeFiLM(
                latent_channels=latent_channels,
                latent_size=latent_size,
                out_dim=out_dim,
                hidden_dim=hidden_dim
            ).to(self.trainer.accelerator.device)
        elif self.trainer.args.predictor_type == "regressor":
            # If using regressor, you typically have a text embedding dimension (e.g., 768)
            embedding_dim = self.trainer.text_encoder.config.hidden_size  # e.g., 768
            self.trainer.concept_predictor = ConceptRegressor(
                latent_channels=latent_channels,
                latent_size=latent_size,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim
            ).to(self.trainer.accelerator.device)
        else:
            raise ValueError(f"Unsupported predictor_type: {self.trainer.args.predictor_type}")
