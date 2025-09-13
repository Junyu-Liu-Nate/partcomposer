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

from utils.arg_parser_inference import parse_args
from inference_modules.model_loader import ModelLoader

class PartComposerInference:
    def __init__(self):
        self.args = parse_args()
        self.model_loader = ModelLoader(self)
        self.model_loader._load_pipeline_lora()

        self.output_dir = os.path.abspath(self.args.output_path)
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def _filename_from_prompt(prompt: str, idx: int) -> str:
        """
        Extract <assetX> numbers and build "<num>_<num>_…_sample{idx}.jpg".
        Returns "no_asset_sample{idx}.jpg" if nothing is found.
        """
        asset_ids = re.findall(r"<asset(\d+)>", prompt)
        if asset_ids:
            asset_part = "_".join(asset_ids)
        else:
            asset_part = "no_asset"
        return f"{asset_part}_sample{idx}.jpg"

    @torch.no_grad()
    def infer_and_save(self, prompts, num_samples: int = 38):
        """
        For every prompt in `prompts` generate `num_samples` images and save them
        to self.output_dir with automatic, prompt‑based names.
        """
        for prompt in prompts:
            for i in range(num_samples):
                torch.manual_seed(i)
                images = self.pipeline(prompt).images      # returns a list
                filename = self._filename_from_prompt(prompt, i)
                save_path = os.path.join(self.output_dir, filename)
                images[0].save(save_path)
                print(f"Saved: {save_path}")

if __name__ == "__main__":
    runner = PartComposerInference()
    runner.infer_and_save(prompts=runner.args.prompts)