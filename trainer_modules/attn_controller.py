import torch
from typing import List, Optional

from diffusers.models.cross_attention import LoRACrossAttnProcessor

from modules.cross_attn_processor import P2PCrossAttnProcessor, P2PCrossAttnProcessorWithLoRA

class AttnController:
    def __init__(self, trainer):
        self.trainer = trainer

    ### Attention control
    def reset_attention_processors(self, unet):
        # Re-initialize the attention processors to standard LoRACrossAttnProcessor
        lora_attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            else:
                continue
            lora_attn_procs[name] = LoRACrossAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=4
            )
        unet.set_attn_processor(lora_attn_procs)

    def register_attention_control(self, controller):
        attn_procs = {}
        cross_att_count = 0
        for name in self.trainer.unet.attn_processors.keys():
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else self.trainer.unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = self.trainer.unet.config.block_out_channels[-1]
                place_in_unet = "mid"
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.trainer.unet.config.block_out_channels))[
                    block_id
                ]
                place_in_unet = "up"
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.trainer.unet.config.block_out_channels[block_id]
                place_in_unet = "down"
            else:
                continue
            cross_att_count += 1
            attn_procs[name] = P2PCrossAttnProcessor(
                controller=controller, place_in_unet=place_in_unet
            )

        self.trainer.unet.set_attn_processor(attn_procs)
        controller.num_att_layers = cross_att_count

    def register_attention_control_with_lora(self, controller):
        attn_procs = {}
        cross_attn_count = 0
        for name in self.trainer.unet.attn_processors.keys():
            # Retrieve the original processor which includes LoRA layers
            original_processor = self.trainer.unet.attn_processors[name]

            # Determine the position in UNet
            if name.startswith("down_blocks"):
                place_in_unet = "down"
            elif name.startswith("up_blocks"):
                place_in_unet = "up"
            elif name.startswith("mid_block"):
                place_in_unet = "mid"
            else:
                continue  # Skip if not matching expected patterns
            cross_attn_count += 1
            # Wrap the original processor with your custom processor
            attn_procs[name] = P2PCrossAttnProcessorWithLoRA(
                controller=controller,
                place_in_unet=place_in_unet,
                original_processor=original_processor
            )

        self.trainer.unet.set_attn_processor(attn_procs)
        controller.num_att_layers = cross_attn_count
        # print(f"Registered {cross_attn_count} attention processors with LoRA")

    def get_average_attention(self):
        average_attention = {
            key: [
                item / self.trainer.controller.cur_step
                for item in self.trainer.controller.attention_store[key]
            ]
            for key in self.trainer.controller.attention_store
        }
        return average_attention

    def aggregate_attention(self, res: int, from_where: List[str], is_cross: bool, select: int):
        '''
        The original implementation in Break-a-Scene
        '''
        out = []
        attention_maps = self.get_average_attention()
        num_pixels = res**2
        for location in from_where:
            for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
                if item.shape[1] == num_pixels:
                    cross_maps = item.reshape(
                        self.trainer.args.train_batch_size, -1, res, res, item.shape[-1]
                    )[select]
                    out.append(cross_maps)
        out = torch.cat(out, dim=0)
        out = out.sum(0) / out.shape[0]
        return out

    def aggregate_attention_batch(self, res: int, from_where: List[str], is_cross: bool, select: int, batch_size: int):
        '''
        Modified implemenatation to remove CFG and to properly handle batch size of 2
        '''
        out = []
        attention_maps = self.get_average_attention()
        # print(f"Attention maps keys: {attention_maps.keys()}")
        num_pixels = res ** 2
        for location in from_where:
            for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
                if item.shape[2] == num_pixels:
                    # item shape is [batch_size, num_heads, seq_len_q, seq_len_k]
                    num_heads = item.shape[1]
                    # Average over heads
                    item = item.mean(dim=1)  # shape [batch_size, seq_len_q, seq_len_k]
                    # Reshape seq_len_q to res x res
                    item = item.view(batch_size, res, res, -1)
                    cross_maps = item[select]  # select the batch index
                    out.append(cross_maps)
        if len(out) == 0:
            raise ValueError("No attention maps found matching the specified resolution.")
        out = torch.stack(out, dim=0)
        out = out.mean(dim=0)  # average over layers
        return out  # shape [res, res, seq_len_k]
