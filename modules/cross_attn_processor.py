import torch
from diffusers.models.cross_attention import CrossAttention, LoRACrossAttnProcessor

class P2PCrossAttnProcessor(torch.nn.Module):
    def __init__(self, controller, place_in_unet):
        # super().__init__()
        super(P2PCrossAttnProcessor, self).__init__()
        self.controller = controller
        self.place_in_unet = place_in_unet

    def __call__(
        self,
        attn: CrossAttention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
    ):
        # print(f'Check check check is calling P2PCrossAttnProcessor!!!')
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length)

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = (
            encoder_hidden_states
            if encoder_hidden_states is not None
            else hidden_states
        )
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # one line change
        self.controller(attention_probs, is_cross, self.place_in_unet, batch_size)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

class P2PCrossAttnProcessorWithLoRA(torch.nn.Module):
    def __init__(self, controller, place_in_unet, original_processor, log_attn_maps = True):
        super(P2PCrossAttnProcessorWithLoRA, self).__init__()
        self.controller = controller
        self.place_in_unet = place_in_unet
        self.lora_processor = original_processor
        self.log_attn_maps = log_attn_maps

    def __call__(
        self,
        attn: CrossAttention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        scale=1.0
    ):
        '''
        Modified from third_party/diffusers/models/cross_attention.py - line 275 class LoRACrossAttnProcessor(nn.Module)
        '''
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length)

        query = attn.to_q(hidden_states) + scale * self.lora_processor.to_q_lora(hidden_states)
        query = attn.head_to_batch_dim(query)

        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

        key = attn.to_k(encoder_hidden_states) + scale * self.lora_processor.to_k_lora(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states) + scale * self.lora_processor.to_v_lora(encoder_hidden_states)

        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # # Store attention_probs for external access
        # self.attention_probs = attention_probs.detach().clone()
        # print(f"Stored attention_probs in P2PCrossAttnProcessorWithLoRA")

        # Log attention maps
        if self.log_attn_maps:
            self.controller(attention_probs, is_cross, self.place_in_unet, batch_size)
        # print(f"Logged attention_probs in P2PCrossAttnProcessorWithLoRA for {self.place_in_unet}")

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # Linear projection
        hidden_states = attn.to_out[0](hidden_states) + scale * self.lora_processor.to_out_lora(hidden_states)
        # Dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states