import torch
import numpy as np

from diffusers.models.cross_attention import LoRACrossAttnProcessor

from modules.cross_attn_processor import P2PCrossAttnProcessor, P2PCrossAttnProcessorWithLoRA

class Monitor:
    def __init__(self, trainer):
        self.trainer = trainer

    ### Weights monitoring
    def monitor_embeddings(self, logs):
        # Metrics for newly added embeddings
        embed_weights = self.trainer.text_encoder.get_input_embeddings().weight.data[self.trainer.placeholder_token_ids]
        logs['embedding_mean_norm'] = embed_weights.norm(dim=1).mean().item()
        logs['embedding_min_norm'] = embed_weights.norm(dim=1).min().item()
        logs['embedding_max_norm'] = embed_weights.norm(dim=1).max().item()
        logs['embedding_mean'] = embed_weights.mean().item()
        logs['embedding_variance'] = embed_weights.var().item()

        ### Log the extract gradients for newly added tokens
        embed_grads = self.trainer.text_encoder.get_input_embeddings().weight.grad
        if embed_grads is not None:
            # Slice gradients for the newly added embeddings
            new_token_grads = embed_grads[self.trainer.placeholder_token_ids]
            # Compute norms
            new_token_grad_norm = new_token_grads.norm(dim=1).mean().item()

            # Log the gradient norm
            logs['embedding_grad_norm'] = new_token_grad_norm
            logs['embedding_grad_mean'] = new_token_grads.mean().item()
            logs['embedding_grad_variance'] = new_token_grads.var().item()

            # Optionally print for debugging
            # print("Gradient of newly added tokens:", new_token_grads)
            # print("Gradient norm of newly added tokens:", new_token_grad_norm)
        else:
            logs['embedding_grad_norm'] = 0  # Default value if gradients are not available
            logs['embedding_grad_mean'] = 0
            logs['embedding_grad_variance'] = 0
            print("No gradient available for newly added tokens.")

        return logs

    def monitor_lora(self, logs):
        lora_weights = []
        lora_grads = []
        for name, param in self.trainer.lora_layers.named_parameters():
            # Log weights
            if param.requires_grad:
                lora_weights.append(param.flatten())

            # Log gradients
            if param.grad is not None:
                lora_grads.append(param.grad.flatten())
            else:
                print(f"Gradient for {name} is None.")

        # Combine all weights and gradients to calculate global metrics
        if len(lora_weights) > 0:
            lora_weights_combined = torch.cat(lora_weights)
            logs['lora_mean_weight'] = lora_weights_combined.mean().item()
            logs['lora_min_weight'] = lora_weights_combined.min().item()
            logs['lora_max_weight'] = lora_weights_combined.max().item()
            logs['lora_weight_norm'] = lora_weights_combined.norm().item()
            logs['lora_weight_variance'] = lora_weights_combined.var().item()

        if len(lora_grads) > 0:
            lora_grads_combined = torch.cat(lora_grads)
            logs['lora_grad_norm'] = lora_grads_combined.norm().item()
            logs['lora_grad_mean'] = lora_grads_combined.mean().item()
            logs['lora_grad_variance'] = lora_grads_combined.var().item()
        else:
            logs['lora_grad_norm'] = 0  # No gradients available
            logs['lora_grad_mean'] = 0
            logs['lora_grad_variance'] = 0

        return logs

    def log_lora_weights_and_grads_debug(self, logs):
        total_weight_norm = 0.0
        total_grad_norm = 0.0

        for name, module in self.trainer.unet.attn_processors.items():
            print(f"Processing {name}: {type(module)}")

            # Check if the module is the wrapped processor
            if isinstance(module, P2PCrossAttnProcessorWithLoRA) and hasattr(module, 'lora_processor'):
                orig_module = module.lora_processor

                if isinstance(orig_module, LoRACrossAttnProcessor):
                    lora_layers = {
                        'to_q_lora': orig_module.to_q_lora,
                        'to_k_lora': orig_module.to_k_lora,
                        'to_v_lora': orig_module.to_v_lora,
                        'to_out_lora': orig_module.to_out_lora
                    }

                    for lora_name, lora_module in lora_layers.items():
                        print(f"  Found LoRA module: {lora_name}")
                        for param_name, param in lora_module.named_parameters():
                            full_param_name = f"{name}.{lora_name}.{param_name}"
                            weight_norm = param.data.norm().item()
                            total_weight_norm += weight_norm
                            print(f"    {full_param_name} weight norm: {weight_norm}")

                            if param.grad is not None:
                                grad_norm = param.grad.data.norm().item()
                                total_grad_norm += grad_norm
                                print(f"    {full_param_name} grad norm: {grad_norm}")
                            else:
                                print(f"    {full_param_name} grad is None")
                else:
                    print(f"  Wrapped processor in {name} is not a LoRACrossAttnProcessor")
            else:
                print(f"  Module {name} is not a P2PCrossAttnProcessorWithLoRA")

        logs['lora_weight_norm'] = total_weight_norm
        logs['lora_grad_norm'] = total_grad_norm
        return logs

    def monitor_concept_predictor(self, logs):
        """
        Logs the total/general weights and gradients statistics of the concept predictor.

        Args:
            logs (dict): A dictionary to store the metrics.

        Returns:
            dict: Updated logs with the general weights and gradients statistics.
        """
        total_weights = 0
        total_weights_norm = 0
        total_weights_variance = 0
        total_params = 0
        total_grads = 0
        total_grads_norm = 0
        total_grads_variance = 0
        total_grads_count = 0

        for param in self.trainer.concept_predictor.parameters():
            if param.requires_grad:
                # Total weight statistics
                total_weights += param.data.sum().item()
                total_weights_norm += param.data.norm().item()
                total_weights_variance += param.data.var().item()
                total_params += param.numel()

                # Total gradient statistics
                if param.grad is not None:
                    total_grads += param.grad.sum().item()
                    total_grads_norm += param.grad.norm().item()
                    total_grads_variance += param.grad.var().item()
                    total_grads_count += param.grad.numel()

        # Log the aggregated statistics
        logs["concept_predictor_total_weight_mean"] = total_weights / total_params if total_params > 0 else 0
        logs["concept_predictor_total_weight_norm"] = total_weights_norm
        logs["concept_predictor_total_weight_variance"] = total_weights_variance / total_params if total_params > 0 else 0

        logs["concept_predictor_total_grad_mean"] = total_grads / total_grads_count if total_grads_count > 0 else 0
        logs["concept_predictor_total_grad_norm"] = total_grads_norm
        logs["concept_predictor_total_grad_variance"] = total_grads_variance / total_grads_count if total_grads_count > 0 else 0

        return logs