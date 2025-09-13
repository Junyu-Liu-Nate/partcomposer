from PIL import Image
import wandb
import numpy as np
import torch

#####----- 64*64 level visualization -----#####
def visualize_combined_masks_64x64(downsampled_synth_mask, downsampled_mask, logs):
    '''
    Generate 2 images, each image is of size 1*num_of_masks
    Used in computing mask diffusion loss
    '''
    synth_mask = downsampled_synth_mask[0, 0].detach().cpu().numpy()
    original_mask = downsampled_mask[0, 0].detach().cpu().numpy()

    # Normalize the masks to [0, 1]
    synth_mask = (synth_mask - synth_mask.min()) / (synth_mask.max() - synth_mask.min() + 1e-8)
    original_mask = (original_mask - original_mask.min()) / (original_mask.max() - original_mask.min() + 1e-8)

    # Scale to [0, 255] and convert to uint8
    synth_mask_img = (synth_mask * 255).astype(np.uint8)
    original_mask_img = (original_mask * 255).astype(np.uint8)

    # Convert NumPy arrays to PIL Images
    synth_mask_pil = Image.fromarray(synth_mask_img)
    original_mask_pil = Image.fromarray(original_mask_img)
    mask_diff_visulization = []
    mask_diff_visulization.append(wandb.Image(synth_mask_pil, caption=f"synthesized image"))
    mask_diff_visulization.append(wandb.Image(original_mask_pil, caption=f"original image"))
    logs[f"training mask diffusion vis"] = mask_diff_visulization

    return logs

def visualize_individual_masks_64x64(synth_masks_flat, instance_masks_flat, logs):
    '''
    Generate 2 images, each image is of size 1*num_of_masks
    Used in computing mask diffusion loss - weighted loss
    '''
    # Normalize Synthesized Masks to [0, 1]
    synth_masks_norm = (synth_masks_flat - synth_masks_flat.min()) / (synth_masks_flat.max() - synth_masks_flat.min() + 1e-8)
    synth_masks_norm = synth_masks_norm.cpu().numpy()  # Move to CPU and convert to NumPy

    # Convert Synthesized Masks to PIL RGB Images
    synth_pil_images = []
    for mask in synth_masks_norm:
        mask_np = (mask * 255).astype(np.uint8).squeeze()  # Scale to [0, 255] and remove channel dimension
        pil_gray = Image.fromarray(mask_np, mode='L')     # Convert to grayscale PIL Image
        pil_rgb = pil_gray.convert("RGB")                # Convert grayscale to RGB
        synth_pil_images.append(pil_rgb)                 # Append to list

    # Create Grid Image for Synthesized Masks (Single Row)
    num_synth = len(synth_pil_images)
    padding = 2  # Padding between images in pixels
    grid_width = num_synth * 64 + (num_synth - 1) * padding  # Total width
    grid_height = 64  # Height of each mask
    synth_grid = Image.new('RGB', (grid_width, grid_height), color=(255, 255, 255))  # White background

    for idx, img in enumerate(synth_pil_images):
        x_position = idx * (64 + padding)  # Calculate x position for each mask
        synth_grid.paste(img, (x_position, 0))  # Paste mask into grid

    # Normalize Instance Masks to [0, 1]
    instance_masks_norm = (instance_masks_flat - instance_masks_flat.min()) / (instance_masks_flat.max() - instance_masks_flat.min() + 1e-8)
    instance_masks_norm = instance_masks_norm.cpu().numpy()  # Move to CPU and convert to NumPy

    # Convert Instance Masks to PIL RGB Images
    instance_pil_images = []
    for mask in instance_masks_norm:
        mask_np = (mask * 255).astype(np.uint8).squeeze()  # Scale to [0, 255] and remove channel dimension
        pil_gray = Image.fromarray(mask_np, mode='L')     # Convert to grayscale PIL Image
        pil_rgb = pil_gray.convert("RGB")                # Convert grayscale to RGB
        instance_pil_images.append(pil_rgb)              # Append to list

    # Create Grid Image for Instance Masks (Single Row)
    num_instance = len(instance_pil_images)
    grid_width = num_instance * 64 + (num_instance - 1) * padding  # Total width
    grid_height = 64  # Height of each mask
    instance_grid = Image.new('RGB', (grid_width, grid_height), color=(255, 255, 255))  # White background

    for idx, img in enumerate(instance_pil_images):
        x_position = idx * (64 + padding)  # Calculate x position for each mask
        instance_grid.paste(img, (x_position, 0))  # Paste mask into grid

    # Log the Grid Images to Weights & Biases (WandB)
    mask_diff_visulization = []
    mask_diff_visulization.append(wandb.Image(synth_grid, caption=f"synthesized image"))
    mask_diff_visulization.append(wandb.Image(instance_grid, caption=f"original image"))
    logs[f"training mask diffusion vis"] = mask_diff_visulization

    return logs

#####----- 16*16 level visualization -----#####
def visualize_mask_atten(batch_mask_attn_visulization, logs):
    '''
    Generate 2 images, each image is of size 2*num_of_masks - the first row is mask, the second row is attention
    Used in computing cross-attention loss
    '''
    mask_attn_visulization = []
    for batch_idx in range(len(batch_mask_attn_visulization)):
        vertical_concats = []
        for GT_mask, asset_attn_mask in batch_mask_attn_visulization[batch_idx]:
            # Normalize masks to [0, 1]
            GT_mask = (GT_mask - GT_mask.min()) / (GT_mask.max() - GT_mask.min() + 1e-8)
            asset_attn_mask = (asset_attn_mask - asset_attn_mask.min()) / (asset_attn_mask.max() - asset_attn_mask.min() + 1e-8)

            # Stack vertically
            concatenated = torch.cat([GT_mask, asset_attn_mask], dim=0)  # Shape: [2*H, W]
            vertical_concats.append(concatenated)
        # Concatenate all vertically concatenated images horizontally
        big_image = torch.cat(vertical_concats, dim=1)  # Shape: [2*H, N*W]

        # Convert to numpy array
        big_image_np = big_image.detach().cpu().numpy()

        # Rescale to [0, 255]
        big_image_np = (big_image_np * 255).astype(np.uint8)

        # Convert to PIL image
        image = Image.fromarray(big_image_np)

        # Log image to wandb
        mask_attn_visulization.append(wandb.Image(image, caption=f"batch idx {batch_idx}"))

    logs[f"training attention vis"] = mask_attn_visulization
    return logs

def visualize_mask_atten_refined(batch_mask_attn_visulization, logs):
    '''
    Generate 2 images, each image is of size 2*num_of_masks - the first row is mask, the second row is attention
    White lines are added to separate each sub image for better visualization
    Used in computing cross-attention loss
    '''
    mask_attn_visulization = []
    for batch_idx in range(len(batch_mask_attn_visulization)):
        vertical_concats = []
        for GT_mask, asset_attn_mask in batch_mask_attn_visulization[batch_idx]:
            # Normalize masks to [0, 1]
            GT_mask = (GT_mask - GT_mask.min()) / (GT_mask.max() - GT_mask.min() + 1e-8)
            asset_attn_mask = (asset_attn_mask - asset_attn_mask.min()) / (asset_attn_mask.max() - asset_attn_mask.min() + 1e-8)

            # Stack vertically
            concatenated = torch.cat([GT_mask, asset_attn_mask], dim=0)  # Shape: [2*H, W]
            vertical_concats.append(concatenated)
        # Concatenate all vertically concatenated images horizontally
        big_image = torch.cat(vertical_concats, dim=1)  # Shape: [2*H, N*W]

        # Convert to numpy array
        big_image_np = big_image.detach().cpu().numpy()

        # Rescale to [0, 255]
        big_image_np = (big_image_np * 255).astype(np.uint8)

        # Add solid white lines to highlight sub-image boundaries
        line_width = 1   # Thickness of the lines

        total_height, total_width = big_image_np.shape
        H = total_height // 2  # Height of one sub-image
        N = len(vertical_concats)  # Number of sub-images horizontally
        W = total_width // N  # Width of one sub-image

        # Draw vertical solid lines
        for k in range(1, N):
            x = k * W
            y_indices = np.arange(total_height)
            for w in range(line_width):
                if x + w < total_width:
                    big_image_np[y_indices, x + w] = 255  # Set pixels to white

        # Draw horizontal solid line at y = H
        x_indices = np.arange(total_width)
        for w in range(line_width):
            if H + w < total_height:
                big_image_np[H + w, x_indices] = 255  # Set pixels to white

        # Convert to PIL image
        image = Image.fromarray(big_image_np)

        # Log image to wandb
        mask_attn_visulization.append(wandb.Image(image, caption=f"batch idx {batch_idx}"))
    
    logs[f"training attention vis"] = mask_attn_visulization
    return logs