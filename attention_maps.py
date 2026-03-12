import torch
import timm
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
from torchvision.transforms.functional import to_pil_image
import numpy as np

def get_attention_maps(model, image_tensor):

    attention_maps = []

    def hook_fn(module, input, output):
        qkv = output  # Get QKV output
        batch_size, tokens, dim = qkv.shape  # Extract dimensions

        num_heads = 12  # Default for ViT-B16
        head_dim = dim // (3 * num_heads)  # Compute head_dim dynamically

        qkv = qkv.view(batch_size, tokens, 3, num_heads, head_dim)  # Reshape properly
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # Extract Q, K, V

        # 🔹 Fix: Correct attention calculation
        attn_scores = (q.transpose(1, 2) @ k.transpose(1, 2).transpose(-2, -1)) / (head_dim ** 0.5)
        # Shape should now be (1, 12, 197, 197)

        attention_maps.append(attn_scores)

    # Register hooks on attention layers
    hooks = []
    for blk in model.blocks:
        hooks.append(blk.attn.qkv.register_forward_hook(hook_fn))

    # Forward pass to capture attention
    with torch.no_grad():
        _ = model.forward_features(image_tensor)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return attention_maps  # List of attention maps for each layer




# 
