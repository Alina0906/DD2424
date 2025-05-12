import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import torch
from transformers import ViTForImageClassification, ViTImageProcessor, ViTConfig


parser = argparse.ArgumentParser("Visualization for the Attention Map of VIT.")
parser.add_argument("--image", type=str, help="Please input the iamge name you hope to visualize.")
args = parser.parse_args()

save_model_path = 'best_model.pth'


def load_image(args):
    
    return Image.open("images/" + args.image)


def load_model():
    if os.path.exists(save_model_path):
        model = torch.load(save_model_path)
    else:
        model_name = 'google/vit-base-patch16-224-in21k'
        model = ViTForImageClassification.from_pretrained(
            model_name, attn_implementation='eager'
        )
        processor = ViTImageProcessor.from_pretrained(model_name, do_rescale=False)
        config = ViTConfig.from_pretrained(model_name)
    return model, processor, config


def attention_rollout(attentions):
    # Initialize rollout with identity matrix
    rollout = torch.eye(attentions[0].size(-1)).to(attentions[0].device)

    # Multiply attention maps layer by layer
    for attention in attentions:
        attention_heads_fused = attention.mean(dim=1) # Average attention across heads
        attention_heads_fused += torch.eye(attention_heads_fused.size(-1)).to(attention_heads_fused.device) # A + I
        attention_heads_fused /= attention_heads_fused.sum(dim=-1, keepdim=True) # Normalizing A
        rollout = torch.matmul(rollout, attention_heads_fused) 

    return rollout


def main():
    model, processor, config = load_model()
    image = load_image(args.image)
    image.show()
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs, output_attentions=True)

    # Getting the attentions
    attentions = outputs.attentions 

    ig, axs = plt.subplots(4, 4, figsize=(20, 20))
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(attentions[-1][0, i, :, :].detach().cpu().numpy())
        ax.axis('off')
    
    rollout = attention_rollout(attentions)

    image_size = config.image_size
    patch_size = config.patch_size
    num_of_patches = (image_size // patch_size) ** 2
    cls_attention = rollout[0, 1:, 0]  # Get attention values from [CLS] token to all patches
    cls_attention = 1 - cls_attention.reshape(int(np.sqrt(num_of_patches)), int(np.sqrt(num_of_patches)))

    # Normalize the attention map for better visualization
    cls_attention = (cls_attention - cls_attention.min()) / (cls_attention.max() - cls_attention.min())

    # Resize and blur the attention map
    cls_attention_resized = Image.fromarray((cls_attention * 255).astype(np.uint8)).resize((image_size, image_size), resample=Image.BICUBIC)
    cls_attention_resized = cls_attention_resized.filter(ImageFilter.GaussianBlur(radius=2))

    cls_attention_colored = np.array(cls_attention_resized.convert("L"))
    cls_attention_colored = np.stack([cls_attention_colored]*3 + [cls_attention_colored], axis=-1)

    # Adjust the alpha channel to control brightness
    cls_attention_colored_img = Image.fromarray(cls_attention_colored, mode="RGBA")
    cls_attention_colored_img.putalpha(100)  # Adjust alpha for blending (lower value for darker overlay)

    orig_rgba = image.convert("RGBA")
    attention_overlay = cls_attention_colored_img.resize(orig_rgba.size, resample=Image.BICUBIC)
    blended = Image.alpha_composite(orig_rgba, attention_overlay)

    plt.figure(figsize=(8, 8))
    plt.imshow(blended.convert("RGB"))
    plt.axis('off')
    plt.title("Attention Map Overlay")
    plt.show()
