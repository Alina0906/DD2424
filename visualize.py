import argparse
import glob
import os

import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

import torch
from transformers import ViTForImageClassification, ViTImageProcessor, ViTConfig


checkpoint_dirs = glob.glob('./outputs/checkpoint-*/')
checkpoint_dir = checkpoint_dirs[0].rstrip('/')

def load_image(args):
    return Image.open("images/" + args.image)

def load_model():
    # save_model_path = './outputs/checkpoint-3494'
    model_name = 'google/vit-base-patch16-224-in21k'
    config_file = os.path.join(checkpoint_dir, 'config.json')
    if os.path.exists(config_file):
      config = ViTConfig.from_pretrained(config_file)
      model = ViTForImageClassification.from_pretrained(checkpoint_dir, config=config)
      processor = ViTImageProcessor.from_pretrained(model_name, do_rescale=False)
      # model = torch.load(save_model_path)
    else:
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
        attention_heads_fused = attention.mean(dim=1)  # Average across heads
        attention_heads_fused += torch.eye(attention_heads_fused.size(-1)).to(attention_heads_fused.device)
        attention_heads_fused /= attention_heads_fused.sum(dim=-1, keepdim=True)  # Normalize
        rollout = torch.matmul(rollout, attention_heads_fused)

    return rollout

def main():
    parser = argparse.ArgumentParser("Visualization for the Attention Map of VIT.")
    parser.add_argument("--image", type=str, help="Image name to visualize (e.g., cat.jpg)")
    args = parser.parse_args()

    model, processor, config = load_model()

    # Load and process image
    image = load_image(args)
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs, output_attentions=True)

    # Compute attention rollout
    attentions = outputs.attentions  # List of [batch_size, num_heads, seq_len, seq_len]
    rollout = attention_rollout(attentions)

    # Extract CLS token attention to patches
    image_size = config.image_size
    patch_size = config.patch_size
    num_patches = (image_size // patch_size) ** 2
    cls_attention = rollout[0, 1:, 0]  # Shape: [num_patches]

    # Reshape and normalize attention
    cls_attention = 1 - cls_attention  
    cls_attention = cls_attention.reshape(int(np.sqrt(num_patches)), int(np.sqrt(num_patches)))
    cls_attention = (cls_attention - cls_attention.min()) / (cls_attention.max() - cls_attention.min())

    # Convert to NumPy array and scale to [0, 255]
    cls_attention_np = cls_attention.detach().cpu().numpy()
    cls_attention_scaled = (cls_attention_np * 255).astype(np.uint8)

    # Resize and blur the attention map
    attention_img = Image.fromarray(cls_attention_scaled).resize(
        (image_size, image_size), resample=Image.BICUBIC
    )
    attention_img = attention_img.filter(ImageFilter.GaussianBlur(radius=2))

    # Overlay attention map on original image
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay.paste(attention_img.convert("L"), (0, 0), mask=attention_img)
    blended = Image.alpha_composite(image.convert("RGBA"), overlay)

    output_dir = "outputs/"
    output_path = "attention_overlay.png"
    blended.convert("RGB").save(output_dir + output_path)
    print(f"Attention map saved to {output_path}")

    original_width, original_height = image.size
    attention_resized = attention_img.resize((original_width, original_height), resample=Image.BICUBIC)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(attention_resized, cmap='viridis')  
    plt.title("Attention Map")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_dir + "side_by_side_pretrain.png")
    plt.show()

    # Heatmap
    attention_np = np.array(attention_resized.convert("L"))
    attention_normalized = (attention_np - attention_np.min()) / (attention_np.max() - attention_np.min())
    heatmap = cm.hot(attention_normalized)[:, :, :3]  
    heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8))
    overlay = Image.blend(image.convert("RGBA"), heatmap_img.convert("RGBA"), alpha=0.4)
    overlay.save(output_dir + "heatmap_overlay_pretrain.png")

if __name__ == "__main__":
    main()