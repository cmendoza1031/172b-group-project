"""Attention rollout visualization for the full fine-tuned ViT.

I implement the attention rollout method from:
    Abnar & Zuidema (2020), "Quantifying Attention Flow in Transformers"
    https://arxiv.org/abs/2005.00928

Rollout accumulates attention across all 12 transformer layers by taking the
product of per-layer attention matrices (with residual connections added),
giving a more faithful view of information flow than raw last-layer attention.
"""

import os
from pathlib import Path
from typing import List, Optional

import matplotlib.cm as mpl_cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import ViTForImageClassification

from config import Config
from dataset import CLASSES, _IMAGE_EXTENSIONS, _SUBDIR_CANDIDATES
from model import build_model
from utils import get_device


class AttentionRollout:
    """I compute attention rollout maps for a ViT model.

    I average attention weights over heads at each layer, add the residual
    (identity) connection, renormalize rows, then accumulate the product
    across all layers. The resulting CLS-to-patch attention map captures
    the effective flow of information from input patches to the final
    class token representation.

    Args:
        model: A ViTForImageClassification instance.
        discard_ratio: Fraction of lowest attention weights to zero out
            before accumulation. Higher values sharpen the visualization.
    """

    def __init__(self, model: ViTForImageClassification, discard_ratio: float = 0.9) -> None:
        self.model = model
        self.discard_ratio = discard_ratio

    @torch.no_grad()
    def __call__(self, pixel_values: torch.Tensor) -> np.ndarray:
        """I return a 2D attention rollout map in [0, 1] for a single image.

        Args:
            pixel_values: Tensor of shape (1, 3, H, W), normalized.

        Returns:
            Numpy array of shape (grid_size, grid_size), e.g. (14, 14)
            for ViT-B/16 with 224x224 input.
        """
        outputs = self.model(pixel_values, output_attentions=True)
        # attentions: tuple of (1, num_heads, seq_len, seq_len) per layer
        attentions = outputs.attentions

        seq_len = attentions[0].shape[-1]
        # I start with the identity matrix representing residual connections
        result = torch.eye(seq_len, device=pixel_values.device)

        for attention in attentions:
            # I average over heads: (1, num_heads, S, S) -> (S, S)
            attn = attention.squeeze(0).mean(dim=0)

            # I discard the bottom discard_ratio fraction to reduce noise
            flat = attn.flatten()
            threshold_idx = int(self.discard_ratio * flat.numel())
            if threshold_idx < flat.numel():
                threshold = flat.kthvalue(threshold_idx).values
                attn = torch.where(attn >= threshold, attn, torch.zeros_like(attn))

            # I add residual and renormalize so rows sum to 1
            attn = attn + torch.eye(seq_len, device=attn.device)
            attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)

            result = attn @ result

        # Row 0 is the CLS token; columns 1: are the patch tokens
        mask = result[0, 1:].cpu().numpy()
        grid_size = int(round(mask.shape[0] ** 0.5))
        mask = mask.reshape(grid_size, grid_size)

        # I normalize to [0, 1] for display
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        return mask


def _overlay_attention(
    image_np: np.ndarray, attention_map: np.ndarray, alpha: float = 0.55
) -> np.ndarray:
    """I blend an attention heatmap over an image using jet colormap."""
    h, w = image_np.shape[:2]
    attn_img = Image.fromarray((attention_map * 255).astype(np.uint8))
    attn_img = attn_img.resize((w, h), Image.BILINEAR)
    attn_np = np.array(attn_img, dtype=np.float32) / 255.0

    heatmap = mpl_cm.jet(attn_np)[:, :, :3]  # RGB, ignoring alpha channel
    base = image_np.astype(np.float32) / 255.0
    overlay = (1 - alpha) * base + alpha * heatmap
    return (np.clip(overlay, 0, 1) * 255).astype(np.uint8)


def visualize_samples(
    model: ViTForImageClassification,
    data_root: str,
    class_names: List[str],
    output_dir: str,
    n_per_class: int = 2,
    device: Optional[torch.device] = None,
) -> None:
    """I generate and save attention rollout figures for sample images.

    For each class I pick n_per_class images and produce a three-panel
    figure: original, raw rollout map, and blended overlay.

    Args:
        model: Trained model; should already be on device.
        data_root: Root directory of the LC25000 dataset.
        class_names: Ordered list of class name strings.
        output_dir: Directory where PNG files are written.
        n_per_class: Number of sample images per class.
        device: Compute device. Inferred from model parameters if None.
    """
    if device is None:
        device = next(model.parameters()).device

    rollout = AttentionRollout(model)
    model.eval()

    attn_dir = os.path.join(output_dir, "attention_maps")
    os.makedirs(attn_dir, exist_ok=True)

    # I use two separate transforms: one for display (no normalization),
    # one for the model (with ImageNet normalization).
    display_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    model_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    root = Path(data_root)

    for cls_idx, cls_name in enumerate(class_names):
        candidates = _SUBDIR_CANDIDATES[cls_name]
        class_dir = None
        for candidate in candidates:
            p = root / candidate
            if p.is_dir():
                class_dir = p
                break
        if class_dir is None:
            print(f"Skipping {cls_name}: directory not found under {root}")
            continue

        img_paths = sorted(
            p for p in class_dir.iterdir()
            if p.suffix.lower() in _IMAGE_EXTENSIONS
        )[:n_per_class]

        fig, axes = plt.subplots(n_per_class, 3, figsize=(12, 4 * n_per_class))
        if n_per_class == 1:
            axes = [axes]

        for row, img_path in enumerate(img_paths):
            image = Image.open(img_path).convert("RGB")

            img_display = (
                display_tf(image).permute(1, 2, 0).numpy() * 255
            ).astype(np.uint8)
            img_tensor = model_tf(image).unsqueeze(0).to(device)

            attn_map = rollout(img_tensor)
            overlay = _overlay_attention(img_display, attn_map)

            axes[row][0].imshow(img_display)
            axes[row][0].set_title("Original")
            axes[row][0].axis("off")

            axes[row][1].imshow(attn_map, cmap="jet")
            axes[row][1].set_title("Attention Rollout")
            axes[row][1].axis("off")

            axes[row][2].imshow(overlay)
            axes[row][2].set_title("Overlay")
            axes[row][2].axis("off")

        fig.suptitle(f"Class: {cls_name} - Full Fine-Tuning", fontsize=14)
        plt.tight_layout()
        save_path = os.path.join(attn_dir, f"{cls_name}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")


if __name__ == "__main__":
    cfg = Config()
    device = get_device()
    print(f"Device: {device}")

    model = build_model(cfg.num_classes, cfg.model_name, device)
    ckpt_path = os.path.join(cfg.output_dir, "best_model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}. Run train.py first.")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    visualize_samples(
        model,
        data_root=cfg.data_root,
        class_names=cfg.class_names,
        output_dir=cfg.output_dir,
        n_per_class=3,
        device=device,
    )
