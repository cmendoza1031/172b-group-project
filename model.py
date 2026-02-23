from typing import Dict

import torch
import torch.nn as nn
from transformers import ViTForImageClassification


def build_model(
    num_classes: int,
    model_name: str,
    device: torch.device,
) -> ViTForImageClassification:
    """I load a pretrained ViT-B/16 and configure it for full fine-tuning.

    This is Condition 4 of the ablation study: all parameters are unfrozen.
    The classifier head is replaced to match the target number of classes.
    Using ignore_mismatched_sizes=True handles the head size mismatch between
    the ImageNet-pretrained head (1000 classes) and our 5-class output.
    """
    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )

    # I unfreeze every parameter — this is the defining property of full fine-tuning.
    for param in model.parameters():
        param.requires_grad = True

    return model.to(device)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """I return total, trainable, and frozen parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}
