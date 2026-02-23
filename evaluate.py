"""Standalone evaluation script and reusable evaluation utilities.

I compute accuracy, macro F1, per-class F1, and a confusion matrix
on a given dataloader. Results are saved to JSON and PNG.
"""

import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from tqdm import tqdm

from config import Config
from dataset import create_dataloaders
from model import build_model
from utils import get_device, set_seed


@torch.no_grad()
def run_evaluation(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: List[str],
    output_dir: str,
) -> Dict:
    """I evaluate model predictions against ground truth labels.

    I collect all predictions in a single pass, then compute metrics
    with scikit-learn for correctness. A confusion matrix PNG is saved
    to output_dir alongside a JSON metrics file.
    """
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []

    for pixel_values, labels in tqdm(loader, desc="Eval", leave=False):
        pixel_values = pixel_values.to(device, non_blocking=True)
        outputs = model(pixel_values)
        preds = outputs.logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = float(accuracy_score(all_labels, all_preds))
    macro_f1 = float(f1_score(all_labels, all_preds, average="macro"))
    per_class_f1_vals = f1_score(all_labels, all_preds, average=None)
    report = classification_report(
        all_labels, all_preds, target_names=class_names, output_dict=True
    )

    metrics = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class_f1": {
            cls: float(f1) for cls, f1 in zip(class_names, per_class_f1_vals)
        },
        "classification_report": report,
    }

    cm = confusion_matrix(all_labels, all_preds)
    _plot_confusion_matrix(cm, class_names, output_dir)

    return metrics


def _plot_confusion_matrix(
    cm: np.ndarray, class_names: List[str], output_dir: str
) -> None:
    """I render and save the confusion matrix as a heatmap."""
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix - Full Fine-Tuning")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    cfg = Config()
    set_seed(cfg.seed)
    device = get_device()
    print(f"Device: {device}")

    pin_memory = device.type == "cuda"
    _, _, test_loader, class_names = create_dataloaders(
        data_root=cfg.data_root,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        seed=cfg.seed,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )

    model = build_model(cfg.num_classes, cfg.model_name, device)
    ckpt_path = os.path.join(cfg.output_dir, "best_model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}. Run train.py first.")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    os.makedirs(cfg.output_dir, exist_ok=True)
    metrics = run_evaluation(model, test_loader, device, class_names, cfg.output_dir)

    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Macro F1 : {metrics['macro_f1']:.4f}")
    print("\nPer-class F1:")
    for cls, f1 in metrics["per_class_f1"].items():
        print(f"  {cls}: {f1:.4f}")

    out_path = os.path.join(cfg.output_dir, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {out_path}")
