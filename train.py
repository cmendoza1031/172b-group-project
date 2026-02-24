"""Full fine-tuning of ViT-B/16 on LC25000 (Condition 4).

This is the upper-bound condition in the group ablation study. Every
parameter in the model is unfrozen and updated during training.

Usage:
    python train.py
    python train.py --epochs 30 --lr 1e-5 --batch_size 16
    python train.py --data_root /path/to/LC25000 --output_dir outputs/run2
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

from config import Config
from dataset import create_dataloaders
from evaluate import run_evaluation
from model import build_model, count_parameters
from prompt_tuning import PromptTunedViT
from utils import get_cosine_schedule_with_warmup, get_device, set_seed


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion: nn.Module,
    device: torch.device,
    clip_norm: float,
    scaler,  # GradScaler for CUDA, None otherwise
    use_amp: bool,
) -> tuple:
    """I run one full pass over the training set and return (loss, accuracy).

    I use torch.autocast for both CUDA (with GradScaler) and MPS (without
    GradScaler, which MPS does not support). This gives roughly 2x throughput
    on Apple Silicon versus full float32.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for pixel_values, labels in tqdm(loader, desc="Train", leave=False):
        pixel_values = pixel_values.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(pixel_values)
            loss = criterion(outputs.logits, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()

        scheduler.step()

        total_loss += loss.item() * labels.size(0)
        preds = outputs.logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> tuple:
    """I evaluate the model on a validation set and return (loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for pixel_values, labels in tqdm(loader, desc="Val  ", leave=False):
        pixel_values = pixel_values.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(pixel_values)
            loss = criterion(outputs.logits, labels)
        total_loss += loss.item() * labels.size(0)
        preds = outputs.logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def plot_training_curves(history: dict, output_dir: str) -> None:
    """I plot and save loss and accuracy curves for both train and val splits."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"],   label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Cross-Entropy Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Train")
    axes[1].plot(epochs, history["val_acc"],   label="Val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    plt.suptitle("Full Fine-Tuning - ViT-B/16 on LC25000", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"), dpi=150)
    plt.close()


def main(cfg: Config) -> None:
    set_seed(cfg.seed)
    device = get_device()
    print(f"Device      : {device}")
    print(f"Output dir  : {cfg.output_dir}")
    print(f"Data root   : {cfg.data_root}")

    os.makedirs(cfg.output_dir, exist_ok=True)

    # Data
    # pin_memory and num_workers are CUDA-only optimizations. On MPS,
    # pinned memory isn't supported and spawning workers on macOS adds
    # overhead that outweighs prefetch benefit when compute is the bottleneck.
    pin_memory = device.type == "cuda"
    num_workers = cfg.num_workers if device.type == "cuda" else 0
    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        data_root=cfg.data_root,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        seed=cfg.seed,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    print(
        f"Splits      : {len(train_loader.dataset)} train / "
        f"{len(val_loader.dataset)} val / {len(test_loader.dataset)} test"
    )

    # Model
    model = build_model(cfg.num_classes, cfg.model_name, device)
    if cfg.num_prompts > 0:
        model = PromptTunedViT(model, num_prompts=cfg.num_prompts,
                               prompt_dropout=cfg.prompt_dropout)
        print(f"Prompt tuning enabled: {cfg.num_prompts} tokens "
              f"({model.count_prompt_parameters():,} extra params)")
    counts = count_parameters(model)
    print(
        f"Parameters  : {counts['total']:,} total | "
        f"{counts['trainable']:,} trainable | "
        f"{counts['frozen']:,} frozen"
    )

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    total_steps = len(train_loader) * cfg.num_epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # MPS does not have Tensor Cores, so float16 autocast adds conversion
    # overhead without a net compute benefit. Only enable AMP on CUDA.
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    best_val_acc = 0.0
    ckpt_path = os.path.join(cfg.output_dir, "best_model.pt")
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"\nStarting training for {cfg.num_epochs} epochs")
    print("-" * 72)

    for epoch in range(1, cfg.num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            criterion, device, cfg.gradient_clip_norm, scaler, use_amp,
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device, use_amp)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        marker = " *" if val_acc > best_val_acc else ""
        print(
            f"Epoch {epoch:3d}/{cfg.num_epochs}  "
            f"train loss {train_loss:.4f}  acc {train_acc:.4f}  |  "
            f"val loss {val_loss:.4f}  acc {val_acc:.4f}{marker}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)

    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    print(f"Checkpoint saved to: {ckpt_path}")

    # Save training curves and history
    plot_training_curves(history, cfg.output_dir)
    with open(os.path.join(cfg.output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Final test evaluation using the best checkpoint
    print("\nRunning test evaluation on best checkpoint...")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    metrics = run_evaluation(model, test_loader, device, class_names, cfg.output_dir)

    print(f"Test accuracy : {metrics['accuracy']:.4f}")
    print(f"Test macro F1 : {metrics['macro_f1']:.4f}")
    print("\nPer-class F1:")
    for cls, f1 in metrics["per_class_f1"].items():
        print(f"  {cls}: {f1:.4f}")

    with open(os.path.join(cfg.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nAll outputs written to: {cfg.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Full fine-tuning of ViT-B/16 on LC25000 (Condition 4)"
    )
    parser.add_argument("--data_root",  type=str,   default=None)
    parser.add_argument("--output_dir", type=str,   default=None)
    parser.add_argument("--epochs",     type=int,   default=None)
    parser.add_argument("--lr",         type=float, default=None)
    parser.add_argument("--batch_size", type=int,   default=None)
    parser.add_argument("--seed",        type=int,   default=None)
    parser.add_argument("--num_prompts", type=int,   default=None,
                        help="number of VPT tokens (0 = no prompt tuning)")
    args = parser.parse_args()

    cfg = Config()
    if args.data_root  is not None: cfg.data_root   = args.data_root
    if args.output_dir is not None: cfg.output_dir  = args.output_dir
    if args.epochs     is not None: cfg.num_epochs  = args.epochs
    if args.lr         is not None: cfg.learning_rate = args.lr
    if args.batch_size is not None: cfg.batch_size  = args.batch_size
    if args.seed        is not None: cfg.seed        = args.seed
    if args.num_prompts is not None: cfg.num_prompts = args.num_prompts

    main(cfg)
