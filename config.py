from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # Dataset
    data_root: str = "data/LC25000"
    num_classes: int = 5
    class_names: List[str] = field(default_factory=lambda: [
        "colon_aca", "colon_n", "lung_aca", "lung_n", "lung_scc"
    ])
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    # test_ratio is implicitly 1 - train_ratio - val_ratio = 0.1
    image_size: int = 224

    # Model
    model_name: str = "google/vit-base-patch16-224"

    # Training hyperparameters
    batch_size: int = 32
    num_epochs: int = 25
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1      # fraction of total steps used for linear warmup
    gradient_clip_norm: float = 1.0
    label_smoothing: float = 0.1

    # Output
    output_dir: str = "outputs/full_finetune"
    seed: int = 42
    num_workers: int = 4
