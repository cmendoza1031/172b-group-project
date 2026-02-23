import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

# Canonical class ordering used throughout this project.
CLASSES: List[str] = ["colon_aca", "colon_n", "lung_aca", "lung_n", "lung_scc"]

# I support both a flat layout (class dirs at root) and the nested layout
# from the Kaggle download (lung_colon_image_set/{colon,lung}_image_sets/...).
_SUBDIR_CANDIDATES: Dict[str, List[str]] = {
    "colon_aca": ["colon_aca", "colon_image_sets/colon_aca"],
    "colon_n":   ["colon_n",   "colon_image_sets/colon_n"],
    "lung_aca":  ["lung_aca",  "lung_image_sets/lung_aca"],
    "lung_n":    ["lung_n",    "lung_image_sets/lung_n"],
    "lung_scc":  ["lung_scc",  "lung_image_sets/lung_scc"],
}

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


class LC25000Dataset(Dataset):
    """I load LC25000 histopathology images from disk.

    I handle both the flat directory layout (class dirs directly under root)
    and the Kaggle nested layout (colon_image_sets/ and lung_image_sets/
    subdirectories). An optional transform is applied at retrieval time.
    """

    def __init__(self, root: str, transform=None) -> None:
        self.root = Path(root)
        self.transform = transform
        self.classes = CLASSES
        self.class_to_idx = {cls: i for i, cls in enumerate(CLASSES)}
        self.samples: List[Tuple[str, int]] = []

        for cls, candidates in _SUBDIR_CANDIDATES.items():
            class_dir = None
            for candidate in candidates:
                p = self.root / candidate
                if p.is_dir():
                    class_dir = p
                    break
            if class_dir is None:
                raise FileNotFoundError(
                    f"Could not find class '{cls}' under {self.root}. "
                    f"Tried: {[str(self.root / c) for c in candidates]}"
                )
            label = self.class_to_idx[cls]
            for path in sorted(class_dir.iterdir()):
                if path.suffix.lower() in _IMAGE_EXTENSIONS:
                    self.samples.append((str(path), label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def get_transforms(image_size: int, split: str) -> transforms.Compose:
    """I return the appropriate transform pipeline for each dataset split.

    Training augmentations are deliberately conservative: histopathology
    staining matters, so I avoid aggressive color distortion. Horizontal
    and vertical flips are valid because tissue orientation is arbitrary.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    if split == "train":
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            normalize,
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])


def create_dataloaders(
    data_root: str,
    image_size: int,
    batch_size: int,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    num_workers: int,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """I build stratified train/val/test dataloaders for LC25000.

    I stratify by class so that each split preserves the original class
    distribution. The RNG is seeded for reproducibility.
    """
    rng = random.Random(seed)

    # I load a reference dataset once just to collect sample paths and labels.
    ref = LC25000Dataset(data_root)
    class_names = ref.classes

    # I group sample indices by class label for stratified splitting.
    class_indices: Dict[int, List[int]] = {}
    for idx, (_, label) in enumerate(ref.samples):
        class_indices.setdefault(label, []).append(idx)

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    for label in sorted(class_indices):
        idxs = class_indices[label].copy()
        rng.shuffle(idxs)
        n = len(idxs)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_idx.extend(idxs[:n_train])
        val_idx.extend(idxs[n_train : n_train + n_val])
        test_idx.extend(idxs[n_train + n_val :])

    # I create three dataset instances so each split gets its own transforms.
    train_ds = LC25000Dataset(data_root, transform=get_transforms(image_size, "train"))
    val_ds   = LC25000Dataset(data_root, transform=get_transforms(image_size, "val"))
    test_ds  = LC25000Dataset(data_root, transform=get_transforms(image_size, "test"))

    loader_kwargs = dict(num_workers=num_workers, pin_memory=pin_memory)
    train_loader = DataLoader(
        Subset(train_ds, train_idx),
        batch_size=batch_size, shuffle=True,  drop_last=True,  **loader_kwargs
    )
    val_loader = DataLoader(
        Subset(val_ds, val_idx),
        batch_size=batch_size, shuffle=False, drop_last=False, **loader_kwargs
    )
    test_loader = DataLoader(
        Subset(test_ds, test_idx),
        batch_size=batch_size, shuffle=False, drop_last=False, **loader_kwargs
    )

    return train_loader, val_loader, test_loader, class_names
