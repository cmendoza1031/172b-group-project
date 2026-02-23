# Full Fine-Tuning: ViT-B/16 on LC25000

Condition 4 of the CS172B group project ablation study. A pretrained ViT-B/16
is fine-tuned with all parameters unfrozen on the LC25000 histopathology
dataset. This serves as the upper-bound condition.

The other three conditions (linear probe, attention-only, everything-except-
attention) each freeze a different subset of the model. This condition freezes
nothing.

## Setup

```
pip install -r requirements.txt
```

Download the LC25000 dataset from Kaggle:

```
pip install kaggle
kaggle datasets download -d andrewmvd/lung-and-colon-cancer-histopathological-images -p data --unzip
```

The dataset should land under `data/` with this structure (the loader handles
both the nested Kaggle layout and a flat layout):

```
data/LC25000/
    colon_image_sets/
        colon_aca/
        colon_n/
    lung_image_sets/
        lung_aca/
        lung_n/
        lung_scc/
```

If the archive extracts to a different directory name, rename it so it matches
`data_root` in `config.py` (default: `data/LC25000`).

## Running

Train (runs test evaluation automatically at the end):

```
python train.py
```

Override defaults from the command line:

```
python train.py --epochs 30 --lr 1e-5 --batch_size 16
python train.py --data_root /path/to/LC25000 --output_dir outputs/run2
```

Re-evaluate a saved checkpoint on the test set:

```
python evaluate.py
```

Generate attention rollout visualizations:

```
python visualize.py
```

## Model

- Base: `google/vit-base-patch16-224` (ViT-B/16, pretrained on ImageNet-1k)
- All 86M parameters unfrozen
- Classifier head replaced for 5-class output

## Dataset

LC25000 contains 25,000 250x250 histopathology images, 5,000 per class:

| Class | Description |
|---|---|
| `colon_aca` | Colon adenocarcinoma |
| `colon_n` | Colon benign tissue |
| `lung_aca` | Lung adenocarcinoma |
| `lung_n` | Lung benign tissue |
| `lung_scc` | Lung squamous cell carcinoma |

Split: 80% train / 10% val / 10% test, stratified by class, seeded for
reproducibility.

## Hyperparameters

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Epochs | 25 |
| Batch size | 32 |
| LR schedule | Cosine with linear warmup (10% of steps) |
| Label smoothing | 0.1 |
| Gradient clip | 1.0 |

## Outputs

All outputs go to `outputs/full_finetune/` by default:

- `best_model.pt` - best checkpoint by validation accuracy
- `history.json` - per-epoch train/val loss and accuracy
- `metrics.json` - test accuracy, macro F1, per-class F1
- `training_curves.png` - loss and accuracy plots
- `confusion_matrix.png` - test set confusion matrix
- `attention_maps/` - attention rollout figures per class

## Attention Visualization

I use attention rollout (Abnar & Zuidema 2020) to visualize which image
regions the model attends to. Attention is averaged over heads at each of
the 12 transformer layers, augmented with the residual connection, and
accumulated via matrix multiplication across layers. This gives a more
faithful picture of information flow than raw last-layer attention.
