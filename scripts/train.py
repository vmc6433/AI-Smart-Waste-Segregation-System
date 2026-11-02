from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import EfficientNet_B0_Weights


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "datasets" / "split_grouped"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"

MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Training configuration
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 20

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0

RANDOM_SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("AI WASTE SEGREGATION SYSTEM")
print("MODEL TRAINING")
print("=" * 70)

print(f"Project root : {PROJECT_ROOT}")
print(f"Dataset      : {DATASET_DIR}")
print(f"Device       : {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU          : {torch.cuda.get_device_name(0)}")
    print(
        f"GPU memory   : "
        f"{torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB"
    )

print("=" * 70)


# ============================================================
# DATA TRANSFORMS
# ============================================================

weights = EfficientNet_B0_Weights.DEFAULT

imagenet_mean = weights.transforms().mean
imagenet_std = weights.transforms().std


train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomRotation(degrees=10),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=imagenet_mean,
        std=imagenet_std
    )
])


val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=imagenet_mean,
        std=imagenet_std
    )
])


# ============================================================
# DATASETS
# ============================================================

train_dataset = datasets.ImageFolder(
    root=TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    root=VAL_DIR,
    transform=val_transform
)


print()
print("DATASET INFORMATION")
print("-" * 70)

print(f"Training images   : {len(train_dataset)}")
print(f"Validation images : {len(val_dataset)}")

print()
print("Classes:")
for index, class_name in enumerate(train_dataset.classes):
    print(f"  {index} -> {class_name}")

print("=" * 70)


# ============================================================
# CHECK CLASS CONSISTENCY
# ============================================================

if train_dataset.classes != val_dataset.classes:
    raise RuntimeError(
        "Training and validation classes do not match."
    )


NUM_CLASSES = len(train_dataset.classes)

if NUM_CLASSES != 4:
    raise RuntimeError(
        f"Expected 4 classes, but found {NUM_CLASSES}."
    )


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# MODEL
# ============================================================

print()
print("Loading EfficientNet-B0...")

model = models.efficientnet_b0(
    weights=weights
)


# Replace the final classifier
in_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    in_features,
    NUM_CLASSES
)

model = model.to(DEVICE)


# ============================================================
# CLASS WEIGHTS
# ============================================================

class_counts = np.bincount(
    train_dataset.targets,
    minlength=NUM_CLASSES
)

print()
print("TRAINING CLASS COUNTS")
print("-" * 70)

for class_name, count in zip(
    train_dataset.classes,
    class_counts
):
    print(f"{class_name:<15} : {count}")


# Balanced class weights
class_weights = len(train_dataset) / (
    NUM_CLASSES * class_counts
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32,
    device=DEVICE
)

print()
print("CLASS WEIGHTS")

for class_name, weight in zip(
    train_dataset.classes,
    class_weights
):
    print(f"{class_name:<15} : {weight.item():.4f}")


# ============================================================
# LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)


# ============================================================
# MIXED PRECISION
# ============================================================

use_amp = torch.cuda.is_available()

if use_amp:
    scaler = torch.amp.GradScaler("cuda")
else:
    scaler = None


# ============================================================
# TRAINING FUNCTIONS
# ============================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        if use_amp:

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

        else:

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()


        running_loss += (
            loss.item() * images.size(0)
        )

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


    epoch_loss = running_loss / total

    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate():

    model.eval()

    running_loss = 0.0

    correct = 0
    total = 0

    for images, labels in val_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        if use_amp:

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

        else:

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )


        running_loss += (
            loss.item() * images.size(0)
        )

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


    epoch_loss = running_loss / total

    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# TRAINING LOOP
# ============================================================

best_val_accuracy = 0.0

best_model_path = MODEL_DIR / "efficientnet_b0_best.pth"

history = []


print()
print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)


for epoch in range(NUM_EPOCHS):

    print()
    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS}"
    )

    print("-" * 70)


    train_loss, train_accuracy = train_one_epoch()

    val_loss, val_accuracy = validate()


    scheduler.step(val_accuracy)


    current_lr = optimizer.param_groups[0]["lr"]


    print(
        f"Train Loss      : {train_loss:.4f}"
    )

    print(
        f"Train Accuracy  : {train_accuracy * 100:.2f}%"
    )

    print(
        f"Val Loss        : {val_loss:.4f}"
    )

    print(
        f"Val Accuracy    : {val_accuracy * 100:.2f}%"
    )

    print(
        f"Learning Rate   : {current_lr:.6f}"
    )


    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "learning_rate": current_lr
    })


    # Save best model
    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_names": train_dataset.classes,
                "image_size": IMAGE_SIZE,
                "model_name": "efficientnet_b0",
                "val_accuracy": val_accuracy,
                "epoch": epoch + 1
            },
            best_model_path
        )

        print()
        print(
            f"✅ New best model saved!"
        )

        print(
            f"   Validation Accuracy: "
            f"{val_accuracy * 100:.2f}%"
        )


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

import json

history_path = RESULTS_DIR / "training_history.json"

with open(
    history_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        history,
        file,
        indent=4
    )


# ============================================================
# TRAINING COMPLETE
# ============================================================

print()
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    f"Best validation accuracy : "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    f"Best model               : "
    f"{best_model_path}"
)

print(
    f"Training history         : "
    f"{history_path}"
)

print("=" * 70)