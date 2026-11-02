from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "datasets" / "split_grouped"
TEST_DIR = DATASET_DIR / "test"

MODEL_PATH = PROJECT_ROOT / "models" / "efficientnet_b0_best.pth"

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print("=" * 70)
print("AI WASTE SEGREGATION SYSTEM")
print("MODEL TEST EVALUATION")
print("=" * 70)

print(f"Project root : {PROJECT_ROOT}")
print(f"Test dataset : {TEST_DIR}")
print(f"Model        : {MODEL_PATH}")
print(f"Device       : {device}")

if torch.cuda.is_available():
    print(f"GPU          : {torch.cuda.get_device_name(0)}")
    print(
        f"GPU memory   : "
        f"{torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB"
    )

print("=" * 70)


# ============================================================
# CHECK PATHS
# ============================================================

if not TEST_DIR.exists():
    raise FileNotFoundError(f"Test dataset not found: {TEST_DIR}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")


# ============================================================
# TRANSFORM
# ============================================================

weights = EfficientNet_B0_Weights.DEFAULT

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=weights.transforms().mean,
        std=weights.transforms().std
    ),
])


# ============================================================
# LOAD TEST DATASET
# ============================================================

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transform
)

class_names = test_dataset.classes
num_classes = len(class_names)

print("\nTEST DATASET INFORMATION")
print("-" * 70)

print(f"Test images : {len(test_dataset)}")

print("\nClasses:")

for index, class_name in enumerate(class_names):
    print(f"  {index} -> {class_name}")


if num_classes != 4:
    raise ValueError(
        f"Expected 4 classes, but found {num_classes}: {class_names}"
    )


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading EfficientNet-B0...")
print("-" * 70)

model = efficientnet_b0(weights=None)

in_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    in_features,
    num_classes
)


# Load trained checkpoint
checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(checkpoint["model_state_dict"])

model = model.to(device)
model.eval()


print("Model loaded successfully.")

if "epoch" in checkpoint:
    print(f"Best model epoch       : {checkpoint['epoch']}")

if "val_accuracy" in checkpoint:
    print(
        f"Validation accuracy   : "
        f"{checkpoint['val_accuracy']:.2f}%"
    )

print("=" * 70)


# ============================================================
# TEST EVALUATION
# ============================================================

print("\nSTARTING TEST EVALUATION")
print("-" * 70)

all_predictions = []
all_labels = []

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)

        predictions = torch.argmax(outputs, dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )


# ============================================================
# CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision_macro, recall_macro, f1_macro, _ = (
    precision_recall_fscore_support(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )
)

precision_weighted, recall_weighted, f1_weighted, _ = (
    precision_recall_fscore_support(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )
)


# ============================================================
# PRINT OVERALL RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

print(f"Total test images     : {total}")
print(f"Correct predictions   : {correct}")
print(f"Incorrect predictions : {total - correct}")

print(f"\nTest Accuracy         : {accuracy * 100:.2f}%")

print("\nMACRO AVERAGE")
print("-" * 70)
print(f"Precision             : {precision_macro * 100:.2f}%")
print(f"Recall                : {recall_macro * 100:.2f}%")
print(f"F1 Score              : {f1_macro * 100:.2f}%")

print("\nWEIGHTED AVERAGE")
print("-" * 70)
print(f"Precision             : {precision_weighted * 100:.2f}%")
print(f"Recall                : {recall_weighted * 100:.2f}%")
print(f"F1 Score              : {f1_weighted * 100:.2f}%")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n")
print("=" * 70)
print("PER-CLASS CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(
    all_labels,
    all_predictions,
    target_names=class_names,
    digits=4,
    zero_division=0
)

print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(cm)


# ============================================================
# SAVE CONFUSION MATRIX IMAGE
# ============================================================

plt.figure(figsize=(8, 7))

plt.imshow(cm, interpolation="nearest")

plt.title("EfficientNet-B0 - Test Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")

plt.xticks(
    np.arange(num_classes),
    class_names,
    rotation=45,
    ha="right"
)

plt.yticks(
    np.arange(num_classes),
    class_names
)

# Add numbers inside cells
for i in range(num_classes):
    for j in range(num_classes):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )

plt.tight_layout()

confusion_matrix_path = (
    RESULTS_DIR / "confusion_matrix.png"
)

plt.savefig(
    confusion_matrix_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SAVE TEXT REPORT
# ============================================================

evaluation_report_path = (
    RESULTS_DIR / "evaluation_report.txt"
)

with open(
    evaluation_report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write("=" * 70 + "\n")
    f.write("AI WASTE SEGREGATION SYSTEM\n")
    f.write("EFFICIENTNET-B0 TEST EVALUATION\n")
    f.write("=" * 70 + "\n\n")

    f.write(f"Test images: {total}\n")
    f.write(f"Correct predictions: {correct}\n")
    f.write(f"Incorrect predictions: {total - correct}\n\n")

    f.write(f"Test Accuracy: {accuracy * 100:.4f}%\n")

    f.write("\nMacro Average\n")
    f.write(f"Precision: {precision_macro * 100:.4f}%\n")
    f.write(f"Recall: {recall_macro * 100:.4f}%\n")
    f.write(f"F1 Score: {f1_macro * 100:.4f}%\n")

    f.write("\nWeighted Average\n")
    f.write(f"Precision: {precision_weighted * 100:.4f}%\n")
    f.write(f"Recall: {recall_weighted * 100:.4f}%\n")
    f.write(f"F1 Score: {f1_weighted * 100:.4f}%\n\n")

    f.write("=" * 70 + "\n")
    f.write("PER-CLASS CLASSIFICATION REPORT\n")
    f.write("=" * 70 + "\n")

    f.write(report)

    f.write("\n\n")
    f.write("=" * 70 + "\n")
    f.write("CONFUSION MATRIX\n")
    f.write("=" * 70 + "\n")

    f.write(str(cm))


# ============================================================
# SAVE JSON RESULTS
# ============================================================

json_report = {
    "model": "EfficientNet-B0",
    "test_images": total,
    "correct_predictions": correct,
    "incorrect_predictions": total - correct,
    "accuracy": float(accuracy),
    "accuracy_percent": float(accuracy * 100),

    "macro_precision": float(precision_macro),
    "macro_recall": float(recall_macro),
    "macro_f1": float(f1_macro),

    "weighted_precision": float(precision_weighted),
    "weighted_recall": float(recall_weighted),
    "weighted_f1": float(f1_weighted),

    "classes": class_names,
    "confusion_matrix": cm.tolist(),
}


json_report_path = (
    RESULTS_DIR / "evaluation_results.json"
)

with open(
    json_report_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        json_report,
        f,
        indent=4
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 70)
print("TEST EVALUATION COMPLETE")
print("=" * 70)

print(f"Confusion matrix : {confusion_matrix_path}")
print(f"Text report      : {evaluation_report_path}")
print(f"JSON results     : {json_report_path}")

print("=" * 70)