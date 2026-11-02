from pathlib import Path
import csv
import shutil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_DIR = PROJECT_ROOT / "datasets" / "split_grouped" / "test"
MODEL_PATH = PROJECT_ROOT / "models" / "efficientnet_b0_best.pth"

RESULTS_DIR = PROJECT_ROOT / "results"
ERROR_DIR = RESULTS_DIR / "error_analysis"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ERROR_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


print("=" * 70)
print("AI WASTE SEGREGATION SYSTEM")
print("ERROR ANALYSIS")
print("=" * 70)

print(f"Project root : {PROJECT_ROOT}")
print(f"Test dataset : {TEST_DIR}")
print(f"Model        : {MODEL_PATH}")
print(f"Device       : {device}")

if torch.cuda.is_available():
    print(f"GPU          : {torch.cuda.get_device_name(0)}")

print("=" * 70)


# ============================================================
# LOAD TRANSFORM
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

print("\nCLASSES")
print("-" * 70)

for i, name in enumerate(class_names):
    print(f"{i} -> {name}")

print(f"\nTotal test images: {len(test_dataset)}")


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading EfficientNet-B0...")

model = efficientnet_b0(weights=None)

in_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    in_features,
    num_classes
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)
model.eval()

print("Model loaded successfully.")


# ============================================================
# PREPARE ERROR DIRECTORIES
# ============================================================

for true_class in class_names:
    for predicted_class in class_names:

        if true_class == predicted_class:
            continue

        folder_name = (
            f"{true_class}_as_{predicted_class}"
        )

        folder_path = ERROR_DIR / folder_name
        folder_path.mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# RUN PREDICTIONS
# ============================================================

errors = []

correct_count = 0
incorrect_count = 0

all_predictions = []
all_confidences = []


with torch.no_grad():

    for batch_start in range(
        0,
        len(test_dataset),
        BATCH_SIZE
    ):

        batch_end = min(
            batch_start + BATCH_SIZE,
            len(test_dataset)
        )

        batch_images = []
        batch_labels = []

        for index in range(
            batch_start,
            batch_end
        ):

            image, label = test_dataset[index]

            batch_images.append(image)
            batch_labels.append(label)

        images = torch.stack(batch_images).to(
            device,
            non_blocking=True
        )

        labels = torch.tensor(
            batch_labels,
            dtype=torch.long,
            device=device
        )

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidences, predictions = torch.max(
            probabilities,
            dim=1
        )

        for local_index in range(
            len(labels)
        ):

            dataset_index = batch_start + local_index

            true_index = labels[local_index].item()
            predicted_index = predictions[local_index].item()

            true_class = class_names[true_index]
            predicted_class = class_names[predicted_index]

            confidence = confidences[
                local_index
            ].item()

            original_path = Path(
                test_dataset.samples[dataset_index][0]
            )

            if true_index == predicted_index:

                correct_count += 1

            else:

                incorrect_count += 1

                folder_name = (
                    f"{true_class}_as_{predicted_class}"
                )

                destination_folder = (
                    ERROR_DIR / folder_name
                )

                destination_path = (
                    destination_folder /
                    original_path.name
                )

                # Avoid overwriting if duplicate filename exists
                if destination_path.exists():

                    destination_path = (
                        destination_folder /
                        f"{dataset_index}_{original_path.name}"
                    )

                shutil.copy2(
                    original_path,
                    destination_path
                )

                errors.append({
                    "index": dataset_index,
                    "image": original_path.name,
                    "original_path": str(original_path),
                    "true_class": true_class,
                    "predicted_class": predicted_class,
                    "confidence": confidence
                })


# ============================================================
# SAVE CSV
# ============================================================

csv_path = (
    RESULTS_DIR /
    "error_analysis.csv"
)

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "index",
            "image",
            "original_path",
            "true_class",
            "predicted_class",
            "confidence"
        ]
    )

    writer.writeheader()

    writer.writerows(errors)


# ============================================================
# ERROR SUMMARY
# ============================================================

pair_counts = {}

for error in errors:

    pair = (
        error["true_class"],
        error["predicted_class"]
    )

    pair_counts[pair] = (
        pair_counts.get(pair, 0) + 1
    )


print("\n")
print("=" * 70)
print("ERROR ANALYSIS RESULTS")
print("=" * 70)

print(f"Correct predictions   : {correct_count}")
print(f"Incorrect predictions : {incorrect_count}")

print("\nERRORS BY CLASS PAIR")
print("-" * 70)

sorted_pairs = sorted(
    pair_counts.items(),
    key=lambda x: x[1],
    reverse=True
)

for (true_class, predicted_class), count in sorted_pairs:

    print(
        f"{true_class:15s} -> "
        f"{predicted_class:15s} : "
        f"{count}"
    )


# ============================================================
# CONFIDENCE ANALYSIS
# ============================================================

if errors:

    errors_sorted = sorted(
        errors,
        key=lambda x: x["confidence"],
        reverse=True
    )

    print("\nMOST CONFIDENT WRONG PREDICTIONS")
    print("-" * 70)

    for error in errors_sorted[:10]:

        print(
            f"{error['true_class']:15s} -> "
            f"{error['predicted_class']:15s} | "
            f"{error['confidence'] * 100:.2f}% | "
            f"{error['image']}"
        )


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 70)
print("ERROR ANALYSIS COMPLETE")
print("=" * 70)

print(f"Error CSV : {csv_path}")
print(f"Error images : {ERROR_DIR}")

print("=" * 70)