from pathlib import Path
import random
import shutil

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE_DIR = PROJECT_ROOT / "datasets" / "final"
OUTPUT_DIR = PROJECT_ROOT / "datasets" / "split"

CLASSES = [
    "PLASTIC",
    "E_WASTE",
    "ORGANIC",
    "RECYCLABLE",
]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42

# ============================================================
# CHECK RATIOS
# ============================================================

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6

random.seed(SEED)

# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

for split in ["train", "val", "test"]:
    for class_name in CLASSES:
        (OUTPUT_DIR / split / class_name).mkdir(
            parents=True,
            exist_ok=True
        )

# ============================================================
# SPLIT DATASET
# ============================================================

print("=" * 70)
print("SPLITTING FINAL DATASET")
print("=" * 70)

total_train = 0
total_val = 0
total_test = 0

for class_name in CLASSES:

    source_class_dir = SOURCE_DIR / class_name

    if not source_class_dir.exists():
        print(f"\nERROR: Missing class folder: {source_class_dir}")
        continue

    images = [
        p for p in source_class_dir.iterdir()
        if p.is_file()
    ]

    random.shuffle(images)

    total = len(images)

    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)
    test_count = total - train_count - val_count

    train_images = images[:train_count]
    val_images = images[train_count:train_count + val_count]
    test_images = images[train_count + val_count:]

    print(f"\n{class_name}")
    print("-" * 70)
    print(f"Total : {total}")
    print(f"Train : {len(train_images)}")
    print(f"Val   : {len(val_images)}")
    print(f"Test  : {len(test_images)}")

    # --------------------------------------------------------
    # COPY TRAIN
    # --------------------------------------------------------

    for image in train_images:
        shutil.copy2(
            image,
            OUTPUT_DIR / "train" / class_name / image.name
        )

    # --------------------------------------------------------
    # COPY VALIDATION
    # --------------------------------------------------------

    for image in val_images:
        shutil.copy2(
            image,
            OUTPUT_DIR / "val" / class_name / image.name
        )

    # --------------------------------------------------------
    # COPY TEST
    # --------------------------------------------------------

    for image in test_images:
        shutil.copy2(
            image,
            OUTPUT_DIR / "test" / class_name / image.name
        )

    total_train += len(train_images)
    total_val += len(val_images)
    total_test += len(test_images)

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SPLIT COMPLETE")
print("=" * 70)

print(f"TRAIN      : {total_train}")
print(f"VALIDATION : {total_val}")
print(f"TEST       : {total_test}")
print("-" * 70)
print(f"TOTAL      : {total_train + total_val + total_test}")

print("\nOutput:")
print(OUTPUT_DIR)

print("\n" + "=" * 70)
print("IMPORTANT")
print("=" * 70)

print("The final dataset was NOT modified.")
print("This script only copied images into train/val/test.")