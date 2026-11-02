from pathlib import Path
from PIL import Image
import shutil


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_DIR = PROJECT_ROOT / "datasets" / "master"
FINAL_DIR = PROJECT_ROOT / "datasets" / "final"

CLASSES = [
    "PLASTIC",
    "E_WASTE",
    "ORGANIC",
    "RECYCLABLE",
]


def check_image(path):
    """
    Verify that the image can actually be opened.
    Returns True if valid, False otherwise.
    """

    try:
        with Image.open(path) as img:
            img.verify()

        # Open again after verify to ensure it is readable.
        with Image.open(path) as img:
            img.load()

        return True

    except Exception:
        return False


print("=" * 70)
print("CREATING FINAL CLEAN DATASET")
print("=" * 70)

print()
print(f"Source : {SOURCE_DIR}")
print(f"Output : {FINAL_DIR}")
print()

# ---------------------------------------------------------
# Safety check
# ---------------------------------------------------------

if not SOURCE_DIR.exists():
    raise FileNotFoundError(
        f"Master dataset not found: {SOURCE_DIR}"
    )


# ---------------------------------------------------------
# Create output directories
# ---------------------------------------------------------

for class_name in CLASSES:

    class_dir = FINAL_DIR / class_name
    class_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Copy valid images
# ---------------------------------------------------------

total_found = 0
total_copied = 0
total_invalid = 0

invalid_files = []

class_counts = {}


for class_name in CLASSES:

    source_class_dir = SOURCE_DIR / class_name
    final_class_dir = FINAL_DIR / class_name

    if not source_class_dir.exists():
        print(
            f"WARNING: Missing class folder: "
            f"{source_class_dir}"
        )
        class_counts[class_name] = 0
        continue

    files = sorted(
        p for p in source_class_dir.iterdir()
        if p.is_file()
    )

    class_copied = 0
    class_invalid = 0

    print()
    print(f"Processing {class_name}...")
    print("-" * 70)

    for source_file in files:

        total_found += 1

        if not check_image(source_file):

            total_invalid += 1
            class_invalid += 1
            invalid_files.append(
                str(source_file)
            )

            continue

        destination = (
            final_class_dir
            / source_file.name
        )

        shutil.copy2(
            source_file,
            destination
        )

        total_copied += 1
        class_copied += 1

    class_counts[class_name] = class_copied

    print(
        f"Valid copied : {class_copied}"
    )

    print(
        f"Invalid      : {class_invalid}"
    )


# ---------------------------------------------------------
# Save invalid-image report
# ---------------------------------------------------------

REPORT_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "duplicate_reports"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

INVALID_REPORT = (
    REPORT_DIR
    / "final_dataset_invalid_images.txt"
)

with INVALID_REPORT.open(
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "INVALID IMAGES EXCLUDED FROM FINAL DATASET\n"
    )

    f.write("=" * 70 + "\n\n")

    for path in invalid_files:
        f.write(path + "\n")


# ---------------------------------------------------------
# Final summary
# ---------------------------------------------------------

print()
print("=" * 70)
print("FINAL DATASET SUMMARY")
print("=" * 70)

for class_name in CLASSES:

    print(
        f"{class_name:<15}: "
        f"{class_counts.get(class_name, 0):>6}"
    )

print("-" * 70)

print(
    f"{'TOTAL':<15}: "
    f"{total_copied:>6}"
)

print()
print(f"Files checked  : {total_found}")
print(f"Files copied   : {total_copied}")
print(f"Invalid files  : {total_invalid}")

print()
print("Invalid-image report:")
print(INVALID_REPORT)

print()
print("Final dataset:")
print(FINAL_DIR)

print()
print("=" * 70)
print("IMPORTANT")
print("=" * 70)

print()
print("The master dataset was NOT modified.")
print("The raw datasets were NOT modified.")
print("This script only created a clean copy.")