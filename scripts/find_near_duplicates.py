from pathlib import Path
from collections import defaultdict
import csv

from PIL import Image
import imagehash


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MASTER_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "master"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "duplicate_reports"
)

REPORT_FILE = (
    REPORT_DIR
    / "near_duplicates.csv"
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff"
}

# Maximum perceptual-hash distance considered a possible
# near duplicate.
#
# IMPORTANT:
# This is intentionally conservative.
# We are detecting candidates, NOT automatically deleting.
PHASH_THRESHOLD = 6


# ============================================================
# FUNCTIONS
# ============================================================

def calculate_phash(image_path):

    with Image.open(image_path) as image:

        # Convert to RGB so all image formats are handled
        # consistently.
        image = image.convert("RGB")

        return imagehash.phash(image)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("NEAR-DUPLICATE IMAGE DETECTION")
    print("=" * 70)

    if not MASTER_DIR.exists():

        print()
        print("ERROR: Master dataset not found.")
        print(MASTER_DIR)
        return

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Find all images
    # --------------------------------------------------------

    image_paths = []

    for class_dir in sorted(MASTER_DIR.iterdir()):

        if not class_dir.is_dir():
            continue

        for image_path in class_dir.rglob("*"):

            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            image_paths.append(image_path)

    print()
    print(f"Images found: {len(image_paths)}")
    print()

    # --------------------------------------------------------
    # Calculate perceptual hashes
    # --------------------------------------------------------

    print("Calculating perceptual hashes...")
    print()

    image_hashes = []

    errors = 0

    for index, image_path in enumerate(image_paths, start=1):

        try:

            phash = calculate_phash(image_path)

            image_hashes.append(
                (
                    image_path,
                    phash
                )
            )

        except Exception as e:

            errors += 1

            print()
            print("ERROR:")
            print(image_path)
            print(e)

        # Progress every 500 images
        if index % 500 == 0 or index == len(image_paths):

            print(
                f"Processed {index}/{len(image_paths)}"
            )

    # --------------------------------------------------------
    # Compare hashes
    # --------------------------------------------------------

    print()
    print("Comparing perceptual hashes...")
    print()
    print(
        "This may take some time because "
        "images are being compared with each other."
    )

    near_duplicate_pairs = []

    total_hashes = len(image_hashes)

    # --------------------------------------------------------
    # Pairwise comparison
    # --------------------------------------------------------

    for i in range(total_hashes):

        path_a, hash_a = image_hashes[i]

        for j in range(i + 1, total_hashes):

            path_b, hash_b = image_hashes[j]

            distance = hash_a - hash_b

            if distance <= PHASH_THRESHOLD:

                class_a = path_a.parent.name
                class_b = path_b.parent.name

                near_duplicate_pairs.append(
                    (
                        distance,
                        class_a,
                        str(
                            path_a.relative_to(PROJECT_ROOT)
                        ),
                        class_b,
                        str(
                            path_b.relative_to(PROJECT_ROOT)
                        )
                    )
                )

    # --------------------------------------------------------
    # Sort results
    # --------------------------------------------------------

    near_duplicate_pairs.sort(
        key=lambda x: x[0]
    )

    # --------------------------------------------------------
    # Write CSV
    # --------------------------------------------------------

    with open(
        REPORT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.writer(csv_file)

        writer.writerow([
            "phash_distance",
            "class_a",
            "file_a",
            "class_b",
            "file_b"
        ])

        writer.writerows(
            near_duplicate_pairs
        )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    cross_class_pairs = 0

    for pair in near_duplicate_pairs:

        if pair[1] != pair[3]:

            cross_class_pairs += 1

    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)

    print()
    print(
        f"Images successfully hashed : {len(image_hashes)}"
    )

    print(
        f"Hashing errors              : {errors}"
    )

    print(
        f"Near-duplicate pairs        : "
        f"{len(near_duplicate_pairs)}"
    )

    print(
        f"Cross-class near-duplicate pairs: "
        f"{cross_class_pairs}"
    )

    print()
    print("pHash threshold:")
    print(PHASH_THRESHOLD)

    print()
    print("Report saved to:")
    print(REPORT_FILE)

    print()
    print("=" * 70)
    print("IMPORTANT")
    print("=" * 70)

    print()
    print("This script ONLY detects possible near duplicates.")

    print(
        "NO images were deleted, moved, or modified."
    )

    print(
        "We will inspect the results before deciding "
        "what, if anything, should be removed."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()