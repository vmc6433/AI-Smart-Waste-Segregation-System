from pathlib import Path
import hashlib
import csv
from collections import defaultdict

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CANDIDATE_DIR = PROJECT_ROOT / "datasets" / "processed" / "candidate"

REPORT_DIR = PROJECT_ROOT / "datasets" / "processed" / "duplicate_reports"

REPORT_FILE = REPORT_DIR / "exact_duplicates.csv"


# ============================================================
# FUNCTIONS
# ============================================================

def calculate_sha256(file_path):
    """
    Calculate SHA-256 hash of an image file.
    Files are read in chunks so large files don't consume
    excessive memory.
    """
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("EXACT DUPLICATE DETECTION")
    print("=" * 60)

    if not CANDIDATE_DIR.exists():
        print()
        print("ERROR: Candidate dataset was not found.")
        print(f"Expected location:")
        print(CANDIDATE_DIR)
        return

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Supported image extensions
    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
        ".tif",
        ".tiff"
    }

    # hash -> list of files
    hash_to_files = defaultdict(list)

    total_images = 0

    print()
    print("Scanning images...")
    print()

    # --------------------------------------------------------
    # Scan every class
    # --------------------------------------------------------

    for class_dir in sorted(CANDIDATE_DIR.iterdir()):

        if not class_dir.is_dir():
            continue

        class_name = class_dir.name

        class_count = 0

        for image_path in class_dir.rglob("*"):

            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in image_extensions:
                continue

            try:
                file_hash = calculate_sha256(image_path)

                hash_to_files[file_hash].append(
                    (class_name, image_path)
                )

                total_images += 1
                class_count += 1

            except Exception as e:
                print(f"ERROR reading: {image_path}")
                print(f"       {e}")

        print(f"{class_name:<15} {class_count:>6} images")

    # --------------------------------------------------------
    # Find duplicate groups
    # --------------------------------------------------------

    duplicate_groups = []

    for file_hash, files in hash_to_files.items():

        if len(files) > 1:

            duplicate_groups.append(
                (file_hash, files)
            )

    # Sort duplicate groups for consistent output
    duplicate_groups.sort(key=lambda x: x[0])

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    unique_images = len(hash_to_files)

    duplicate_files = sum(
        len(files)
        for _, files in duplicate_groups
    )

    extra_duplicate_files = sum(
        len(files) - 1
        for _, files in duplicate_groups
    )

    cross_class_groups = 0

    for _, files in duplicate_groups:

        classes = {
            class_name
            for class_name, _ in files
        }

        if len(classes) > 1:
            cross_class_groups += 1

    # --------------------------------------------------------
    # Write CSV report
    # --------------------------------------------------------

    with open(
        REPORT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.writer(csv_file)

        writer.writerow([
            "duplicate_group",
            "sha256",
            "class",
            "file_path"
        ])

        group_number = 1

        for file_hash, files in duplicate_groups:

            for class_name, image_path in files:

                relative_path = image_path.relative_to(
                    PROJECT_ROOT
                )

                writer.writerow([
                    group_number,
                    file_hash,
                    class_name,
                    str(relative_path)
                ])

            group_number += 1

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("RESULT")
    print("=" * 60)

    print()
    print(f"Total images scanned       : {total_images}")
    print(f"Unique image files         : {unique_images}")
    print(f"Duplicate groups           : {len(duplicate_groups)}")
    print(f"Images in duplicate groups : {duplicate_files}")
    print(f"Extra duplicate files      : {extra_duplicate_files}")
    print(f"Cross-class duplicate groups: {cross_class_groups}")

    print()
    print("Report saved to:")
    print(REPORT_FILE)

    print()
    print("=" * 60)

    if len(duplicate_groups) == 0:

        print("GOOD: No exact duplicates found.")

    else:

        print(
            "Exact duplicates were found."
        )

        print(
            "DO NOT DELETE ANYTHING YET."
        )

        print(
            "We will inspect the duplicate report first."
        )

    print("=" * 60)


if __name__ == "__main__":
    main()