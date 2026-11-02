from pathlib import Path
import hashlib
import shutil
import csv
from collections import defaultdict


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CANDIDATE_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "candidate"
)

MASTER_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "master"
)

DUPLICATE_REPORT = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "duplicate_reports"
    / "exact_duplicates.csv"
)

CLEANUP_REPORT = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "duplicate_reports"
    / "master_cleanup.csv"
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


# ============================================================
# FUNCTIONS
# ============================================================

def calculate_sha256(file_path):

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

    print("=" * 70)
    print("BUILDING CLEAN MASTER DATASET")
    print("=" * 70)

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    if not CANDIDATE_DIR.exists():

        print()
        print("ERROR: Candidate dataset not found.")
        print(CANDIDATE_DIR)
        return

    if not DUPLICATE_REPORT.exists():

        print()
        print("ERROR: Exact duplicate report not found.")
        print(DUPLICATE_REPORT)
        return

    # --------------------------------------------------------
    # Create master directory
    # --------------------------------------------------------

    MASTER_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    classes = [
        "PLASTIC",
        "E_WASTE",
        "ORGANIC",
        "RECYCLABLE"
    ]

    for class_name in classes:

        (MASTER_DIR / class_name).mkdir(
            parents=True,
            exist_ok=True
        )

    # --------------------------------------------------------
    # Read duplicate report
    # --------------------------------------------------------

    print()
    print("Reading exact duplicate report...")

    groups = defaultdict(list)

    with open(
        DUPLICATE_REPORT,
        "r",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        reader = csv.DictReader(csv_file)

        for row in reader:

            groups[row["duplicate_group"]].append({
                "sha256": row["sha256"],
                "class": row["class"],
                "file_path": row["file_path"]
            })

    # --------------------------------------------------------
    # Determine which files to keep
    # --------------------------------------------------------

    keep_files = set()
    skip_files = {}

    total_duplicate_groups = 0

    for group_id, files in groups.items():

        if len(files) <= 1:
            continue

        total_duplicate_groups += 1

        classes_in_group = {
            item["class"]
            for item in files
        }

        # ----------------------------------------------------
        # Cross-class duplicate
        # ----------------------------------------------------

        if len(classes_in_group) > 1:

            # Our confirmed rule:
            # PLASTIC wins over RECYCLABLE for the
            # two known conflicts.

            if "PLASTIC" in classes_in_group:

                selected = None

                for item in files:

                    if item["class"] == "PLASTIC":

                        selected = item
                        break

                keep_files.add(selected["file_path"])

                for item in files:

                    if item["file_path"] != selected["file_path"]:

                        skip_files[
                            item["file_path"]
                        ] = (
                            f"Exact duplicate conflict; "
                            f"keeping PLASTIC copy "
                            f"(group {group_id})"
                        )

            else:

                # Safety fallback.
                # This should not occur with our current data.

                selected = files[0]

                keep_files.add(selected["file_path"])

                for item in files[1:]:

                    skip_files[
                        item["file_path"]
                    ] = (
                        f"Exact duplicate conflict; "
                        f"keeping first class "
                        f"(group {group_id})"
                    )

        # ----------------------------------------------------
        # Same-class duplicate
        # ----------------------------------------------------

        else:

            # Keep exactly one copy.
            selected = files[0]

            keep_files.add(selected["file_path"])

            for item in files[1:]:

                skip_files[
                    item["file_path"]
                ] = (
                    f"Exact duplicate; "
                    f"same class (group {group_id})"
                )

    # --------------------------------------------------------
    # Find all candidate images
    # --------------------------------------------------------

    print()
    print("Scanning candidate dataset...")

    all_images = []

    for class_dir in sorted(CANDIDATE_DIR.iterdir()):

        if not class_dir.is_dir():
            continue

        for image_path in class_dir.rglob("*"):

            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            relative_path = image_path.relative_to(
                PROJECT_ROOT
            )

            all_images.append(str(relative_path))

    # --------------------------------------------------------
    # Add non-duplicate images to keep list
    # --------------------------------------------------------

    duplicate_paths = set()

    for files in groups.values():

        for item in files:

            duplicate_paths.add(
                item["file_path"]
            )

    for relative_path in all_images:

        if relative_path not in duplicate_paths:

            keep_files.add(relative_path)

    # --------------------------------------------------------
    # Copy selected images
    # --------------------------------------------------------

    print()
    print("Copying selected images into master dataset...")
    print()

    copied = 0
    skipped = 0

    cleanup_rows = []

    for relative_path in all_images:

        source = PROJECT_ROOT / relative_path

        class_name = source.parent.name

        if relative_path in skip_files:

            skipped += 1

            cleanup_rows.append([
                relative_path,
                class_name,
                "SKIPPED",
                skip_files[relative_path]
            ])

            continue

        if relative_path not in keep_files:

            skipped += 1

            cleanup_rows.append([
                relative_path,
                class_name,
                "SKIPPED",
                "Duplicate not selected"
            ])

            continue

        destination = (
            MASTER_DIR
            / class_name
            / source.name
        )

        # Extremely unlikely filename collision safeguard.
        if destination.exists():

            stem = source.stem
            suffix = source.suffix

            counter = 1

            while destination.exists():

                destination = (
                    MASTER_DIR
                    / class_name
                    / f"{stem}__copy{counter}{suffix}"
                )

                counter += 1

        shutil.copy2(
            source,
            destination
        )

        copied += 1

        cleanup_rows.append([
            relative_path,
            class_name,
            "KEPT",
            "Selected for clean master"
        ])

    # --------------------------------------------------------
    # Write cleanup report
    # --------------------------------------------------------

    with open(
        CLEANUP_REPORT,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.writer(csv_file)

        writer.writerow([
            "candidate_file",
            "class",
            "action",
            "reason"
        ])

        writer.writerows(cleanup_rows)

    # --------------------------------------------------------
    # Count final master dataset
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("MASTER DATASET CREATED")
    print("=" * 70)

    print()

    total_master = 0

    for class_name in classes:

        class_dir = MASTER_DIR / class_name

        count = sum(
            1
            for p in class_dir.rglob("*")
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
        )

        total_master += count

        print(
            f"{class_name:<15} {count:>6} images"
        )

    print("-" * 70)

    print(
        f"{'TOTAL':<15} {total_master:>6} images"
    )

    print()
    print(f"Exact duplicate groups processed: {total_duplicate_groups}")
    print(f"Images copied to master          : {copied}")
    print(f"Images skipped                   : {skipped}")

    print()
    print("Master dataset:")
    print(MASTER_DIR)

    print()
    print("Cleanup report:")
    print(CLEANUP_REPORT)

    print()
    print("=" * 70)
    print("IMPORTANT")
    print("=" * 70)
    print()
    print("RAW datasets were NOT modified.")
    print("CANDIDATE dataset was NOT modified.")
    print("Only the new MASTER dataset was created.")
    print()


if __name__ == "__main__":
    main()