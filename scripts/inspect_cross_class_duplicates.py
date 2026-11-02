from pathlib import Path
import csv
from collections import defaultdict


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORT_FILE = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "duplicate_reports"
    / "exact_duplicates.csv"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CROSS-CLASS EXACT DUPLICATE INSPECTION")
    print("=" * 70)

    if not REPORT_FILE.exists():
        print()
        print("ERROR: Duplicate report not found.")
        print()
        print(f"Expected:")
        print(REPORT_FILE)
        return

    # --------------------------------------------------------
    # Read duplicate report
    # --------------------------------------------------------

    groups = defaultdict(list)

    with open(
        REPORT_FILE,
        "r",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        reader = csv.DictReader(csv_file)

        for row in reader:

            group_id = row["duplicate_group"]

            groups[group_id].append({
                "sha256": row["sha256"],
                "class": row["class"],
                "file_path": row["file_path"]
            })

    # --------------------------------------------------------
    # Find cross-class duplicate groups
    # --------------------------------------------------------

    cross_class_groups = []

    for group_id, files in groups.items():

        classes = {
            item["class"]
            for item in files
        }

        if len(classes) > 1:

            cross_class_groups.append(
                (group_id, files)
            )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()
    print(
        f"Cross-class duplicate groups found: "
        f"{len(cross_class_groups)}"
    )

    print()

    if not cross_class_groups:

        print("GOOD: No cross-class exact duplicates found.")

    else:

        for group_id, files in cross_class_groups:

            print("-" * 70)
            print(f"DUPLICATE GROUP: {group_id}")
            print(f"SHA-256: {files[0]['sha256']}")
            print()

            for item in files:

                print(f"CLASS: {item['class']}")
                print(f"FILE : {item['file_path']}")
                print()

    print("=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)

    print()
    print("IMPORTANT:")
    print("No files were deleted, moved, or modified.")


if __name__ == "__main__":
    main()