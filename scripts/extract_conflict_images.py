from pathlib import Path
import shutil


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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "duplicate_reports"
    / "cross_class_inspection"
)


# ============================================================
# CONFLICTING IMAGE PATHS
# ============================================================

CONFLICTING_IMAGES = [
    (
        "group_1817",
        "PLASTIC",
        "garbage12__plastic__plastic556.jpg"
    ),
    (
        "group_1817",
        "RECYCLABLE",
        "garbage12__white-glass__white-glass761.jpg"
    ),
    (
        "group_2148",
        "PLASTIC",
        "garbage12__plastic__plastic159.jpg"
    ),
    (
        "group_2148",
        "RECYCLABLE",
        "garbage12__white-glass__white-glass129.jpg"
    ),
]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("EXTRACTING CROSS-CLASS CONFLICT IMAGES")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    copied = 0

    for group, class_name, filename in CONFLICTING_IMAGES:

        source = CANDIDATE_DIR / class_name / filename

        if not source.exists():

            print()
            print("WARNING: File not found:")
            print(source)

            continue

        destination_folder = OUTPUT_DIR / group / class_name

        destination_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        destination = destination_folder / filename

        shutil.copy2(source, destination)

        print()
        print(f"Copied: {filename}")
        print(f"Class : {class_name}")
        print(f"Group : {group}")

        copied += 1

    print()
    print("=" * 70)
    print(f"Images copied: {copied}")
    print()
    print("Inspection folder:")
    print(OUTPUT_DIR)
    print("=" * 70)

    print()
    print("No original candidate images were modified.")


if __name__ == "__main__":
    main()