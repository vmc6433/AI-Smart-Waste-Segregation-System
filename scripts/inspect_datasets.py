from pathlib import Path
from PIL import Image
import zipfile


# ============================================================
# AI POWERED WASTE SEGREGATION SYSTEM
# Dataset Inspection Tool
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "datasets" / "raw"


def is_image_file(path):
    """Check whether a file looks like an image."""
    return path.suffix.lower() in {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
        ".gif",
    }


def inspect_zip(zip_path):
    """Inspect a ZIP file without modifying/extracting it."""

    print("\n" + "=" * 70)
    print(f"ZIP FILE: {zip_path.name}")
    print("=" * 70)

    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            files = z.namelist()

        image_files = [
            f for f in files
            if is_image_file(Path(f))
        ]

        print(f"Total files in ZIP : {len(files)}")
        print(f"Image files         : {len(image_files)}")

        # Show directory/class structure
        folders = set()

        for file in image_files:
            parts = Path(file).parts

            if len(parts) >= 2:
                folders.add(parts[-2])

        if folders:
            print("\nPossible classes/folders:")
            for folder in sorted(folders):
                count = sum(
                    1
                    for f in image_files
                    if Path(f).parent.name == folder
                )
                print(f"  {folder:<30} {count:>6} images")

    except zipfile.BadZipFile:
        print("ERROR: This ZIP file appears to be corrupted.")

    except Exception as e:
        print(f"ERROR: {e}")


def inspect_extracted_folder(folder):
    """Inspect an already extracted dataset folder."""

    print("\n" + "=" * 70)
    print(f"FOLDER: {folder}")
    print("=" * 70)

    image_files = [
        p
        for p in folder.rglob("*")
        if p.is_file() and is_image_file(p)
    ]

    print(f"Image files found: {len(image_files)}")

    if not image_files:
        print("No images found.")
        return

    # Count images by immediate parent folder
    class_counts = {}

    for image in image_files:
        class_name = image.parent.name
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

    print("\nPossible classes/folders:")

    for class_name, count in sorted(class_counts.items()):
        print(f"  {class_name:<30} {count:>6} images")


def main():

    print("=" * 70)
    print("AI POWERED WASTE SEGREGATION SYSTEM")
    print("DATASET INSPECTION")
    print("=" * 70)

    if not RAW_DIR.exists():
        print("\nERROR: Raw dataset directory does not exist.")
        return

    # --------------------------------------------------------
    # Inspect ZIP files
    # --------------------------------------------------------

    zip_files = list(RAW_DIR.rglob("*.zip"))

    if not zip_files:
        print("\nNo ZIP files found.")
    else:

        print(f"\nZIP files found: {len(zip_files)}")

        for zip_file in sorted(zip_files):
            inspect_zip(zip_file)

    # --------------------------------------------------------
    # Inspect already extracted folders
    # --------------------------------------------------------

    dataset_folders = [
        RAW_DIR / "trashnet",
        RAW_DIR / "garbage12",
        RAW_DIR / "realwaste",
        RAW_DIR / "ewaste",
        RAW_DIR / "waste_classification",
    ]

    for folder in dataset_folders:

        if folder.exists():

            # Ignore folders containing only ZIP files
            images = [
                p
                for p in folder.rglob("*")
                if p.is_file() and is_image_file(p)
            ]

            if images:
                inspect_extracted_folder(folder)

    print("\n" + "=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()