from pathlib import Path
from PIL import Image
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "datasets" / "split_grouped"

CLASSES = ["PLASTIC", "E_WASTE", "ORGANIC", "RECYCLABLE"]
SPLITS = ["train", "val", "test"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_images(folder):
    return [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def main():
    print("=" * 70)
    print("FINAL DATASET SANITY CHECK")
    print("=" * 70)

    if not DATASET_DIR.exists():
        print(f"\nERROR: Dataset directory not found:")
        print(DATASET_DIR)
        return

    print(f"\nDataset:")
    print(DATASET_DIR)

    total_images = 0
    total_invalid = 0

    all_filenames = Counter()

    print("\n" + "-" * 70)
    print("CLASS COUNTS")
    print("-" * 70)

    for split in SPLITS:
        split_dir = DATASET_DIR / split

        if not split_dir.exists():
            print(f"\nERROR: Missing split folder: {split}")
            continue

        print(f"\n[{split.upper()}]")

        split_total = 0

        for class_name in CLASSES:
            class_dir = split_dir / class_name

            if not class_dir.exists():
                print(f"  {class_name:<12} MISSING")
                continue

            images = get_images(class_dir)

            valid_count = 0
            invalid_count = 0

            for image_path in images:
                all_filenames[image_path.name] += 1

                try:
                    with Image.open(image_path) as img:
                        img.verify()

                    # Open again after verify()
                    with Image.open(image_path) as img:
                        img.load()

                    valid_count += 1

                except Exception as e:
                    invalid_count += 1
                    total_invalid += 1

                    print(f"    INVALID: {image_path}")
                    print(f"             {e}")

            print(
                f"  {class_name:<12} "
                f"{valid_count:>5} valid | "
                f"{invalid_count:>3} invalid"
            )

            split_total += valid_count
            total_images += valid_count

        print(f"  {'TOTAL':<12} {split_total:>5}")

    print("\n" + "-" * 70)
    print("DUPLICATE FILENAME CHECK")
    print("-" * 70)

    duplicate_names = {
        name: count
        for name, count in all_filenames.items()
        if count > 1
    }

    if duplicate_names:
        print(f"Found {len(duplicate_names)} duplicated filenames.")

        for name, count in list(duplicate_names.items())[:20]:
            print(f"  {name} -> {count} occurrences")

    else:
        print("No duplicate filenames found.")

    print("\n" + "-" * 70)
    print("FINAL SUMMARY")
    print("-" * 70)

    print(f"Total valid images : {total_images}")
    print(f"Total invalid      : {total_invalid}")
    print(f"Expected images    : 15109")

    if total_images == 15109 and total_invalid == 0:
        print("\n✅ DATASET SANITY CHECK PASSED")
        print("✅ Ready for model training.")

    else:
        print("\n⚠️ DATASET SANITY CHECK DID NOT MATCH EXPECTED VALUES.")
        print("Do NOT start training yet.")

    print("=" * 70)


if __name__ == "__main__":
    main()