from pathlib import Path
from PIL import Image


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
    / "image_quality_report.txt"
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
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("IMAGE QUALITY CHECK")
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
    # Find images
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
    print("Checking images...")
    print()

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    valid_images = 0
    corrupt_images = []
    zero_size_images = []
    very_small_images = []
    grayscale_images = []

    # --------------------------------------------------------
    # Check each image
    # --------------------------------------------------------

    for index, image_path in enumerate(image_paths, start=1):

        relative_path = image_path.relative_to(PROJECT_ROOT)

        # ----------------------------------------------------
        # Zero-byte / empty file
        # ----------------------------------------------------

        try:

            file_size = image_path.stat().st_size

        except Exception as e:

            corrupt_images.append(
                (
                    relative_path,
                    f"Could not read file information: {e}"
                )
            )

            continue

        if file_size == 0:

            zero_size_images.append(relative_path)

            continue

        # ----------------------------------------------------
        # Open image
        # ----------------------------------------------------

        try:

            with Image.open(image_path) as image:

                # Verify image integrity
                image.verify()

            # Re-open after verify because verify() invalidates
            # the image object.
            with Image.open(image_path) as image:

                width, height = image.size
                mode = image.mode

        except Exception as e:

            corrupt_images.append(
                (
                    relative_path,
                    str(e)
                )
            )

            continue

        # ----------------------------------------------------
        # Very small image check
        # ----------------------------------------------------

        if width < 64 or height < 64:

            very_small_images.append(
                (
                    relative_path,
                    width,
                    height
                )
            )

        # ----------------------------------------------------
        # Grayscale check
        # ----------------------------------------------------

        if mode in ("1", "L", "I", "F"):

            grayscale_images.append(
                (
                    relative_path,
                    width,
                    height,
                    mode
                )
            )

        valid_images += 1

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if index % 500 == 0 or index == len(image_paths):

            print(
                f"Processed {index}/{len(image_paths)}"
            )

    # --------------------------------------------------------
    # Write report
    # --------------------------------------------------------

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as report:

        report.write(
            "IMAGE QUALITY REPORT\n"
        )

        report.write(
            "=" * 70 + "\n\n"
        )

        report.write(
            f"Total images scanned: {len(image_paths)}\n"
        )

        report.write(
            f"Valid images: {valid_images}\n"
        )

        report.write(
            f"Corrupt/unreadable images: "
            f"{len(corrupt_images)}\n"
        )

        report.write(
            f"Zero-size images: "
            f"{len(zero_size_images)}\n"
        )

        report.write(
            f"Very small images (<64x64): "
            f"{len(very_small_images)}\n"
        )

        report.write(
            f"Grayscale images: "
            f"{len(grayscale_images)}\n\n"
        )

        # ----------------------------------------------------
        # Corrupt
        # ----------------------------------------------------

        report.write(
            "\nCORRUPT / UNREADABLE IMAGES\n"
        )

        report.write(
            "-" * 70 + "\n"
        )

        if corrupt_images:

            for path, reason in corrupt_images:

                report.write(
                    f"{path}\n"
                )

                report.write(
                    f"Reason: {reason}\n\n"
                )

        else:

            report.write(
                "None\n"
            )

        # ----------------------------------------------------
        # Zero size
        # ----------------------------------------------------

        report.write(
            "\nZERO-SIZE IMAGES\n"
        )

        report.write(
            "-" * 70 + "\n"
        )

        if zero_size_images:

            for path in zero_size_images:

                report.write(
                    f"{path}\n"
                )

        else:

            report.write(
                "None\n"
            )

        # ----------------------------------------------------
        # Very small
        # ----------------------------------------------------

        report.write(
            "\nVERY SMALL IMAGES (<64x64)\n"
        )

        report.write(
            "-" * 70 + "\n"
        )

        if very_small_images:

            for path, width, height in very_small_images:

                report.write(
                    f"{path} | "
                    f"{width}x{height}\n"
                )

        else:

            report.write(
                "None\n"
            )

        # ----------------------------------------------------
        # Grayscale
        # ----------------------------------------------------

        report.write(
            "\nGRAYSCALE IMAGES\n"
        )

        report.write(
            "-" * 70 + "\n"
        )

        if grayscale_images:

            for path, width, height, mode in grayscale_images:

                report.write(
                    f"{path} | "
                    f"{width}x{height} | "
                    f"mode={mode}\n"
                )

        else:

            report.write(
                "None\n"
            )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)

    print()
    print(
        f"Total images scanned       : "
        f"{len(image_paths)}"
    )

    print(
        f"Valid images               : "
        f"{valid_images}"
    )

    print(
        f"Corrupt/unreadable images  : "
        f"{len(corrupt_images)}"
    )

    print(
        f"Zero-size images           : "
        f"{len(zero_size_images)}"
    )

    print(
        f"Very small images (<64x64) : "
        f"{len(very_small_images)}"
    )

    print(
        f"Grayscale images            : "
        f"{len(grayscale_images)}"
    )

    print()
    print("Report saved to:")
    print(REPORT_FILE)

    print()
    print("=" * 70)
    print("IMPORTANT")
    print("=" * 70)

    print()
    print("NO images were deleted, moved, or modified.")

    print(
        "We will inspect the report before deciding "
        "what should be removed."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()