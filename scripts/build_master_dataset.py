from pathlib import Path
import zipfile
import shutil
import re
from collections import Counter


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "datasets" / "raw"
CANDIDATE_DIR = PROJECT_ROOT / "datasets" / "processed" / "candidate"


# ============================================================
# SETTINGS
# ============================================================

DRY_RUN = False

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# FINAL PROJECT CLASSES
# ============================================================

FINAL_CLASSES = [
    "PLASTIC",
    "E_WASTE",
    "ORGANIC",
    "RECYCLABLE",
]


# ============================================================
# DATASETS
# ============================================================

DATASETS = {
    "trashnet": {
        "zip": RAW_DIR
        / "trashnet"
        / "trashnet-master"
        / "data"
        / "dataset-resized.zip",

        "type": "trashnet",
    },

    "garbage12": {
        "zip": RAW_DIR
        / "garbage12"
        / "archive.zip",

        "type": "garbage12",
    },

    "realwaste": {
        "zip": RAW_DIR
        / "realwaste"
        / "realwaste.zip",

        "type": "realwaste",
    },

    "ewaste": {
        "zip": RAW_DIR
        / "ewaste"
        / "Laptop Components Image Dataset to Classify Different Components.zip",

        "type": "ewaste",
    },
}


# ============================================================
# CLASS MAPPINGS
# ============================================================

TRASHNET_MAPPING = {
    "plastic": "PLASTIC",

    "paper": "RECYCLABLE",
    "cardboard": "RECYCLABLE",
    "glass": "RECYCLABLE",
    "metal": "RECYCLABLE",

    "trash": None,
}


GARBAGE12_MAPPING = {
    "plastic": "PLASTIC",

    "battery": "E_WASTE",

    "biological": "ORGANIC",

    "paper": "RECYCLABLE",
    "cardboard": "RECYCLABLE",
    "green-glass": "RECYCLABLE",
    "brown-glass": "RECYCLABLE",
    "white-glass": "RECYCLABLE",
    "metal": "RECYCLABLE",

    "clothes": None,
    "shoes": None,
    "trash": None,
}


REALWASTE_MAPPING = {
    "Plastic": "PLASTIC",

    "Food Organics": "ORGANIC",
    "Vegetation": "ORGANIC",

    "Cardboard": "RECYCLABLE",
    "Paper": "RECYCLABLE",
    "Glass": "RECYCLABLE",
    "Metal": "RECYCLABLE",

    "Miscellaneous Trash": None,
    "Textile Trash": None,
}


# ============================================================
# E-WASTE CLASSES
# ============================================================

EWASTE_CLASSES = {
    "Battery",
    "DVDRom",
    "Hinge",
    "Keyboard",
    "Motherboard",
    "DCCable",
    "PowerSwitch",
    "RAM",
    "ScrewKit",
    "USBPort",
    "WifiCard",
    "Speaker",
    "CMOSBattery",
    "HardDiskDrive",
    "WebCam",
    "LCDScreen",
    "Bezel",
    "HeatSink",
    "SSD",
    "CPUFan",
    "TouchPad",
    "Processor",
    "RAMCover",
    "BasePanel",
    "TopPanel",
    "LVDSCable",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def is_image(path):
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def normalize_class_name(name):
    """
    Clean class names.

    Example:

        1. Battery
        10. USBPort
        26. LVDSCable

    become:

        Battery
        USBPort
        LVDSCable
    """

    name = name.strip()

    match = re.match(r"^\d+\.\s*(.+)$", name)

    if match:
        name = match.group(1).strip()

    return name


def get_parent_class(path_inside_zip):
    """
    Get the folder immediately containing the image.
    """

    parts = Path(path_inside_zip).parts

    if len(parts) < 2:
        return None

    return parts[-2]


def get_ewaste_data_type(path_inside_zip):
    """
    Detect RAW vs AUGMENTED E-Waste data.
    """

    parts = [p.lower() for p in Path(path_inside_zip).parts]

    if "raw data" in parts:
        return "RAW"

    if "augmented data" in parts:
        return "AUGMENTED"

    return "UNKNOWN"


def get_unique_filename(dataset_name, original_path):
    """
    Create a safe unique filename.

    Example:

        trashnet__plastic__plastic_001.jpg
        realwaste__Plastic__Plastic_001.jpg
    """

    path = Path(original_path)

    parent = path.parent.name
    filename = path.name

    safe_parent = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        parent
    )

    safe_filename = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        filename
    )

    return f"{dataset_name}__{safe_parent}__{safe_filename}"


# ============================================================
# CLASS MAPPING
# ============================================================

def map_class(dataset_type, original_class):

    if dataset_type == "trashnet":
        return TRASHNET_MAPPING.get(original_class)

    if dataset_type == "garbage12":
        return GARBAGE12_MAPPING.get(original_class)

    if dataset_type == "realwaste":
        return REALWASTE_MAPPING.get(original_class)

    if dataset_type == "ewaste":

        if original_class in EWASTE_CLASSES:
            return "E_WASTE"

        return None

    return None


# ============================================================
# PREPARE DIRECTORIES
# ============================================================

def prepare_directories():

    if DRY_RUN:
        return

    CANDIDATE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for class_name in FINAL_CLASSES:

        class_dir = CANDIDATE_DIR / class_name

        class_dir.mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# PROCESS ONE DATASET
# ============================================================

def process_dataset(dataset_name, dataset_info):

    zip_path = dataset_info["zip"]
    dataset_type = dataset_info["type"]

    print("\n" + "=" * 70)
    print(f"PROCESSING DATASET: {dataset_name}")
    print("=" * 70)

    print(f"ZIP: {zip_path}")

    if not zip_path.exists():

        print("ERROR: ZIP FILE NOT FOUND")

        return Counter(), Counter()

    accepted_counts = Counter()
    skipped_counts = Counter()

    total_images = 0

    with zipfile.ZipFile(zip_path, "r") as z:

        members = z.namelist()

        for index, filename in enumerate(members, start=1):

            # ------------------------------------------------
            # Ignore non-images
            # ------------------------------------------------

            if not is_image(filename):
                continue

            total_images += 1

            # ------------------------------------------------
            # E-WASTE: RAW DATA ONLY
            # ------------------------------------------------

            if dataset_type == "ewaste":

                data_type = get_ewaste_data_type(filename)

                if data_type != "RAW":

                    skipped_counts[data_type] += 1

                    continue

            # ------------------------------------------------
            # ORIGINAL CLASS
            # ------------------------------------------------

            original_class = get_parent_class(filename)

            if original_class is None:

                skipped_counts["NO_CLASS_FOLDER"] += 1

                continue

            original_class = normalize_class_name(
                original_class
            )

            # ------------------------------------------------
            # FINAL CLASS
            # ------------------------------------------------

            final_class = map_class(
                dataset_type,
                original_class
            )

            if final_class is None:

                skipped_counts[original_class] += 1

                continue

            # ------------------------------------------------
            # OUTPUT PATH
            # ------------------------------------------------

            output_filename = get_unique_filename(
                dataset_name,
                filename
            )

            output_path = (
                CANDIDATE_DIR
                / final_class
                / output_filename
            )

            # ------------------------------------------------
            # COPY IMAGE
            # ------------------------------------------------

            if not DRY_RUN:

                with z.open(filename) as source:

                    with open(output_path, "wb") as destination:

                        shutil.copyfileobj(
                            source,
                            destination
                        )

            accepted_counts[final_class] += 1

            # ------------------------------------------------
            # PROGRESS
            # ------------------------------------------------

            total_processed = (
                sum(accepted_counts.values())
                + sum(skipped_counts.values())
            )

            if total_processed % 1000 == 0:

                print(
                    f"  Processed: "
                    f"{total_processed}/{total_images}"
                )

    # ========================================================
    # DATASET SUMMARY
    # ========================================================

    print("\nAccepted:")

    for class_name in FINAL_CLASSES:

        print(
            f"  {class_name:<15}"
            f"{accepted_counts[class_name]:>7}"
        )

    print("\nSkipped:")

    if skipped_counts:

        for reason, count in sorted(
            skipped_counts.items()
        ):

            print(
                f"  {reason:<25}"
                f"{count:>7}"
            )

    else:

        print("  None")

    print(
        f"\nAccepted total: "
        f"{sum(accepted_counts.values())}"
    )

    print(
        f"Skipped total : "
        f"{sum(skipped_counts.values())}"
    )

    return accepted_counts, skipped_counts


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AI POWERED WASTE SEGREGATION SYSTEM")
    print("CANDIDATE DATASET BUILDER")
    print("=" * 70)

    print("\nFinal classes:")

    for class_name in FINAL_CLASSES:

        print(f"  - {class_name}")

    print("\nMode:")

    if DRY_RUN:

        print("  DRY RUN")
        print("  No files will be created.")

    else:

        print("  REAL BUILD")
        print("  Candidate images WILL be created.")

    print("\nOutput directory:")

    print(f"  {CANDIDATE_DIR}")

    # --------------------------------------------------------
    # Prepare folders
    # --------------------------------------------------------

    prepare_directories()

    total_accepted = Counter()
    total_skipped = Counter()

    # --------------------------------------------------------
    # Process datasets
    # --------------------------------------------------------

    for dataset_name, dataset_info in DATASETS.items():

        accepted, skipped = process_dataset(
            dataset_name,
            dataset_info
        )

        total_accepted.update(accepted)
        total_skipped.update(skipped)

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n\n" + "=" * 70)
    print("CANDIDATE DATASET BUILD SUMMARY")
    print("=" * 70)

    print("\nFinal dataset:")

    for class_name in FINAL_CLASSES:

        print(
            f"  {class_name:<15}"
            f"{total_accepted[class_name]:>7}"
        )

    print("\n" + "-" * 70)

    print(
        f"TOTAL ACCEPTED : "
        f"{sum(total_accepted.values())}"
    )

    print(
        f"TOTAL SKIPPED  : "
        f"{sum(total_skipped.values())}"
    )

    print("\n" + "=" * 70)

    if DRY_RUN:

        print("DRY RUN COMPLETE")

    else:

        print("CANDIDATE DATASET CREATED")

    print("=" * 70)


if __name__ == "__main__":
    main()