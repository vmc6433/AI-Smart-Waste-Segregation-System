from pathlib import Path
import csv
import random
import shutil
from collections import defaultdict

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FINAL_DIR = PROJECT_ROOT / "datasets" / "final"

NEAR_DUPLICATE_FILE = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "duplicate_reports"
    / "near_duplicates.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "datasets" / "split_grouped"

REPORT_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "duplicate_reports"
)

GROUP_REPORT = REPORT_DIR / "grouped_split_report.csv"

CLASSES = [
    "PLASTIC",
    "E_WASTE",
    "ORGANIC",
    "RECYCLABLE",
]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

MAX_DISTANCE = 2

SEED = 42


# ============================================================
# UNION-FIND
# ============================================================

class UnionFind:

    def __init__(self):
        self.parent = {}
        self.rank = {}

    def add(self, x):

        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x):

        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])

        return self.parent[x]

    def union(self, a, b):

        self.add(a)
        self.add(b)

        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            return

        if self.rank[root_a] < self.rank[root_b]:

            self.parent[root_a] = root_b

        elif self.rank[root_a] > self.rank[root_b]:

            self.parent[root_b] = root_a

        else:

            self.parent[root_b] = root_a
            self.rank[root_a] += 1


# ============================================================
# START
# ============================================================

print("=" * 70)
print("CREATING CORRECTED GROUP-AWARE DATASET SPLIT")
print("=" * 70)

print(f"\nFinal dataset  : {FINAL_DIR}")
print(f"Near-duplicates: {NEAR_DUPLICATE_FILE}")
print(f"Output         : {OUTPUT_DIR}")
print(f"Threshold      : pHash distance <= {MAX_DISTANCE}")
print(f"Random seed    : {SEED}")


# ============================================================
# CHECK SOURCE
# ============================================================

if not FINAL_DIR.exists():

    print("\nERROR: Final dataset not found.")
    raise SystemExit(1)

if not NEAR_DUPLICATE_FILE.exists():

    print("\nERROR: Near-duplicate report not found.")
    raise SystemExit(1)


# ============================================================
# COLLECT FINAL IMAGES
# ============================================================

print("\nCollecting final dataset images...")

all_images = []
image_class = {}

# Key:
# (class_name, filename)
#
# This allows us to match the CSV even when the absolute
# directory paths are different.

image_lookup = {}

for class_name in CLASSES:

    class_dir = FINAL_DIR / class_name

    if not class_dir.exists():

        print(f"\nERROR: Missing class directory:")
        print(class_dir)
        raise SystemExit(1)

    images = [
        p.resolve()
        for p in class_dir.iterdir()
        if p.is_file()
    ]

    for image in images:

        all_images.append(image)

        image_class[image] = class_name

        key = (
            class_name.upper(),
            image.name.lower()
        )

        image_lookup[key] = image


print(f"Total images: {len(all_images)}")


# ============================================================
# INITIALIZE UNION-FIND
# ============================================================

uf = UnionFind()

for image in all_images:
    uf.add(image)


# ============================================================
# READ NEAR-DUPLICATE PAIRS
# ============================================================

print("\nReading near-duplicate relationships...")

pairs_used = 0
pairs_ignored = 0

with open(
    NEAR_DUPLICATE_FILE,
    "r",
    encoding="utf-8-sig",
    newline=""
) as f:

    reader = csv.DictReader(f)

    print("Columns found:")
    print(reader.fieldnames)

    required = {
        "phash_distance",
        "class_a",
        "file_a",
        "class_b",
        "file_b",
    }

    if not required.issubset(reader.fieldnames):

        print("\nERROR: Unexpected CSV columns.")
        raise SystemExit(1)

    for row in reader:

        distance = int(row["phash_distance"])

        if distance > MAX_DISTANCE:
            continue

        class_a = row["class_a"].strip().upper()
        class_b = row["class_b"].strip().upper()

        # ----------------------------------------------------
        # Extract filename only.
        # ----------------------------------------------------

        file_a_name = Path(row["file_a"]).name.lower()
        file_b_name = Path(row["file_b"]).name.lower()

        key_a = (
            class_a,
            file_a_name
        )

        key_b = (
            class_b,
            file_b_name
        )

        image_a = image_lookup.get(key_a)
        image_b = image_lookup.get(key_b)

        if image_a is None or image_b is None:

            pairs_ignored += 1
            continue

        uf.union(image_a, image_b)

        pairs_used += 1


print(f"\nStrong near-duplicate pairs used : {pairs_used}")
print(f"Pairs ignored                     : {pairs_ignored}")


# ============================================================
# BUILD GROUPS
# ============================================================

print("\nBuilding groups...")

groups = defaultdict(list)

for image in all_images:

    root = uf.find(image)

    groups[root].append(image)

group_list = list(groups.values())

print(f"Total groups: {len(group_list)}")

# Count how many images participate in actual groups.
multi_image_groups = [
    group
    for group in group_list
    if len(group) > 1
]

images_in_groups = sum(
    len(group)
    for group in multi_image_groups
)

print(f"Multi-image groups: {len(multi_image_groups)}")
print(f"Images in groups  : {images_in_groups}")


# ============================================================
# GROUP INFORMATION
# ============================================================

group_info = []

for group_id, images in enumerate(group_list, start=1):

    class_counts = defaultdict(int)

    for image in images:

        class_counts[
            image_class[image]
        ] += 1

    group_info.append({
        "group_id": group_id,
        "images": images,
        "size": len(images),
        "class_counts": dict(class_counts),
    })


# ============================================================
# RANDOMIZE + SORT LARGE GROUPS FIRST
# ============================================================

random.seed(SEED)

random.shuffle(group_info)

group_info.sort(
    key=lambda x: x["size"],
    reverse=True
)


# ============================================================
# TARGET COUNTS
# ============================================================

total_images = len(all_images)

target = {
    "train": total_images * TRAIN_RATIO,
    "val": total_images * VAL_RATIO,
    "test": total_images * TEST_RATIO,
}

print("\nTarget image counts:")

for split in ["train", "val", "test"]:

    print(
        f"{split:5} : {target[split]:.1f}"
    )


# ============================================================
# ASSIGN WHOLE GROUPS
# ============================================================

split_images = {
    "train": 0,
    "val": 0,
    "test": 0,
}

split_groups = {
    "train": [],
    "val": [],
    "test": [],
}

split_class_counts = {
    "train": defaultdict(int),
    "val": defaultdict(int),
    "test": defaultdict(int),
}


for group in group_info:

    # --------------------------------------------------------
    # Find split with largest remaining capacity.
    # --------------------------------------------------------

    deficits = {}

    for split in ["train", "val", "test"]:

        deficits[split] = (
            target[split]
            - split_images[split]
        )

    best_split = max(
        deficits,
        key=deficits.get
    )

    split_groups[best_split].append(group)

    split_images[best_split] += group["size"]

    for class_name, count in group["class_counts"].items():

        split_class_counts[
            best_split
        ][class_name] += count


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

print("\nCreating output directories...")

# We intentionally remove the previous incorrect
# split_grouped directory so old files cannot remain.

if OUTPUT_DIR.exists():

    print("Removing previous incorrect split_grouped directory...")

    shutil.rmtree(OUTPUT_DIR)


for split in ["train", "val", "test"]:

    for class_name in CLASSES:

        (
            OUTPUT_DIR
            / split
            / class_name
        ).mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# COPY FILES
# ============================================================

print("\nCopying files...")

copied = 0

for split in ["train", "val", "test"]:

    print(f"\n{split.upper()}")
    print("-" * 70)

    for group in split_groups[split]:

        for image in group["images"]:

            class_name = image_class[image]

            destination = (
                OUTPUT_DIR
                / split
                / class_name
                / image.name
            )

            shutil.copy2(
                image,
                destination
            )

            copied += 1


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CORRECTED GROUP-AWARE SPLIT SUMMARY")
print("=" * 70)

print("\nOverall:")

print(
    f"TRAIN      : {split_images['train']}"
)

print(
    f"VALIDATION : {split_images['val']}"
)

print(
    f"TEST       : {split_images['test']}"
)

print("-" * 70)

print(
    f"TOTAL      : {sum(split_images.values())}"
)

print(
    f"COPIED     : {copied}"
)


print("\nBy class:")

print(
    f"{'CLASS':15}"
    f"{'TRAIN':>10}"
    f"{'VAL':>10}"
    f"{'TEST':>10}"
    f"{'TOTAL':>10}"
)

print("-" * 55)

for class_name in CLASSES:

    train_count = (
        split_class_counts["train"][class_name]
    )

    val_count = (
        split_class_counts["val"][class_name]
    )

    test_count = (
        split_class_counts["test"][class_name]
    )

    total = (
        train_count
        + val_count
        + test_count
    )

    print(
        f"{class_name:15}"
        f"{train_count:10}"
        f"{val_count:10}"
        f"{test_count:10}"
        f"{total:10}"
    )


# ============================================================
# SAVE GROUP REPORT
# ============================================================

print("\nSaving group report...")

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    GROUP_REPORT,
    "w",
    encoding="utf-8",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "group_id",
        "assigned_split",
        "group_size",
        "class_counts",
        "files",
    ])

    for split in ["train", "val", "test"]:

        for group in split_groups[split]:

            class_counts = "; ".join(
                f"{key}={value}"
                for key, value
                in group["class_counts"].items()
            )

            files = "; ".join(
                str(image)
                for image in group["images"]
            )

            writer.writerow([
                group["group_id"],
                split,
                group["size"],
                class_counts,
                files,
            ])


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("CORRECTED GROUP-AWARE SPLIT COMPLETE")
print("=" * 70)

print("\nOutput:")
print(OUTPUT_DIR)

print("\nGroup report:")
print(GROUP_REPORT)

print("\nIMPORTANT:")
print("Final dataset was NOT modified.")
print("Previous random split was NOT modified.")
print("The old incorrect grouped split was replaced.")

print("\nNext step:")
print("Run the leakage checker against split_grouped.")

print("=" * 70)