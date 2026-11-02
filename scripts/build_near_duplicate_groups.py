from pathlib import Path
import csv
from collections import defaultdict


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORT = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "duplicate_reports"
    / "near_duplicates.csv"
)

OUTPUT = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "duplicate_reports"
    / "near_duplicate_groups.csv"
)

# We only group very strong similarities.
# Distance 0-2 = strong similarity.
MAX_DISTANCE = 2


# ---------------------------------------------------------
# Union-Find / Disjoint Set
# ---------------------------------------------------------

class UnionFind:

    def __init__(self):
        self.parent = {}
        self.rank = {}

    def add(self, item):
        if item not in self.parent:
            self.parent[item] = item
            self.rank[item] = 0

    def find(self, item):
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])

        return self.parent[item]

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


# ---------------------------------------------------------
# Read near-duplicate report
# ---------------------------------------------------------

uf = UnionFind()

selected_pairs = 0

with REPORT.open(
    "r",
    encoding="utf-8-sig",
    newline=""
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        try:
            distance = int(row["phash_distance"])
        except (ValueError, KeyError):
            continue

        if distance > MAX_DISTANCE:
            continue

        file_a = row["file_a"]
        file_b = row["file_b"]

        uf.union(file_a, file_b)

        selected_pairs += 1


# ---------------------------------------------------------
# Build groups
# ---------------------------------------------------------

groups = defaultdict(list)

for file_path in uf.parent:

    root = uf.find(file_path)
    groups[root].append(file_path)


# Sort groups by size
group_list = sorted(
    groups.values(),
    key=lambda x: (-len(x), x[0])
)


# ---------------------------------------------------------
# Save CSV
# ---------------------------------------------------------

with OUTPUT.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "group_id",
        "group_size",
        "class",
        "file_path"
    ])

    group_id = 1

    for group in group_list:

        group.sort()

        # Determine classes from path
        classes = set()

        for file_path in group:

            parts = Path(file_path).parts

            if "master" in parts:

                index = parts.index("master")

                if index + 1 < len(parts):
                    classes.add(parts[index + 1])

        class_name = "|".join(sorted(classes))

        for file_path in group:

            writer.writerow([
                group_id,
                len(group),
                class_name,
                file_path
            ])

        group_id += 1


# ---------------------------------------------------------
# Statistics
# ---------------------------------------------------------

group_sizes = [
    len(group)
    for group in group_list
]

singletons = sum(
    1 for size in group_sizes
    if size == 1
)

multi_image_groups = sum(
    1 for size in group_sizes
    if size > 1
)

largest_group = max(group_sizes) if group_sizes else 0

images_in_groups = sum(
    size for size in group_sizes
    if size > 1
)


print("=" * 70)
print("NEAR-DUPLICATE GROUP ANALYSIS")
print("=" * 70)

print()
print(f"Similarity threshold : pHash distance <= {MAX_DISTANCE}")
print(f"Pairs used            : {selected_pairs}")
print(f"Images in groups      : {len(uf.parent)}")
print(f"Total groups          : {len(group_list)}")
print(f"Singleton groups      : {singletons}")
print(f"Multi-image groups    : {multi_image_groups}")
print(f"Images in multi-groups: {images_in_groups}")
print(f"Largest group         : {largest_group}")

print()
print("LARGEST GROUPS")
print("-" * 70)

for i, group in enumerate(group_list[:20], start=1):

    if len(group) <= 1:
        break

    print(
        f"Group {i:4}: "
        f"{len(group):3} images"
    )

    for file_path in group[:5]:
        print(f"    {file_path}")

    if len(group) > 5:
        print(
            f"    ... and "
            f"{len(group) - 5} more"
        )

print()
print("=" * 70)
print("REPORT SAVED")
print("=" * 70)

print(OUTPUT)

print()
print("IMPORTANT")
print("-" * 70)
print("This is analysis only.")
print("NO images were deleted, moved, or modified.")