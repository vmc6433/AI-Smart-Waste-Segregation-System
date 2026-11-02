from pathlib import Path
import csv
from collections import Counter, defaultdict

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
    / "same_class_near_duplicate_analysis.txt"
)


def normalize_class(name):
    return name.strip().upper()


rows = []

with REPORT.open("r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:

        try:
            distance = int(row["phash_distance"])
        except (ValueError, KeyError):
            continue

        class_a = normalize_class(row["class_a"])
        class_b = normalize_class(row["class_b"])

        if class_a == class_b:
            rows.append(row)


print("=" * 70)
print("SAME-CLASS NEAR-DUPLICATE ANALYSIS")
print("=" * 70)

print()
print(f"Same-class pairs: {len(rows)}")


# ---------------------------------------------------------
# Overall distance distribution
# ---------------------------------------------------------

distance_counts = Counter(
    int(row["phash_distance"])
    for row in rows
)

print()
print("DISTANCE DISTRIBUTION")
print("-" * 70)

for distance in sorted(distance_counts):
    print(
        f"Distance {distance:<2}: "
        f"{distance_counts[distance]:6} pairs"
    )


# ---------------------------------------------------------
# Class distribution
# ---------------------------------------------------------

class_counts = Counter()

for row in rows:
    class_a = normalize_class(row["class_a"])
    class_counts[class_a] += 1


print()
print("SAME-CLASS PAIRS BY CLASS")
print("-" * 70)

for cls, count in class_counts.most_common():
    print(f"{cls:<15}: {count:6} pairs")


# ---------------------------------------------------------
# Distance by class
# ---------------------------------------------------------

class_distance = defaultdict(Counter)

for row in rows:
    cls = normalize_class(row["class_a"])
    distance = int(row["phash_distance"])

    class_distance[cls][distance] += 1


print()
print("DISTANCE DISTRIBUTION BY CLASS")
print("-" * 70)

for cls in sorted(class_distance):

    print()
    print(cls)

    for distance in sorted(class_distance[cls]):
        count = class_distance[cls][distance]

        print(
            f"  Distance {distance}: "
            f"{count:6}"
        )


# ---------------------------------------------------------
# Very strong similarity
# ---------------------------------------------------------

print()
print("VERY STRONG SAME-CLASS SIMILARITY")
print("-" * 70)

for cls in sorted(class_distance):

    d0 = class_distance[cls].get(0, 0)
    d1 = class_distance[cls].get(1, 0)
    d2 = class_distance[cls].get(2, 0)

    total = d0 + d1 + d2

    print(
        f"{cls:<15}: "
        f"distance 0-2 = {total:6} pairs"
    )


# ---------------------------------------------------------
# Save report
# ---------------------------------------------------------

with OUTPUT.open("w", encoding="utf-8") as f:

    f.write("SAME-CLASS NEAR-DUPLICATE ANALYSIS\n")
    f.write("=" * 70 + "\n\n")

    f.write(
        f"Same-class pairs: {len(rows)}\n\n"
    )

    f.write("DISTANCE DISTRIBUTION\n")
    f.write("-" * 70 + "\n")

    for distance in sorted(distance_counts):

        f.write(
            f"Distance {distance}: "
            f"{distance_counts[distance]} pairs\n"
        )

    f.write("\n\nSAME-CLASS PAIRS BY CLASS\n")
    f.write("-" * 70 + "\n")

    for cls, count in class_counts.most_common():

        f.write(
            f"{cls}: {count} pairs\n"
        )

    f.write("\n\nDISTANCE DISTRIBUTION BY CLASS\n")
    f.write("-" * 70 + "\n")

    for cls in sorted(class_distance):

        f.write(f"\n{cls}\n")

        for distance in sorted(class_distance[cls]):

            f.write(
                f"  Distance {distance}: "
                f"{class_distance[cls][distance]}\n"
            )

    f.write("\n\nVERY STRONG SAME-CLASS SIMILARITY\n")
    f.write("-" * 70 + "\n")

    for cls in sorted(class_distance):

        d0 = class_distance[cls].get(0, 0)
        d1 = class_distance[cls].get(1, 0)
        d2 = class_distance[cls].get(2, 0)

        f.write(
            f"{cls}: distance 0-2 = "
            f"{d0 + d1 + d2} pairs\n"
        )


print()
print("=" * 70)
print("REPORT SAVED")
print("=" * 70)

print(OUTPUT)

print()
print("=" * 70)
print("IMPORTANT")
print("=" * 70)

print()
print("No images were deleted, moved, or modified.")