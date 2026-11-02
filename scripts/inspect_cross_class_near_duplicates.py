from pathlib import Path
import csv
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORT = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "duplicate_reports"
    / "near_duplicates.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "duplicate_reports"
    / "near_duplicate_cross_class_inspection"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_DISTANCE = 2


def normalize_class(name):
    return name.strip().upper()


rows = []

with REPORT.open("r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        try:
            distance = int(row["distance"])
        except (ValueError, KeyError):
            continue

        if distance > MAX_DISTANCE:
            continue

        class1 = normalize_class(row["class1"])
        class2 = normalize_class(row["class2"])

        if class1 == class2:
            continue

        rows.append(row)


print("=" * 70)
print("CROSS-CLASS NEAR-DUPLICATE INSPECTION")
print("=" * 70)

print()
print(f"Distance threshold : <= {MAX_DISTANCE}")
print(f"Cross-class pairs  : {len(rows)}")
print()

if not rows:
    print("No cross-class near-duplicates found.")
    raise SystemExit


# Sort strongest matches first
rows.sort(key=lambda x: int(x["distance"]))


print("CLASS PAIRS")
print("-" * 70)

pair_counts = {}

for row in rows:
    c1 = normalize_class(row["class1"])
    c2 = normalize_class(row["class2"])

    pair = " <-> ".join(sorted([c1, c2]))
    pair_counts[pair] = pair_counts.get(pair, 0) + 1

for pair, count in sorted(pair_counts.items(), key=lambda x: -x[1]):
    print(f"{pair:35} : {count:4}")


print()
print("COPYING INSPECTION IMAGES")
print("-" * 70)


copied = 0
missing = 0

for index, row in enumerate(rows, start=1):

    file1 = Path(row["file1"])
    file2 = Path(row["file2"])

    distance = row["distance"]
    class1 = normalize_class(row["class1"])
    class2 = normalize_class(row["class2"])

    pair_folder = f"{index:04d}_D{distance}_{class1}_vs_{class2}"
    pair_dir = OUTPUT_DIR / pair_folder

    pair_dir.mkdir(parents=True, exist_ok=True)

    for source, label in [
        (file1, f"{class1}__A"),
        (file2, f"{class2}__B"),
    ]:

        if not source.exists():
            missing += 1
            continue

        destination = pair_dir / f"{label}{source.suffix.lower()}"

        shutil.copy2(source, destination)
        copied += 1

    # Save metadata
    metadata = pair_dir / "metadata.txt"

    metadata.write_text(
        f"Distance: {distance}\n"
        f"Class A: {class1}\n"
        f"Class B: {class2}\n"
        f"Image A: {file1}\n"
        f"Image B: {file2}\n",
        encoding="utf-8"
    )


print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"Pairs selected       : {len(rows)}")
print(f"Images copied        : {copied}")
print(f"Missing source files : {missing}")

print()
print("Inspection folder:")
print(OUTPUT_DIR)

print()
print("=" * 70)
print("IMPORTANT")
print("=" * 70)
print()
print("Analysis/inspection only.")
print("NO images were deleted or modified.")