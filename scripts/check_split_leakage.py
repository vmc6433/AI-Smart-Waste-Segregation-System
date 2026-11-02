from pathlib import Path
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SPLIT_DIR = PROJECT_ROOT / "datasets" / "split_grouped"

NEAR_DUPLICATE_REPORT = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "duplicate_reports"
    / "near_duplicates.csv"
)

OUTPUT_REPORT = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "duplicate_reports"
    / "split_leakage_report.csv"
)

# ============================================================
# FUNCTIONS
# ============================================================

def get_split(filename):
    """
    Determine whether an image belongs to train, val, or test.
    """

    filename = Path(filename).name

    for split in ["train", "val", "test"]:

        matches = list(SPLIT_DIR.glob(f"{split}/*/{filename}"))

        if matches:
            return split

    return "NOT_FOUND"


# ============================================================
# LOAD REPORT
# ============================================================

print("=" * 70)
print("CHECKING TRAIN / VALIDATION / TEST LEAKAGE")
print("=" * 70)

if not NEAR_DUPLICATE_REPORT.exists():
    print("\nERROR:")
    print("Near-duplicate report was not found:")
    print(NEAR_DUPLICATE_REPORT)
    raise SystemExit(1)

df = pd.read_csv(NEAR_DUPLICATE_REPORT)

print("\nNear-duplicate pairs loaded:", len(df))

print("\nColumns found:")
print(list(df.columns))

# ============================================================
# PROCESS PAIRS
# ============================================================

results = []

cross_split_count = 0
same_split_count = 0
not_found_count = 0

for _, row in df.iterrows():

    file_a = row["file_a"]
    file_b = row["file_b"]

    class_a = row["class_a"]
    class_b = row["class_b"]

    distance = row["phash_distance"]

    split_a = get_split(file_a)
    split_b = get_split(file_b)

    if split_a == "NOT_FOUND" or split_b == "NOT_FOUND":
        not_found_count += 1
        continue

    if split_a != split_b:

        cross_split_count += 1

        results.append({
            "phash_distance": distance,
            "class_a": class_a,
            "file_a": file_a,
            "split_a": split_a,
            "class_b": class_b,
            "file_b": file_b,
            "split_b": split_b,
        })

    else:
        same_split_count += 1

# ============================================================
# SAVE REPORT
# ============================================================

leakage_df = pd.DataFrame(results)

if len(leakage_df) > 0:
    leakage_df = leakage_df.sort_values(
        by="phash_distance"
    )

    leakage_df.to_csv(
        OUTPUT_REPORT,
        index=False
    )

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("LEAKAGE CHECK SUMMARY")
print("=" * 70)

print(f"Total near-duplicate pairs : {len(df)}")
print(f"Same split                 : {same_split_count}")
print(f"Cross-split pairs          : {cross_split_count}")
print(f"Not found                  : {not_found_count}")

print("\n" + "-" * 70)

if cross_split_count == 0:

    print("RESULT: NO CROSS-SPLIT NEAR-DUPLICATE PAIRS FOUND.")
    print("The current split looks clean according to this report.")

else:

    print("RESULT: CROSS-SPLIT NEAR-DUPLICATES FOUND.")
    print("These should be reviewed before training.")

print("-" * 70)

if cross_split_count > 0:

    print("\nCross-split pairs by split combination:")

    combinations = (
        leakage_df
        .apply(
            lambda row:
            " + ".join(
                sorted(
                    [row["split_a"], row["split_b"]]
                )
            ),
            axis=1
        )
        .value_counts()
    )

    for combination, count in combinations.items():
        print(f"{combination:20s}: {count}")

    print("\nCross-split pairs by pHash distance:")

    distance_counts = leakage_df["phash_distance"].value_counts().sort_index()

    for distance, count in distance_counts.items():
        print(f"Distance {distance}: {count}")

print("\nReport:")

if cross_split_count > 0:
    print(OUTPUT_REPORT)
else:
    print("No leakage report was necessary.")

print("\n" + "=" * 70)