from pathlib import Path
import csv
from collections import Counter


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORT_FILE = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "duplicate_reports"
    / "near_duplicates.csv"
)

SUMMARY_FILE = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "duplicate_reports"
    / "near_duplicate_analysis.txt"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("NEAR-DUPLICATE ANALYSIS")
    print("=" * 70)

    if not REPORT_FILE.exists():

        print()
        print("ERROR: Near-duplicate report not found.")
        print(REPORT_FILE)
        return

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    distance_counts = Counter()
    class_pair_counts = Counter()
    class_distance_counts = Counter()

    total_pairs = 0

    # --------------------------------------------------------
    # Read report
    # --------------------------------------------------------

    with open(
        REPORT_FILE,
        "r",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        reader = csv.DictReader(csv_file)

        for row in reader:

            distance = int(row["phash_distance"])

            class_a = row["class_a"]
            class_b = row["class_b"]

            # Keep class pair order consistent
            class_pair = tuple(
                sorted([class_a, class_b])
            )

            distance_counts[distance] += 1
            class_pair_counts[class_pair] += 1

            class_distance_counts[
                (class_pair, distance)
            ] += 1

            total_pairs += 1

    # --------------------------------------------------------
    # Distance ranges
    # --------------------------------------------------------

    distance_ranges = {
        "0": 0,
        "1-2": 0,
        "3-4": 0,
        "5-6": 0,
    }

    for distance, count in distance_counts.items():

        if distance == 0:
            distance_ranges["0"] += count

        elif distance <= 2:
            distance_ranges["1-2"] += count

        elif distance <= 4:
            distance_ranges["3-4"] += count

        elif distance <= 6:
            distance_ranges["5-6"] += count

    # --------------------------------------------------------
    # Cross-class counts
    # --------------------------------------------------------

    cross_class_pairs = 0

    same_class_pairs = 0

    for (class_a, class_b), count in class_pair_counts.items():

        if class_a == class_b:
            same_class_pairs += count
        else:
            cross_class_pairs += count

    # --------------------------------------------------------
    # Print distance distribution
    # --------------------------------------------------------

    print()
    print("DISTANCE DISTRIBUTION")
    print("-" * 70)

    for distance in sorted(distance_counts):

        print(
            f"Distance {distance:<2} : "
            f"{distance_counts[distance]:>6} pairs"
        )

    print()
    print("DISTANCE RANGES")
    print("-" * 70)

    for range_name, count in distance_ranges.items():

        print(
            f"{range_name:<8} : "
            f"{count:>6} pairs"
        )

    # --------------------------------------------------------
    # Print class pair distribution
    # --------------------------------------------------------

    print()
    print("CLASS PAIR DISTRIBUTION")
    print("-" * 70)

    sorted_class_pairs = sorted(
        class_pair_counts.items(),
        key=lambda x: (-x[1], x[0])
    )

    for (class_a, class_b), count in sorted_class_pairs:

        print(
            f"{class_a:<12} <-> "
            f"{class_b:<12} : "
            f"{count:>6} pairs"
        )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print()
    print(f"Total near-duplicate pairs : {total_pairs}")
    print(f"Same-class pairs           : {same_class_pairs}")
    print(f"Cross-class pairs          : {cross_class_pairs}")

    # --------------------------------------------------------
    # Write detailed text report
    # --------------------------------------------------------

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8"
    ) as report:

        report.write(
            "NEAR-DUPLICATE ANALYSIS\n"
        )

        report.write(
            "=" * 70 + "\n\n"
        )

        report.write(
            f"Total near-duplicate pairs: "
            f"{total_pairs}\n"
        )

        report.write(
            f"Same-class pairs: "
            f"{same_class_pairs}\n"
        )

        report.write(
            f"Cross-class pairs: "
            f"{cross_class_pairs}\n\n"
        )

        report.write(
            "DISTANCE DISTRIBUTION\n"
        )

        report.write(
            "-" * 70 + "\n"
        )

        for distance in sorted(distance_counts):

            report.write(
                f"Distance {distance}: "
                f"{distance_counts[distance]} pairs\n"
            )

        report.write(
            "\nDISTANCE RANGES\n"
        )

        report.write(
            "-" * 70 + "\n"
        )

        for range_name, count in distance_ranges.items():

            report.write(
                f"{range_name}: "
                f"{count} pairs\n"
            )

        report.write(
            "\nCLASS PAIR DISTRIBUTION\n"
        )

        report.write(
            "-" * 70 + "\n"
        )

        for (class_a, class_b), count in sorted_class_pairs:

            report.write(
                f"{class_a} <-> {class_b}: "
                f"{count} pairs\n"
            )

        report.write(
            "\n"
        )

        report.write(
            "NO FILES WERE MODIFIED.\n"
        )

    # --------------------------------------------------------
    # Final message
    # --------------------------------------------------------

    print()
    print("Analysis report saved to:")
    print(SUMMARY_FILE)

    print()
    print("=" * 70)
    print("IMPORTANT")
    print("=" * 70)
    print()
    print("This was an analysis-only operation.")
    print("No images were deleted, moved, or modified.")
    print()


if __name__ == "__main__":
    main()