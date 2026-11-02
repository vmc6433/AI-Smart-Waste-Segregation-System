from pathlib import Path
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ZIP_PATH = PROJECT_ROOT / "datasets" / "raw" / "ewaste" / "Laptop Components Image Dataset to Classify Different Components.zip"

print("=" * 70)
print("E-WASTE DATASET STRUCTURE INSPECTION")
print("=" * 70)

with zipfile.ZipFile(ZIP_PATH, "r") as z:
    files = z.namelist()

print(f"\nTotal files: {len(files)}")

print("\nFirst 100 paths inside ZIP:")
print("-" * 70)

for path in files[:100]:
    print(path)

print("\n" + "=" * 70)
print("FOLDERS CONTAINING 'RAW', 'AUGMENT', OR 'DATA'")
print("=" * 70)

folders = set()

for path in files:
    parts = Path(path).parts

    for part in parts:
        lower = part.lower()

        if (
            "raw" in lower
            or "augment" in lower
            or lower == "data"
        ):
            folders.add(part)

for folder in sorted(folders):
    print(folder)

print("\n" + "=" * 70)
print("TOP-LEVEL / SECOND-LEVEL STRUCTURE")
print("=" * 70)

structures = set()

for path in files:
    parts = Path(path).parts

    if len(parts) >= 2:
        structures.add("/".join(parts[:2]))

for structure in sorted(structures):
    print(structure)

print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)