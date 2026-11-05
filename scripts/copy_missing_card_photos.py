"""Fill missing card photo files with the design/photos/COPY.jpg placeholder."""

import csv
import shutil
import unicodedata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CARDS_CSV = ROOT / "cards.csv"
PHOTOS_DIR = ROOT / "design" / "photos"
PLACEHOLDER = PHOTOS_DIR / "COPY.jpg"


def image_name(title: str) -> str:
    """Match the expected photo filename convention in add_card_images.py."""
    name = unicodedata.normalize("NFKD", title)
    name = name.encode("ascii", "ignore").decode("ascii").lower()
    return name.replace(" ", "_")


def main() -> None:
    if not PLACEHOLDER.is_file():
        raise FileNotFoundError(f"Placeholder photo not found: {PLACEHOLDER}")

    with CARDS_CSV.open("r", encoding="utf-8-sig", newline="") as cards_file:
        reader = csv.DictReader(cards_file)
        if reader.fieldnames is None or "Title" not in reader.fieldnames:
            raise ValueError(f"{CARDS_CSV} must have a 'Title' column")
        titles = [row["Title"] for row in reader]

    copied = 0
    skipped = 0
    for title in titles:
        destination = PHOTOS_DIR / f"{image_name(title)}.jpg"
        if destination.exists():
            skipped += 1
            continue
        shutil.copy2(PLACEHOLDER, destination)
        copied += 1
        print(f"Copied placeholder for {title}: {destination.name}")

    print(f"Done: copied {copied} missing photo(s); skipped {skipped} existing photo(s).")


if __name__ == "__main__":
    main()
