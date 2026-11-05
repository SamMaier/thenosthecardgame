"""Add the expected photo path for each card to design/real_card_data.csv."""

import csv
import unicodedata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "cards.csv"
DESTINATION = ROOT / "design" / "real_card_data.csv"


def main() -> None:
    with SOURCE.open("r", encoding="utf-8-sig", newline="") as source_file:
        reader = csv.DictReader(source_file)
        if reader.fieldnames is None or "Title" not in reader.fieldnames:
            raise ValueError(f"{SOURCE} must have a 'Title' column")

        fieldnames = [name for name in reader.fieldnames if name != "Image"] + ["Image"]
        rows = list(reader)

    for row in rows:
        image_name = unicodedata.normalize("NFKD", row["Title"])
        image_name = image_name.encode("ascii", "ignore").decode("ascii").lower()
        image_name = image_name.replace(" ", "_")
        row["Image"] = f"photos/{image_name}.jpg"
        row["Tags"] = "~".join(tag.strip() for tag in row["Tags"].split(","))

    with DESTINATION.open("w", encoding="utf-8", newline="") as destination_file:
        writer = csv.DictWriter(destination_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
