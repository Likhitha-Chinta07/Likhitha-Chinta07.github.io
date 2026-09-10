"""Download the public Pima Indians Diabetes CSV used by this project."""

from pathlib import Path
from urllib.request import urlopen

DATA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
OUTPUT = Path(__file__).parent / "data" / "diabetes.csv"
COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]


def download_dataset() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(DATA_URL, timeout=30) as response:
        raw = response.read().decode("utf-8")

    rows = [line.strip() for line in raw.splitlines() if line.strip()]
    expected_width = len(COLUMNS)
    if not rows or any(len(row.split(",")) != expected_width for row in rows):
        raise ValueError("Downloaded dataset does not match the expected 9-column format.")

    OUTPUT.write_text(",".join(COLUMNS) + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    print(f"Saved {len(rows):,} records to {OUTPUT}")


if __name__ == "__main__":
    download_dataset()
