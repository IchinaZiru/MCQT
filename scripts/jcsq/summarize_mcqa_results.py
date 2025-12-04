from pathlib import Path
import csv
from collections import defaultdict

ROOT_DIR = Path(__file__).resolve().parents[2]
RESULT_CSV = ROOT_DIR / "outputs" / "mcqa_results" / "gpt4o_jcsq_mcqa.csv"


def main():
    counts = defaultdict(int)
    corrects = defaultdict(int)

    with RESULT_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            variant = row["variant"]
            if row["correct"] == "":
                continue
            counts[variant] += 1
            if row["correct"] == "1":
                corrects[variant] += 1

    for variant in sorted(counts.keys()):
        acc = corrects[variant] / counts[variant]
        print(f"{variant:8s}: {corrects[variant]}/{counts[variant]} = {acc:.4f}")


if __name__ == "__main__":
    main()
