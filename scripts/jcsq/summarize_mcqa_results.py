from pathlib import Path
import csv
from collections import defaultdict
import argparse

# プロジェクトのルート想定（必要ならここだけ調整）
ROOT_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT_DIR / "outputs_jcsq" / "mcqa_results"


def choose_result_csv_interactively() -> Path:
    """
    outputs_jcsq/mcqa_results 以下の CSV を列挙して、
    ユーザーに番号で選んでもらう。
    """
    if not RESULTS_DIR.exists():
        raise SystemExit(f"結果ディレクトリが存在しません: {RESULTS_DIR}")

    csv_files = sorted(RESULTS_DIR.glob("*.csv"))
    if not csv_files:
        raise SystemExit(f"結果CSVが1つも見つかりませんでした: {RESULTS_DIR}")

    print("\n=== 集計対象とする結果CSVを選んでください ===")
    for i, p in enumerate(csv_files, start=1):
        size = p.stat().st_size
        print(f"{i:2d}: {p.name}   ({size} bytes)")

    while True:
        s = input(f"\n番号を選んでください (1-{len(csv_files)}): ").strip()
        if not s.isdigit():
            print("数字を入力してください。")
            continue
        idx = int(s)
        if 1 <= idx <= len(csv_files):
            return csv_files[idx - 1]
        else:
            print("番号が範囲外です。")


def resolve_result_csv(args) -> Path:
    """
    --csv が指定されていればそれを使う。
    なければ対話的に outputs_jcsq/mcqa_results から選ぶ。
    """
    if args.csv:
        p = Path(args.csv)
        if not p.is_absolute():
            p = ROOT_DIR / p
        return p
    else:
        return choose_result_csv_interactively()


def main():
    parser = argparse.ArgumentParser(
        description="MCQA結果CSVから variant ごとの全問題精度を集計するスクリプト"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="結果CSVファイルへのパス。指定しない場合は mcqa_results から対話的に選択。",
    )
    args = parser.parse_args()

    result_csv = resolve_result_csv(args)

    print("\nRESULT_CSV =", result_csv)
    if not result_csv.exists():
        raise SystemExit(f"結果CSVが存在しません: {result_csv}")
    print("size      =", result_csv.stat().st_size, "bytes\n")

    counts = defaultdict(int)
    corrects = defaultdict(int)

    with result_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            variant = row["variant"]
            if row["correct"] == "":
                continue
            counts[variant] += 1
            if row["correct"] == "1":
                corrects[variant] += 1

    print("=== Accuracy on ALL questions ===")
    for variant in sorted(counts.keys()):
        acc = corrects[variant] / counts[variant]
        print(f"{variant:8s}: {corrects[variant]}/{counts[variant]} = {acc:.4f}")


if __name__ == "__main__":
    main()
