"""MCQT 回答回収スクリプト（採点版）"""
import os
import json
import argparse
import pandas as pd
from pathlib import Path
from typing import List, Tuple

LETTERS = ["A", "B", "C", "D"]

def parse_levels(levels_arg: str) -> List[int]:
    """Parse '0-7' or '0,2,4,6' into a list of ints."""
    levels_arg = str(levels_arg).strip()
    if "-" in levels_arg:
        a, b = levels_arg.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in levels_arg.split(",") if x.strip() != ""]

def get_correct_letter(pool_json_path: str) -> str:
    """Read a pool JSON and return the correct letter (A-D) from 0-based answer_index."""
    with open(pool_json_path, encoding="utf-8") as f:
        pool = json.load(f)
    idx = int(pool["answer_index"])
    if idx < 0 or idx >= len(LETTERS):
        raise ValueError(f"answer_index out of range in {pool_json_path}: {idx}")
    return LETTERS[idx]

def score_one_level(model: str, level: int, pool_dir: str, out_root: str) -> Tuple[pd.DataFrame, float]:
    """
    Score predictions for one level.
    Inputs:
      - predictions: outputs/mcqa_results/<model>/L<level>.csv
      - pool_dir: directory with <function>.json
    Outputs:
      - outputs/mcqa_scored/<model>/L<level>_scored.csv
      - outputs/mcqa_scored/<model>/L<level>_confusion.csv
    Returns:
      (scored_df, accuracy_float)
    """
    in_csv = os.path.join("outputs", "mcqt_results", model.replace(":", "-"), f"L{level}.csv")
    if not os.path.exists(in_csv):
        raise FileNotFoundError(f"回答CSVが見つかりません: {in_csv}")

    df = pd.read_csv(in_csv)
    # Drop the trailing summary row added by run_mcqa.py (e.g., "TOTAL_ELAPSED_TIME(sec),...")
    if "function" in df.columns:
        df = df[df["function"].notna()]
        df = df[df["function"] != "TOTAL_ELAPSED_TIME(sec)"]

    rows = []
    for _, row in df.iterrows():
        func = str(row.get("function", "")).strip()
        pred = str(row.get("answer", "")).strip().upper()
        # normalize to A-D or '?'
        pred = pred if pred in LETTERS else "?"

        pool_json = os.path.join(pool_dir, f"{func}.json")
        if not os.path.exists(pool_json):
            rows.append({
                "function": func,
                "predicted": pred,
                "correct": None,
                "is_correct": None,
                "note": f"POOL NOT FOUND: {pool_json}"
            })
            continue

        try:
            correct = get_correct_letter(pool_json)
        except Exception as e:
            rows.append({
                "function": func,
                "predicted": pred,
                "correct": None,
                "is_correct": None,
                "note": f"POOL PARSE ERROR: {e}"
            })
            continue

        is_correct = (pred == correct)
        rows.append({
            "function": func,
            "predicted": pred,
            "correct": correct,
            "is_correct": is_correct,
            "note": ""
        })

    scored = pd.DataFrame(rows)

    # accuracy on rows with is_correct not null
    if "is_correct" in scored and scored["is_correct"].notna().any():
        acc = float(scored["is_correct"].dropna().mean())
    else:
        acc = 0.0

    # Output
    out_dir = os.path.join(out_root, model.replace(":", "-"))
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    out_scored = os.path.join(out_dir, f"L{level}_scored.csv")
    scored.to_csv(out_scored, index=False, encoding="utf-8")

    # Confusion matrix (correct x predicted), only on non-null rows
    cm = (
        scored.dropna(subset=["predicted", "correct"])
              .groupby(["correct", "predicted"])
              .size()
              .unstack(fill_value=0)
              .reindex(index=LETTERS, columns=LETTERS + ["?"], fill_value=0)  # include '?' column
    )
    out_cm = os.path.join(out_dir, f"L{level}_confusion.csv")
    cm.to_csv(out_cm, encoding="utf-8")

    print(f"[L{level}] 正答率: {acc:.2%}  | 採点: {out_scored}  | 混同行列: {out_cm}")
    return scored, acc

def main():
    ap = argparse.ArgumentParser(description="MCQA結果を採点（モデル名・複数レベル対応版）")
    ap.add_argument("--model", type=str, required=True, help="採点対象のモデル名（例: gpt-4o）")
    ap.add_argument("--levels", type=str, default="0-7", help="レベル指定（例: 0-7 または 0,2,4,6）")
    ap.add_argument("--pool_dir", type=str, default="data/mcqa_pools_refined_v4", help="MCQAプールJSONのディレクトリ")
    ap.add_argument("--out_root", type=str, default="outputs/mcqa_scored", help="採点結果の出力先ルート")
    args = ap.parse_args()

    levels = parse_levels(args.levels)

    summary_rows = []
    print(f"=== 採点開始 | model={args.model} | levels={levels} | pool_dir={args.pool_dir} ===")
    for lv in levels:
        try:
            scored, acc = score_one_level(args.model, lv, args.pool_dir, args.out_root)
            total = int(scored["function"].notna().sum())
            ok = int(scored["is_correct"].fillna(False).sum())
            ng = int((scored["is_correct"] == False).sum())
            missing = int(scored["is_correct"].isna().sum())
        except FileNotFoundError as e:
            print(f"[L{lv}] ERROR: {e}")
            acc, total, ok, ng, missing = 0.0, 0, 0, 0, 0

        summary_rows.append({
            "level": lv,
            "model": args.model,
            "accuracy": acc,
            "num_total": total,
            "num_correct": ok,
            "num_incorrect": ng,
            "num_missing": missing,
            "pool_dir": args.pool_dir
        })

    # Summary CSV
    summary_df = pd.DataFrame(summary_rows)
    out_summary_dir = os.path.join(args.out_root, args.model.replace(":", "-"))
    Path(out_summary_dir).mkdir(parents=True, exist_ok=True)
    out_summary_csv = os.path.join(out_summary_dir, "summary.csv")
    summary_df.to_csv(out_summary_csv, index=False, encoding="utf-8")

    print("\n=== 採点サマリ ===")
    if not summary_df.empty:
        print(summary_df.to_string(index=False))
    print(f"\nサマリ保存: {out_summary_csv}")

if __name__ == "__main__":
    main()
