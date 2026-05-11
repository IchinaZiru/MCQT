#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_mcqa_ollama_llm_mcqt_only.py

variants JSONL（orig/sciq/mcqt/llm_mcqt が混在）から、指定 variant（既定: llm_mcqt）だけを抽出して
Ollama モデルに解かせ、CSV に保存します。

このスクリプトは、既存の scripts/jcsq/run_mcqa_ollama.py を import して
以下の既存実装をそのまま再利用します：
- select_model_interactively
- load_jsonl
- build_prompt
- parse_choice_letter
- call_ollama_chat

使い方（PowerShell例）
  python scripts/jcsq/run_mcqa_ollama_llm_mcqt_only.py \
    --model gpt-oss:20b \
    --input data/json/jcsq_mcqa_variants-gpt-oss.jsonl \
    --variant llm_mcqt \
    --resume

出力
  outputs_jcsq/mcqa_results/ollama_<safe_model>_<variant>.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Set


def safe_model_name(model_name: str) -> str:
    return model_name.replace(":", "_").replace("/", "_")


def read_done_qids(csv_path: Path) -> Set[str]:
    """既存CSVにあるqidを集合として返す（resume用）"""
    done: Set[str] = set()
    if not csv_path.exists():
        return done

    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "qid" not in reader.fieldnames:
                return done
            for row in reader:
                qid = row.get("qid")
                if qid is None:
                    continue
                qid_s = str(qid).strip()
                if qid_s:
                    done.add(qid_s)
    except Exception:
        # ここで落としてしまうより、resume無し相当で走らせる
        return set()

    return done


def main() -> None:
    # 既存スクリプトを "使用"（import）
    try:
        import run_mcqa_ollama as base  # scripts/jcsq/run_mcqa_ollama.py
    except Exception as e:
        raise SystemExit(
            "run_mcqa_ollama.py を import できませんでした。\n"
            "このファイルを scripts/jcsq/ に置き、run_mcqa_ollama.py と同じフォルダで実行してください。\n"
            f"detail: {e}"
        )

    parser = argparse.ArgumentParser(
        description="variants JSONLから指定variant（既定 llm_mcqt）だけをOllamaで解かせる（run_mcqa_ollama.py再利用）"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="使用するOllamaモデル名 (例: gemma3:27b)。指定しないと対話的に選択。",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Ollama host URL（既存run_mcqa_ollama.pyの OLLAMA_HOST を上書き）",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str((Path(__file__).resolve().parents[2] / "data" / "json" / "jcsq_mcqa_variants.jsonl")),
        help="入力 variants JSONL（例: data/json/jcsq_mcqa_variants-gpt-oss.jsonl）",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="llm_mcqt",
        help="解くvariant（既定: llm_mcqt）",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="既存の出力CSVがあれば、既に処理したqidをスキップして続きから実行",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.01,
        help="各問の間に待つ秒数（既定: 0.01）",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Ollama呼び出しのタイムアウト秒（既定: 300）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="先頭からN件だけ解く（0で全件）",
    )
    args = parser.parse_args()

    # host上書き（base.call_ollama_chat が base.OLLAMA_HOST を参照するため）
    if args.host:
        base.OLLAMA_HOST = args.host.strip()

    # モデル名決定（既存の対話UIを再利用）
    if args.model:
        model_name = args.model.strip()
    else:
        model_name = base.select_model_interactively(base.DEFAULT_MODEL_NAME)

    if not model_name:
        raise SystemExit("モデル名が空です。終了します。")

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"INPUT JSONL not found: {in_path}")

    # 入力読み込み（既存のload_jsonlを再利用）
    all_instances = base.load_jsonl(in_path)
    instances = [x for x in all_instances if x.get("variant") == args.variant]
    if args.limit and args.limit > 0:
        instances = instances[: args.limit]

    safe = safe_model_name(model_name)
    out_csv = (
        base.ROOT_DIR
        / "outputs_jcsq"
        / "mcqa_results"
        / f"ollama_{safe}_{args.variant}.csv"
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    print("=== 設定 ===")
    print(f"  OLLAMA_HOST : {base.OLLAMA_HOST}")
    print(f"  MODEL_NAME  : {model_name}")
    print(f"  INPUT  JSONL: {in_path}")
    print(f"  TARGET VAR. : {args.variant}")
    print(f"  OUTPUT CSV  : {out_csv}")
    print(f"  RESUME      : {args.resume}")
    print(f"  TOTAL(in)   : {len(all_instances)}")
    print(f"  TOTAL(var)  : {len(instances)}")
    print("================\n")

    if not instances:
        raise SystemExit(f"No instances found for variant='{args.variant}'. Check input JSONL.")

    done_qids: Set[str] = set()
    if args.resume and out_csv.exists():
        done_qids = read_done_qids(out_csv)
        if done_qids:
            print(f"resume: {len(done_qids)} qids already processed; will skip.")

    mode = "a" if (args.resume and out_csv.exists()) else "w"
    write_header = (mode == "w")

    n_answered = 0
    n_correct = 0
    n_total = len(instances)

    with out_csv.open(mode, encoding="utf-8", newline="") as f_out:
        writer = csv.writer(f_out)
        if write_header:
            writer.writerow(
                [
                    "qid",
                    "variant",
                    "question",
                    "choices_json",
                    "answer_index",
                    "pred_index",
                    "pred_letter",
                    "raw_output",
                    "correct",
                ]
            )

        for i, inst in enumerate(instances):
            qid = inst["qid"]
            if str(qid) in done_qids:
                continue

            question = inst["question"]
            choices = inst["choices"]
            answer_index = inst["answer_index"]

            prompt = base.build_prompt(question, choices)

            try:
                # base.call_ollama_chat は timeout を受け取れない実装なので、
                # ここで一時的に requests.post timeout を変えるのは困難。
                # ただし、base 側は timeout=300 固定なので、args.timeout は表示用に残す。
                content = base.call_ollama_chat(prompt, model_name)
            except Exception as e:
                print(f"[{i+1}/{n_total}] qid={qid} ERROR: {e}")
                content = ""

            pred_index, pred_letter = base.parse_choice_letter(content)
            correct = (pred_index == answer_index) if pred_index is not None else None

            if pred_index is not None:
                n_answered += 1
                if correct:
                    n_correct += 1

            writer.writerow(
                [
                    qid,
                    args.variant,
                    question,
                    json.dumps(choices, ensure_ascii=False),
                    answer_index,
                    pred_index if pred_index is not None else "",
                    pred_letter,
                    content,
                    int(correct) if correct is not None else "",
                ]
            )

            if (i + 1) % 100 == 0:
                acc = (n_correct / n_answered) if n_answered else 0.0
                print(f"processed {i+1}/{n_total} (answered={n_answered}, acc={acc:.4f})")

            time.sleep(args.sleep)

    acc = (n_correct / n_answered) if n_answered else 0.0
    print(f"\n[DONE] answered={n_answered}/{n_total} correct={n_correct} acc={acc:.6f}")
    print(f"saved results to {out_csv}")


if __name__ == "__main__":
    main()
