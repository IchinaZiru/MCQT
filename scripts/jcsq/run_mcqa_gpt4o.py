from pathlib import Path
import json
import csv
import time

from dotenv import load_dotenv
from openai import OpenAI

ROOT_DIR = Path(__file__).resolve().parents[2]
VARIANTS_JSONL = ROOT_DIR / "data" / "json" / "jcsq_mcqa_variants.jsonl"
OUT_CSV        = ROOT_DIR / "outputs" / "mcqa_results" / "gpt4o_jcsq_mcqa.csv"

load_dotenv()
client = OpenAI()


def load_jsonl(path: Path):
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def build_prompt(question: str, choices: list[str]) -> str:
    letters = ["A", "B", "C", "D"]
    lines = []
    lines.append(f"次の四択問題に答えてください。")
    lines.append("")
    lines.append(f"問題: {question}")
    lines.append("")
    for i, choice in enumerate(choices):
        lines.append(f"{letters[i]}. {choice}")
    lines.append("")
    lines.append("最も適切な選択肢のアルファベット1文字（A, B, C, D）のみを出力してください。理由や解説は書かないでください。")
    return "\n".join(lines)


def parse_choice_letter(text: str) -> tuple[int | None, str]:
    """
    モデル出力から A/B/C/D を1文字抜き出して index に変換する。
    うまく取れなければ None を返す。
    """
    if not text:
        return None, ""

    raw = text.strip()
    # 最初に現れる A/B/C/D/a/b/c/d を採用
    for ch in raw:
        if ch.upper() in ["A", "B", "C", "D"]:
            idx = ["A", "B", "C", "D"].index(ch.upper())
            return idx, ch.upper()
    return None, raw


def main():
    instances = load_jsonl(VARIANTS_JSONL)
    print(f"loaded {len(instances)} MCQA instances from {VARIANTS_JSONL}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow([
            "qid",
            "variant",
            "question",
            "choices_json",
            "answer_index",
            "pred_index",
            "pred_letter",
            "raw_output",
            "correct",
        ])

        for i, inst in enumerate(instances):
            qid     = inst["qid"]
            variant = inst["variant"]
            question = inst["question"]
            choices  = inst["choices"]
            answer_index = inst["answer_index"]

            prompt = build_prompt(question, choices)

            try:
                res = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": "あなたは日本語の常識問題を解くアシスタントです。"
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.0,
                    max_tokens=4,
                )
                content = res.choices[0].message.content or ""
            except Exception as e:
                print(f"[{i+1}/{len(instances)}] qid={qid} ERROR: {e}")
                content = ""

            pred_index, pred_letter = parse_choice_letter(content)
            correct = (pred_index == answer_index)

            writer.writerow([
                qid,
                variant,
                question,
                json.dumps(choices, ensure_ascii=False),
                answer_index,
                pred_index if pred_index is not None else "",
                pred_letter,
                content,
                int(correct) if pred_index is not None else "",
            ])

            if (i + 1) % 100 == 0:
                print(f"processed {i+1}/{len(instances)}")

            # レート制限ケア（必要に応じて調整）
            time.sleep(0.05)

    print(f"saved results to {OUT_CSV}")


if __name__ == "__main__":
    main()
