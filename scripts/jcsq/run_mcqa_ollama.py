from pathlib import Path
import json
import csv
import time
import requests
import argparse

# ====== 基本設定（必要ならここだけ書き換え） ======
# Ollama のホスト
OLLAMA_HOST = "http://localhost:11434"

# デフォルトのモデル名（対話選択時の初期値）
DEFAULT_MODEL_NAME = "gpt-oss:20b"

# 最初は 500 問だけ評価したい場合は True にする
USE_SAMPLE_500 = False
# ================================================

ROOT_DIR = Path(__file__).resolve().parents[2]


def get_ollama_models():
    """
    Ollama の /api/tags から現在利用可能なモデル一覧を取得する。
    失敗した場合は空リストを返す。
    """
    url = f"{OLLAMA_HOST}/api/tags"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        models = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
        return models
    except Exception as e:
        print(f"[WARN] Ollamaのモデル一覧取得に失敗しました: {e}")
        return []


def select_model_interactively(default_model: str | None = None) -> str:
    """
    Ollama のモデル一覧を表示して、対話的にモデルを選択する。
    default_model が一覧にあれば「(default)」を表示し、Enter だけ押すとそれを使う。
    """
    models = get_ollama_models()

    if not models:
        print("Ollama からモデル一覧を取得できませんでした。")
        if default_model:
            print(f"既定モデル {default_model} を使用します。")
            return default_model
        # 最終手段：手入力
        return input("使用するモデル名を手入力してください (例: gemma3:27b): ").strip()

    print("\n=== Ollama に存在するモデル一覧 ===")
    for i, name in enumerate(models, start=1):
        mark = ""
        if default_model and name == default_model:
            mark = "  (default)"
        print(f"{i:2d}: {name}{mark}")

    while True:
        prompt_msg = (
            f"\n使用するモデル番号を選んでください (1-{len(models)}) "
            f"またはモデル名を直接入力（Enterで既定 {default_model} を使用）: "
        )
        s = input(prompt_msg).strip()

        # Enter → 既定モデル
        if not s:
            if default_model:
                return default_model
            else:
                print("モデル名を入力してください。")
                continue

        # 数字ならインデックス
        if s.isdigit():
            idx = int(s)
            if 1 <= idx <= len(models):
                return models[idx - 1]
            else:
                print("番号が範囲外です。")
                continue

        # それ以外はそのままモデル名として扱う
        return s


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
    lines.append("次の四択問題に答えてください。")
    lines.append("")
    lines.append(f"問題: {question}")
    lines.append("")
    for i, choice in enumerate(choices):
        lines.append(f"{letters[i]}. {choice}")
    lines.append("")
    lines.append(
        "最も適切な選択肢のアルファベット1文字（A, B, C, D）のみを出力してください。"
        "理由や解説は書かないでください。"
    )
    return "\n".join(lines)


def parse_choice_letter(text: str) -> tuple[int | None, str]:
    """
    モデル出力から A/B/C/D を1文字抜き出して index に変換する。
    うまく取れなければ None を返す。
    """
    if not text:
        return None, ""

    raw = text.strip()
    for ch in raw:
        if ch.upper() in ["A", "B", "C", "D"]:
            idx = ["A", "B", "C", "D"].index(ch.upper())
            return idx, ch.upper()
    return None, raw


def call_ollama_chat(prompt: str, model_name: str) -> str:
    """
    Ollama の /api/chat に投げて、返ってきた content を返す。
    """
    url = f"{OLLAMA_HOST}/api/chat"
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "あなたは日本語の常識問題を解くアシスタントです。"},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }
    resp = requests.post(url, json=payload, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    # Ollama のレスポンス: {"message": {"role": "...", "content": "..."}, "done": true, ...}
    return data["message"]["content"]


def main():
    parser = argparse.ArgumentParser(
        description="JCommonsenseQA MCQT を Ollama モデルで解かせるスクリプト"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="使用するOllamaモデル名 (例: gemma3:27b)。指定しないと対話的に選択。",
    )
    parser.add_argument(
        "--sample500",
        action="store_true",
        help="500問サンプル (jcsq_mcqa_variants_sample500.jsonl) を使う",
    )
    args = parser.parse_args()

    # モデル名の決定
    if args.model:
        model_name = args.model.strip()
    else:
        model_name = select_model_interactively(DEFAULT_MODEL_NAME)

    if not model_name:
        raise SystemExit("モデル名が空です。終了します。")

    # サンプル500を使うかどうか
    use_sample_500 = args.sample500 or USE_SAMPLE_500

    # ファイル名用の安全なモデル名
    safe_model_name = (
        model_name
        .replace(":", "_")
        .replace("/", "_")
    )

    # 入出力ファイルパスの決定
    if use_sample_500:
        variants_jsonl = ROOT_DIR / "data" / "json" / "jcsq_mcqa_variants_sample500.jsonl"
        out_csv = ROOT_DIR / "outputs_jcsq" / "mcqa_results" / f"ollama_{safe_model_name}_sample500.csv"
    else:
        variants_jsonl = ROOT_DIR / "data" / "json" / "jcsq_mcqa_variants.jsonl"
        out_csv = ROOT_DIR / "outputs_jcsq" / "mcqa_results" / f"ollama_{safe_model_name}_full.csv"

    print("=== 設定 ===")
    print(f"  OLLAMA_HOST : {OLLAMA_HOST}")
    print(f"  MODEL_NAME  : {model_name}")
    print(f"  SAFE_NAME   : {safe_model_name}")
    print(f"  INPUT  JSONL: {variants_jsonl}")
    print(f"  OUTPUT CSV  : {out_csv}")
    print("================\n")

    instances = load_jsonl(variants_jsonl)
    print(f"loaded {len(instances)} MCQA instances from {variants_jsonl}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.writer(f_out)
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
            variant = inst["variant"]
            question = inst["question"]
            choices = inst["choices"]
            answer_index = inst["answer_index"]

            prompt = build_prompt(question, choices)

            try:
                content = call_ollama_chat(prompt, model_name)
            except Exception as e:
                print(f"[{i+1}/{len(instances)}] qid={qid} ERROR: {e}")
                content = ""

            pred_index, pred_letter = parse_choice_letter(content)
            correct = (pred_index == answer_index)

            writer.writerow(
                [
                    qid,
                    variant,
                    question,
                    json.dumps(choices, ensure_ascii=False),
                    answer_index,
                    pred_index if pred_index is not None else "",
                    pred_letter,
                    content,
                    int(correct) if pred_index is not None else "",
                ]
            )

            if (i + 1) % 100 == 0:
                print(f"processed {i+1}/{len(instances)}")

            # モデルが重い場合は少し待つ（必要に応じて調整）
            time.sleep(0.01)

    print(f"saved results to {out_csv}")


if __name__ == "__main__":
    main()
