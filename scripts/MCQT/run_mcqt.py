"""
MCQをレベル別(0-7)で解かせるスクリプト (プロンプトにソースコードを一括で送信する)
python scripts/MCQT/run_mcqt.py --model gpt-4o --pool_root data/mcqt/choices/guide --levels 0-7 \--out_root outputs/mcqt_results --code_file data/original/2048.c
"""
import os
import re
import csv
import time
import json
import argparse
from pathlib import Path
from typing import List, Dict

# === モデル周り（run_mcqa.py準拠の最小実装） ===
from openai import OpenAI
import anthropic
import requests
from dotenv import load_dotenv
load_dotenv()


DEFAULT_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_LLAMACPP_HOST = os.getenv("LLAMACPP_HOST", "http://localhost:8080")

class OllamaClient:
    def __init__(self, host): self.host = host
    def ask(self, model, prompt):
        r = requests.post(f"{self.host}/api/generate", json={"model": model, "prompt": prompt, "stream": False})
        return r.json().get("response","").strip()

class LlamacppClient:
    def __init__(self, host): self.host = host
    def ask(self, prompt, n_predict=256):
        r = requests.post(f"{self.host}/completion", json={"prompt": prompt, "n_predict": n_predict})
        try: return r.json().get("content","").strip()
        except Exception: return r.text.strip()

def build_clients(model: str):
    if model.startswith("ollama-"):
        return ("ollama", model.split("-",1)[1], None, OllamaClient(DEFAULT_OLLAMA_HOST), None, None)
    if model.startswith("llamacpp-"):
        return ("llamacpp", model.split("-",1)[1], None, None, LlamacppClient(DEFAULT_LLAMACPP_HOST), None)
    if model.startswith("claude"):
        return ("claude", model, None, None, None, anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")))
    # default OpenAI
    return ("openai", model, OpenAI(api_key=os.getenv("OPENAI_API_KEY")), None, None, None)

# === 入力 ===
def parse_levels(s: str) -> List[int]:
    s = s.strip()
    if "-" in s:
        a,b = s.split("-",1)
        return list(range(int(a), int(b)+1))
    return [int(x) for x in s.split(",") if x.strip()!=""]

def list_pools_for_level(root: Path, level: int) -> List[Path]:
    # レベル別ディレクトリがあれば優先
    lvl_dir = root / f"L{level}"
    if lvl_dir.exists():
        return sorted(lvl_dir.glob("*.json"))
    # ルート直下（共通）の場合はそれを使う
    return sorted(root.glob("*.json"))

def load_json(p: Path) -> Dict:
    with p.open(encoding="utf-8") as f:
        return json.load(f)

# === プロンプト/抽出 ===
def make_prompt(question: str, choices: list, code_block: str = None, deepseek_mode: bool = False) -> str:
    choice_lines = "\n".join(f"{chr(65+i)}. {c}" for i,c in enumerate(choices))
    if deepseek_mode:
        return (
            "次の問題に対し、必ず A / B / C / D の1文字だけを1行で出力してください。\n"
            "説明や理由は禁止です。\n"
            "<think>タグも禁止です。\n"
            "出力例: B\n\n"
            + (f"```c\n{code_block}\n```\n\n" if code_block else "")
            + f"問題: {question}\n{choice_lines}"
        )
    else:
        return (
            "以下の四択問題に答えてください。\n"
            "【重要】出力は必ず以下の中から1文字だけ選んでください：\n"
            "A / B / C / D\n\n"
            "絶対に文章や説明は書かないでください。\n"
            "もしわからない場合でも必ず A〜D の中から最も適切なものを選んでください。\n\n"
            + (f"```c\n{code_block}\n```\n\n" if code_block else "")
            + f"問題: {question}\n{choice_lines}"
        )

def extract_answer(raw: str, deepseek_mode: bool = False) -> str:
    m = re.search(r"\b([ABCD])\b", raw, re.IGNORECASE)
    if not m:
        m = re.search(r"(?:Option|Answer|選択肢)?\s*([ABCD])", raw, re.IGNORECASE)
    if not m:
        n = re.search(r"\b([1-4])\b", raw)
        if n:
            return {"1":"A","2":"B","3":"C","4":"D"}.get(n.group(1), "?")
        return "?"
    return m.group(1).upper()

# === 実行 ===
def run_level(mode, target_model, openai_client, ollama_client, llamacpp_client, claude_client,
              level: int, pool_root: Path, out_dir: Path, code_block: str, sleep: float):
    pools = list_pools_for_level(pool_root, level)
    if not pools:
        print(f"[WARN] L{level}: プールが見つかりません: {pool_root}/(L{level}|*)/*.json")
        return None

    out_csv = out_dir / f"L{level}.csv"
    rows = []
    deepseek_mode = "deepseek" in (target_model or "").lower()

    for i, p in enumerate(pools, 1):
        data = load_json(p)
        q = data.get("question","").strip()
        choices = data.get("choices", [])
        if len(choices) != 4:
            print(f"[SKIP] L{level}: 4択でない: {p.name}")
            continue
        prompt = make_prompt(q, choices, code_block=code_block if code_block else None, deepseek_mode=deepseek_mode)

        if mode == "openai":
            resp = openai_client.chat.completions.create(
                model=target_model, messages=[{"role":"user","content":prompt}], temperature=0
            )
            raw = resp.choices[0].message.content.strip()
        elif mode == "claude":
            resp = claude_client.messages.create(
                model=target_model, max_tokens=10, messages=[{"role":"user","content":prompt}]
            )
            raw = resp.content[0].text.strip()
        elif mode == "ollama":
            raw = ollama_client.ask(target_model, prompt)
        elif mode == "llamacpp":
            raw = llamacpp_client.ask(prompt)
        else:
            raw = "?"

        ans = extract_answer(raw, deepseek_mode=deepseek_mode)
        rows.append({"function": p.stem, "answer": ans})
        print(f"[L{level}] {p.stem}: {ans} | raw={raw}".encode("cp932", errors="ignore").decode("cp932"))

        # polite sleep for API-based models
        if mode in ("openai","claude"):
            time.sleep(sleep)

    # CSV出力
    out_dir.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["function","answer"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[DONE] L{level} -> {out_csv.as_posix()}")
    return out_csv

def main():
    ap = argparse.ArgumentParser(description="GUIDEプールをレベル別(0-7)で解かせるランナー")
    ap.add_argument("--model", required=True, help="モデル名 (openai/claude/ollama/llamacpp 前置子対応)")
    ap.add_argument("--pool_root", default="data/mcqt/choices/guide", help="GUIDEプールのルート")
    ap.add_argument("--levels", default="0-7", help="レベル指定: 例 '0-7' or '0,2,5'")
    ap.add_argument("--out_root", default="outputs/mcqa_results", help="出力ルート")
    ap.add_argument("--code_file", default="", help="任意: 全レベル共通で添付するCコード（2048.cなど）")
    ap.add_argument("--sleep", type=float, default=1.0, help="API呼び出し間隔（OpenAI/Claude）")
    args = ap.parse_args()

    levels = parse_levels(args.levels)
    pool_root = Path(args.pool_root)
    out_dir = Path(args.out_root) / args.model.replace(":", "-")

    code_block = ""
    if args.code_file:
        p = Path(args.code_file)
        if not p.exists():
            raise FileNotFoundError(f"コードが見つかりません: {p}")
        code_block = p.read_text(encoding="utf-8")

    mode, target_model, openai_client, ollama_client, llamacpp_client, claude_client = build_clients(args.model)

    generated = []
    for lv in levels:
        csv_path = run_level(mode, target_model, openai_client, ollama_client, llamacpp_client, claude_client,
                             lv, pool_root, out_dir, code_block, args.sleep)
        if csv_path: generated.append(csv_path.as_posix())

    print("=== 完了 ===")
    for p in generated:
        print(" -", p)
    print("採点: python mcqt_scorecheck.py --model", args.model, "--levels", args.levels, "--pool_dir", args.pool_root, "--out_root outputs/mcqa_scored")

if __name__ == "__main__":
    main()
