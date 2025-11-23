"""旧MCQTスクリプト 近1・遠2選択肢生成＋MMR選定"""
import os
import re
import json
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util

# ================= Setup =================
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model = SentenceTransformer(os.getenv("SBERT_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"))

FUNC_CSV = os.getenv("FUNC_CSV", "data/csv/functions.csv")
SRC_FILE = os.getenv("SRC_FILE", "data/original/2048.c")
POOL_DIR = os.getenv("POOL_DIR", "data/mcqt/choices")
DEBUG_DIR = os.getenv("DEBUG_DIR", "debug_outputs")
METRIC_DIR = os.getenv("METRIC_DIR", "debug_metrics")
os.makedirs(POOL_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)
os.makedirs(METRIC_DIR, exist_ok=True)

# thresholds & knobs
SIM_CLOSE_MIN = float(os.getenv("SIM_CLOSE_MIN", "0.78"))
SIM_FAR_MAX   = float(os.getenv("SIM_FAR_MAX", "0.55"))
MMR_LAMBDA    = float(os.getenv("MMR_LAMBDA", "0.55"))
MAX_REGEN     = int(os.getenv("REGEN_ROUNDS", "3"))

# controls
ONLY_FUNC       = (os.getenv("ONLY_FUNC", "") or "").strip()
SKIP_EXISTING   = os.getenv("SKIP_EXISTING", "0") == "1"
LEGACY_POOL_DIR = (os.getenv("LEGACY_POOL_DIR", "") or "").strip()

random.seed(42)
np.random.seed(42)

# ================= Utils =================
def save_text(path: str, content: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def save_json(path: str, obj: Dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def extract_json_hardened(text: str, debug_name: str = "unknown"):
    try:
        if not isinstance(text, str) or not text.strip():
            print(f"[{debug_name}] extract_json_hardened: empty/non-str")
            return None
        s = text.strip()
        if s.startswith("{") and s.endswith("}"):
            try: return json.loads(s)
            except Exception as e: print(f"[{debug_name}] whole-text parse failed: {e}")
        m = re.search(r"```json\s*([\s\S]+?)\s*```", s, flags=re.IGNORECASE)
        if m:
            chunk = m.group(1).strip()
            try: return json.loads(chunk)
            except Exception as e: print(f"[{debug_name}] fenced json parse failed: {e}")
        m2 = re.search(r"```\s*([\s\S]+?)\s*```", s)
        if m2:
            chunk = m2.group(1).strip()
            lb = chunk.find("{"); rb = chunk.rfind("}")
            if lb != -1 and rb != -1 and rb > lb:
                chunk = chunk[lb:rb+1]
            try: return json.loads(chunk)
            except Exception as e: print(f"[{debug_name}] fenced block parse failed: {e}")
        lb = s.find("{"); rb = s.rfind("}")
        if lb != -1 and rb != -1 and rb > lb:
            chunk = s[lb:rb+1]
            try: return json.loads(chunk)
            except Exception as e: print(f"[{debug_name}] heuristic slice parse failed: {e}")
        print(f"[{debug_name}] extract_json_hardened: JSONブロックが見つかりませんでした")
        return None
    except Exception as e:
        print(f"[{debug_name}] extract_json_hardened: unexpected error: {e}")
        return None

def cos_sim_text(t1: str, t2: str) -> float:
    emb = model.encode([t1, t2], convert_to_tensor=True)
    return float(util.cos_sim(emb[0], emb[1]).item())

def batch_sims(ref: str, cands: List[str]) -> List[float]:
    if not cands: return []
    E = model.encode([ref] + cands, convert_to_tensor=True)
    sims = util.cos_sim(E[0], E[1:]).cpu().numpy()[0].tolist()
    return [float(s) for s in sims]

def pairwise_sims(texts: List[str]):
    if not texts: return np.zeros((0,0), dtype=float)
    E = model.encode(texts, convert_to_tensor=True)
    return util.cos_sim(E, E).cpu().numpy()

def dedup_semantic(lines: List[str], thresh: float = 0.90) -> List[str]:
    out = []
    for t in lines:
        t = (t or "").strip()
        if not t: continue
        if all(cos_sim_text(t, u) < thresh for u in out):
            out.append(t)
    return out

def load_legacy_correct(name: str) -> Optional[str]:
    if not LEGACY_POOL_DIR:
        return None
    try:
        p = os.path.join(LEGACY_POOL_DIR, f"{name}.json")
        if not os.path.exists(p):
            return None
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
        idx = obj.get("answer_index", 0)
        if isinstance(idx, list): idx = idx[0]
        ch = obj.get("choices", [])
        if 0 <= idx < len(ch):
            return (ch[idx] or "").strip()
    except Exception as e:
        print(f"[{name}] load_legacy_correct error: {e}")
    return None

# ================= LLM calls =================
def ask_initial_mcqa(code: str, name: str):
    prompt = f"""
以下のC関数 `{name}` の目的を説明する4択問題を作成してください。

```c
{code}
```

出力条件：
- 問題文は「関数の目的を最も正しく説明している選択肢は？」に固定
- 選択肢は4つ。正解は必ず1つだけ
- 出力は必ずJSONのみ、注釈や説明は不要
- "answer_index" は0〜3の整数（例：2）

{{
  "question": "関数の目的を最も正しく説明している選択肢は？",
  "choices": ["選択肢A", "選択肢B", "選択肢C", "選択肢D"],
  "answer_index": 2
}}
""".strip()
    res = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL_CORRECT", "gpt-5-chat-2025-09-30"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.25,
        response_format={"type": "json_object"},
        max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "900")),
    )
    parsed = getattr(res.choices[0].message, "parsed", None)
    if isinstance(parsed, dict):
        save_text(os.path.join(DEBUG_DIR, f"{name}_initial.txt"), json.dumps(parsed, ensure_ascii=False, indent=2))
        return parsed
    content = res.choices[0].message.content or ""
    save_text(os.path.join(DEBUG_DIR, f"{name}_initial.txt"), content)
    return extract_json_hardened(content, debug_name=name)

def ask_structured_distractors(correct: str, code: str, name: str, n_far=6, n_mid=4, round_id=1):
    prompt = f"""
以下の正解文に対して、似ているが誤った1文 (close) を1つ、
意味的に異なる誤った文 (medium) を{n_mid}、
正解と大きく異なる誤った文 (far) を{n_far} 作成してください。合計 {1+n_mid+n_far} 個。

正解:
"{correct}"

対象Cコード（参考・厳密一致は不要）:
```c
{code}
```

出力はJSONのみ。以下の形式で厳密に：
{{
  "distractors": {{
    "close": ["..."],
    "medium": ["...", "..."],
    "far": ["...", "..."]
  }}
}}
""".strip()
    res = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL_DIST", "gpt-5-chat-2025-09-30"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.85 if round_id == 1 else 0.95,
        top_p=0.95,
        response_format={"type": "json_object"},
        max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "900")),
    )
    parsed = getattr(res.choices[0].message, "parsed", None)
    if isinstance(parsed, dict):
        save_text(os.path.join(DEBUG_DIR, f"{name}_distractors_round{round_id}.txt"), json.dumps(parsed, ensure_ascii=False, indent=2))
        return parsed
    content = res.choices[0].message.content or ""
    save_text(os.path.join(DEBUG_DIR, f"{name}_distractors_round{round_id}.txt"), content)
    return extract_json_hardened(content, debug_name=f"{name}/round{round_id}")

# ================= Selection =================
def mmr_select_far(correct: str, far_pool: List[str], k: int = 2, lam: float = MMR_LAMBDA) -> List[str]:
    far_pool = [t.strip() for t in far_pool if t and t.strip()]
    if not far_pool: return []
    sims_to_correct = batch_sims(correct, far_pool)
    S_pair = pairwise_sims(far_pool)  # pairwise sims among candidates

    # start with farthest
    order = np.argsort(sims_to_correct).tolist()
    selected = [order[0]] if order else []

    while len(selected) < min(k, len(far_pool)):
        best_i, best_score = None, float("inf")
        for i in range(len(far_pool)):
            if i in selected: continue
            red = 0.0
            if selected:
                red = max(float(S_pair[i, j]) for j in selected)
            score = lam * float(sims_to_correct[i]) + (1.0 - lam) * red
            if score < best_score:
                best_score, best_i = score, i
        if best_i is None: break
        selected.append(best_i)

    return [far_pool[i] for i in selected]

def pick_choices(correct: str, structured: Dict) -> Dict:
    d = structured.get("distractors", {}) if structured else {}
    close  = dedup_semantic(d.get("close", []) or [], 0.92)[:3]
    medium = dedup_semantic(d.get("medium", []) or [], 0.90)
    far    = dedup_semantic(d.get("far", []) or [], 0.88)

    sims_close = batch_sims(correct, close)
    sims_mid   = batch_sims(correct, medium)
    sims_far   = batch_sims(correct, far)

    picked_close = None
    if close:
        idx = int(np.argmax(sims_close))
        picked_close = close[idx]  # しきい値で弾く場合は sims_close[idx] >= SIM_CLOSE_MIN を確認

    far_candidates = [t for t, s in zip(far, sims_far) if s <= SIM_FAR_MAX]

    if len(far_candidates) < 2 and medium:
        far_from_mid = [t for t, s in zip(medium, sims_mid) if s <= SIM_FAR_MAX]
        for t in far_from_mid:
            if all(cos_sim_text(t, u) < 0.85 for u in far_candidates):
                far_candidates.append(t)

    if len(far_candidates) < 2:
        flat = dedup_semantic((far or []) + (medium or []) + (close or []), 0.90)
        sims_flat = batch_sims(correct, flat)
        order = np.argsort(sims_flat).tolist()
        for i in order:
            t = flat[i]
            if all(cos_sim_text(t, u) < 0.85 for u in far_candidates):
                far_candidates.append(t)
            if len(far_candidates) >= 2:
                break

    picked_far = mmr_select_far(correct, far_candidates, k=2, lam=MMR_LAMBDA)

    return {"picked_close": picked_close, "picked_far": picked_far[:2]}

# ================= Main =================
def main():
    print(f"[config] ONLY_FUNC={ONLY_FUNC!r}, SKIP_EXISTING={SKIP_EXISTING}, LEGACY_POOL_DIR={bool(LEGACY_POOL_DIR)}, POOL_DIR={POOL_DIR}")
    func_df = pd.read_csv(FUNC_CSV, encoding="utf-8-sig")

    # ---- Strict filtering at DataFrame level (確実に1件だけにする) ----
    if ONLY_FUNC:
        if "関数名" in func_df.columns:
            func_df = func_df[func_df["関数名"].astype(str).str.strip() == ONLY_FUNC]
        elif "function_name" in func_df.columns:
            func_df = func_df[func_df["function_name"].astype(str).str.strip() == ONLY_FUNC]
        else:
            print("[WARN] 関数名の列が見つかりませんでした（'関数名' or 'function_name'）")
            func_df = func_df.iloc[0:0]  # empty
        print(f"[filter] rows after ONLY_FUNC filter: {len(func_df)}")

    with open(SRC_FILE, encoding="utf-8") as f:
        src_lines = f.readlines()

    for _, row in func_df.iterrows():
        # 名前列の互換取得
        if "関数名" in row:
            name = str(row["関数名"]).strip()
        elif "function_name" in row:
            name = str(row["function_name"]).strip()
        else:
            print("[WARN] rowに関数名列がありません。スキップ")
            continue

        out_path = os.path.join(POOL_DIR, f"{name}.json")
        if SKIP_EXISTING and os.path.exists(out_path):
            print(f"[skip] exists: {out_path}")
            continue

        start = int(row["開始行"] if "開始行" in row else row["start_line"]) - 1
        end   = int(row["終了行"] if "終了行" in row else row["end_line"])
        code = "".join(src_lines[start:end])

        # 正解：legacy優先
        correct = load_legacy_correct(name)
        if correct:
            save_text(os.path.join(DEBUG_DIR, f"{name}_legacy_correct.txt"), correct)
        else:
            initial = ask_initial_mcqa(code, name)
            if not initial or "choices" not in initial:
                print(f"{name}: 正解抽出失敗 (skip)")
                continue
            idx = initial.get("answer_index", 0)
            if isinstance(idx, list): idx = idx[0]
            ch = initial.get("choices", [])
            if not (0 <= idx < len(ch)):
                print(f"{name}: answer_index範囲外 (skip)")
                continue
            correct = (ch[idx] or "").strip()
            if not correct:
                print(f"{name}: 正解が空 (skip)")
                continue

        pick_info = None
        structured = None
        for r in range(1, MAX_REGEN + 1):
            structured = ask_structured_distractors(correct, code, name, round_id=r)
            if not structured or "distractors" not in structured:
                continue
            pick_info = pick_choices(correct, structured)
            if pick_info["picked_close"] and len(pick_info["picked_far"]) >= 2:
                break

        near = pick_info["picked_close"] if pick_info else None
        fars = pick_info["picked_far"] if pick_info else []

        # 最終4択構築
        final = [correct]
        if near: final.append(near)
        final += (fars[:2] if fars else [])
        # 4未満なら適当に埋める（念のため）
        while len(final) < 4:
            final.append(f"その他の選択肢{len(final)+1}")
        random.shuffle(final)
        try:
            answer_index = final.index(correct)
        except ValueError:
            final[0] = correct
            answer_index = 0

        out = {"question": "関数の目的を最も正しく説明している選択肢は？",
               "choices": final[:4], "answer_index": answer_index}
        save_json(out_path, out)
        save_json(os.path.join(METRIC_DIR, f"{name}.json"), {
            "picked_close": near, "picked_far": fars[:2], "correct": correct,
            "SIM_CLOSE_MIN": SIM_CLOSE_MIN, "SIM_FAR_MAX": SIM_FAR_MAX,
            "MMR_LAMBDA": MMR_LAMBDA, "legacy_used": bool(load_legacy_correct(name))
        })
        print(f"MCQT saved: {out_path}")

if __name__ == "__main__":
    main()
