"""MCQTスクリプト 近1・中1・遠1方式
中帯域はguide / TAEC で切替可能。
"""
import os, re, json, random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util

# ================= Setup =================
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model = SentenceTransformer(os.getenv("SBERT_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"))

FUNC_CSV   = os.getenv("FUNC_CSV", "data/csv/functions.csv")
SRC_FILE   = os.getenv("SRC_FILE", "data/original/2048.c")
AXES_CSV   = os.getenv("AXES_CSV", "data/csv/comment_templates.csv")  # 擬似TAEC（Precise/Unambiguous/Exhaustive）

# thresholds & knobs（1–1–1）
CLOSE_MIN      = float(os.getenv("SIM_CLOSE_MIN", "0.78"))  # near 下限
FAR_MAX        = float(os.getenv("SIM_FAR_MAX", "0.55"))    # far 上限
DIVERSITY_RHO  = float(os.getenv("DIVERSITY_RHO", "0.85"))  # near/mid 等との「似すぎ」上限
MMR_LAMBDA     = float(os.getenv("MMR_LAMBDA", "0.55"))     # far 用MMRの重み（距離 vs 非冗長）
LAMBDA_MID     = float(os.getenv("LAMBDA_MID", "0.50"))     # mid の中心距離 vs near類似の重み

MID_SELECTOR   = os.getenv("MID_SELECTOR", "taec").lower()  # "guide" or "taec"

# taec（擬似TAEC）検品用しきい値
MID_PRECISE_MIN = float(os.getenv("MID_PRECISE_MIN", "0.65"))
MID_UNAMBIG_MAX = float(os.getenv("MID_UNAMBIG_MAX", "0.55"))
MID_EXHAUST_MAX = float(os.getenv("MID_EXHAUST_MAX", "0.55"))

# Run directories (mode-separated + optional tag)
POOL_DIR_BASE   = os.getenv("POOL_DIR", "data/mcqt/choices")
DEBUG_DIR_BASE  = os.getenv("DEBUG_DIR", "data/mcqt/debug_outputs")
METRIC_DIR_BASE = os.getenv("METRIC_DIR", "data/mcqt/debug_metrics")
RUN_TAG         = os.getenv("RUN_TAG", "").strip()  # 任意の実験タグを更にサブフォルダに
_sub = [MID_SELECTOR] + ([RUN_TAG] if RUN_TAG else [])
POOL_DIR   = os.path.join(POOL_DIR_BASE,   *[p for p in _sub if p])
DEBUG_DIR  = os.path.join(DEBUG_DIR_BASE,  *[p for p in _sub if p])
METRIC_DIR = os.path.join(METRIC_DIR_BASE, *[p for p in _sub if p])
os.makedirs(POOL_DIR, exist_ok=True); os.makedirs(DEBUG_DIR, exist_ok=True); os.makedirs(METRIC_DIR, exist_ok=True)

# controls
ONLY_FUNC       = (os.getenv("ONLY_FUNC", "") or "").strip()
SKIP_EXISTING   = os.getenv("SKIP_EXISTING", "0") == "1"
LEGACY_POOL_DIR = (os.getenv("LEGACY_POOL_DIR", "") or "").strip()
MAX_REGEN       = int(os.getenv("REGEN_ROUNDS", "3"))

random.seed(42); np.random.seed(42)

# ================= Utils =================
def save_text(path: str, content: str):
    with open(path, "w", encoding="utf-8") as f: f.write(content)

def save_json(path: str, obj: Dict):
    with open(path, "w", encoding="utf-8") as f: json.dump(obj, f, ensure_ascii=False, indent=2)

def extract_json_hardened(text: str, debug_name: str = "unknown"):
    try:
        if not isinstance(text, str) or not text.strip(): return None
        s = text.strip()
        if s.startswith("{") and s.endswith("}"):
            try: return json.loads(s)
            except: pass
        m = re.search(r"```json\s*([\s\S]+?)\s*```", s, flags=re.I)
        if m:
            try: return json.loads(m.group(1).strip())
            except: pass
        m2 = re.search(r"```\s*([\s\S]+?)\s*```", s)
        if m2:
            chunk = m2.group(1).strip()
            lb = chunk.find("{"); rb = chunk.rfind("}")
            if lb != -1 and rb != -1 and rb > lb:
                try: return json.loads(chunk[lb:rb+1])
                except: pass
        lb = s.find("{"); rb = s.rfind("}")
        if lb != -1 and rb != -1 and rb > lb:
            try: return json.loads(s[lb:rb+1])
            except: pass
        return None
    except Exception:
        return None

def cos_sim_text(t1: str, t2: str) -> float:
    if not t1 or not t2: return 0.0
    emb = model.encode([t1, t2], convert_to_tensor=True)
    return float(util.cos_sim(emb[0], emb[1]).item())

def batch_sims(ref: str, cands: List[str]) -> List[float]:
    if not cands: return []
    E = model.encode([ref] + cands, convert_to_tensor=True)
    sims = util.cos_sim(E[0], E[1:]).cpu().numpy()[0].tolist()
    return [float(s) for s in sims]

def dedup_semantic(lines: List[str], thresh: float = 0.90) -> List[str]:
    out = []
    for t in lines:
        t = (t or "").strip()
        if not t: continue
        if all(cos_sim_text(t, u) < thresh for u in out): out.append(t)
    return out

def pairwise_max_sim(a: str, arr: List[str]) -> float:
    return max([cos_sim_text(a, b) for b in arr], default=0.0)

def load_legacy_correct(name: str) -> Optional[str]:
    if not LEGACY_POOL_DIR: return None
    try:
        p = os.path.join(LEGACY_POOL_DIR, f"{name}.json")
        if not os.path.exists(p): return None
        with open(p, "r", encoding="utf-8") as f: obj = json.load(f)
        idx = obj.get("answer_index", 0); idx = (idx[0] if isinstance(idx, list) else idx)
        ch = obj.get("choices", []); 
        return (ch[idx] or "").strip() if 0 <= idx < len(ch) else None
    except Exception:
        return None

# --- Auto-detect CORRECT_JSON_DIR if not explicitly set (safe) ---
def _autodetect_correct_dir():
    """
    CORRECT_JSON_DIR が未設定/未検出の場合に、自動で data/mcqt/correct_choices を探してセットする。
    globals() を使うので、関数定義位置が inputs セクションの前でも安全。
    """
    import os as _os
    base = globals().get("CORRECT_JSON_DIR", "")
    if base and _os.path.isdir(base):
        return

    cand = [
        _os.path.normpath("data/mcqt/correct_choices"),
    ]
    try:
        _here = _os.path.dirname(_os.path.abspath(__file__))
        cand.append(_os.path.normpath(_os.path.join(_here, "..", "..", "data", "mcqt", "correct_choices")))
        cand.append(_os.path.normpath(_os.path.join(_here, "..", "data", "mcqt", "correct_choices")))
    except Exception:
        pass

    for c in cand:
        if _os.path.isdir(c):
            globals()["CORRECT_JSON_DIR"] = c
            print(f"[autodetect] CORRECT_JSON_DIR -> {c}")
            return

    # 見つからなかった場合も、未定義 NameError は起こさないように空文字で定義しておく
    if "CORRECT_JSON_DIR" not in globals():
        globals()["CORRECT_JSON_DIR"] = ""
    print("[autodetect] CORRECT_JSON_DIR not found; continue with current config.")


def _aggregate_pools(structured_list):
    """複数ラウンドの distractors を集約して重複削除。"""
    close_all, mid_all, far_all = [], [], []
    for st in structured_list:
        if not st or "distractors" not in st: continue
        d = st["distractors"]
        close_all += (d.get("close") or [])
        mid_all   += (d.get("medium") or [])
        far_all   += (d.get("far") or [])
    close_all = dedup_semantic(close_all, 0.92)
    mid_all   = dedup_semantic(mid_all, 0.90)
    far_all   = dedup_semantic(far_all, 0.88)
    return close_all, mid_all, far_all

def _fallback_fill(correct, want, close_all, mid_all, far_all, near=None, mid=None):
    """
    欠けた near/mid/far を埋める簡易フォールバック。
    want: {"near":bool, "mid":bool, "far":bool}
    """
    filled = {}
    # near
    if want.get("near"):
        pool = dedup_semantic((close_all or []) + (mid_all or []), 0.92)
        sims = batch_sims(correct, pool)
        cand = []
        for t, s in zip(pool, sims):
            if s >= 0.97:  # ほぼコピペは除外
                continue
            if near and cos_sim_text(t, near) >= 0.95:
                continue
            cand.append((t, s))
        if cand:
            t, s = max(cand, key=lambda x: x[1])
            filled["near"] = t

    # mid
    if want.get("mid"):
        pool = dedup_semantic((mid_all or []) + (close_all or []), 0.92)
        sims = batch_sims(correct, pool)
        mu = 0.5 * (CLOSE_MIN + FAR_MAX)
        band = [(t, s) for t, s in zip(pool, sims) if FAR_MAX < s < CLOSE_MIN]
        if band:
            t, s = min(band, key=lambda ts: abs(ts[1] - mu))
            filled["mid"] = t

    # far
    if want.get("far"):
        pool = far_all[:]
        sims = batch_sims(correct, pool)
        cands = [(t, s) for t, s in zip(pool, sims) if s <= FAR_MAX]
        if not cands:
            pool = mid_all[:]
            sims = batch_sims(correct, pool)
            cands = [(t, s) for t, s in zip(pool, sims) if s <= FAR_MAX]
        if cands:
            def score(ts):
                t, s = ts
                red = 0.0
                if near: red = max(red, cos_sim_text(t, near))
                if mid:  red = max(red, cos_sim_text(t, mid))
                return (s + 0.2*red)  # s小 & 冗長性小
            t, s = min(cands, key=score)
            filled["far"] = t
    return filled

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
        model=os.getenv("OPENAI_MODEL_CORRECT", "gpt-4o"),
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

def ask_structured_distractors(correct: str, code: str, name: str, n_far=6, n_mid=6, round_id=1):
    prompt = f"""
以下の正解文に対して、似ているが誤った1文 (close) を1つ、
意味的にやや異なる誤った文 (medium) を{n_mid}個、
正解と大きく異なる誤った文 (far) を{n_far}個 作成してください。合計 {1+n_mid+n_far} 個。

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
        model=os.getenv("OPENAI_MODEL_DIST", "gpt-4o"),
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

# ================= Selection helpers =================
def mmr_select_one_far(correct: str, far_pool: List[str], already: List[str], lam: float = MMR_LAMBDA) -> Optional[str]:
    """
    候補から1件。目的：正答から遠く（sim小）＆既選択(near/mid)に似すぎない（max_sim小）
    スコア = lam * sim(correct,cand) + (1-lam) * max_sim(cand, already) を最小化。
    """
    pool = [t.strip() for t in far_pool if t and t.strip()]
    if not pool: return None
    sims_c = batch_sims(correct, pool)
    best_i, best_score = None, float("inf")
    for i, t in enumerate(pool):
        red = pairwise_max_sim(t, already)
        score = lam * float(sims_c[i]) + (1.0 - lam) * red
        if score < best_score and all(cos_sim_text(t, a) < DIVERSITY_RHO for a in already):
            best_score, best_i = score, i
    return pool[best_i] if best_i is not None else None

def load_axes_map(path=AXES_CSV) -> Dict[str, Dict[str, str]]:
    """擬似TAEC（Precise/Unambiguous/Exhaustive）を読み込む。無ければ空。"""
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
        if "function_name" not in df.columns and "関数名" in df.columns:
            df = df.rename(columns={"関数名": "function_name"})
        # 列名ゆらぎ
        ren = {}
        for c in df.columns:
            lc = c.lower()
            if lc == "precise": ren[c] = "Precise"
            if lc == "unambiguous": ren[c] = "Unambiguous"
            if lc == "exhaustive": ren[c] = "Exhaustive"
        if ren: df = df.rename(columns=ren)
        df["function_name"] = df["function_name"].astype(str)
        cols = [c for c in ["Precise","Unambiguous","Exhaustive"] if c in df.columns]
        if not cols: return {}
        return df.set_index("function_name")[cols].to_dict(orient="index")
    except Exception:
        return {}

def good_mid_precise_not_unamb_exhau(text: str, axes_row: Dict[str, str], near: Optional[str]) -> bool:
    """Preciseは高類似、Unambiguous/Exhaustiveは低類似、nearとも似すぎない。"""
    p = (axes_row.get("Precise") or "").strip()
    u = (axes_row.get("Unambiguous") or "").strip()
    e = (axes_row.get("Exhaustive") or "").strip()
    ok_prec = (cos_sim_text(text, p) >= MID_PRECISE_MIN) if p else True
    ng_unam = (cos_sim_text(text, u) <= MID_UNAMBIG_MAX) if u else True
    ng_exha = (cos_sim_text(text, e) <= MID_EXHAUST_MAX) if e else True
    ok_div  = (cos_sim_text(text, near) < DIVERSITY_RHO) if near else True
    return ok_prec and ng_unam and ng_exha and ok_div

# ================= 1–1–1 Selection =================
def select_near(correct: str, close: List[str], medium: List[str]) -> Tuple[Optional[str], Optional[float], str]:
    """near: 正答に最も近い候補（close優先、無ければmedium）"""
    cand = close[:] if close else medium[:]
    if not cand: return None, None, "none"
    sims = batch_sims(correct, cand)
    i = int(np.argmax(sims))
    return cand[i], float(sims[i]), ("close" if close else "medium")

def select_mid(correct: str, near: Optional[str], medium: List[str], close: List[str],
               axes_row: Optional[Dict[str, str]]) -> Tuple[Optional[str], Optional[float]]:
    """
    mid: FAR_MAX < sim < CLOSE_MIN の“間”から選ぶ。
    - guide: 中心 |sim - μ| を最小に（nearとの非冗長も軽く）
    - taec : Precise高・Unamb/Exhaust低 の検品を通した上で中心最小
    """
    pool = (medium or []) + (close or [])
    if not pool: return None, None
    sims = batch_sims(correct, pool)
    band = [(t, s) for t, s in zip(pool, sims) if FAR_MAX < s < CLOSE_MIN and (not near or cos_sim_text(t, near) < DIVERSITY_RHO)]
    if not band: return None, None

    # taecモードなら検品
    if MID_SELECTOR == "taec" and axes_row:
        band = [(t, s) for (t, s) in band if good_mid_precise_not_unamb_exhau(t, axes_row, near)]
        if not band: return None, None

    mu = 0.5 * (CLOSE_MIN + FAR_MAX)
    # guide: 中心に近い + nearに似すぎない（score最小）
    def score_mid(ts):
        t, s = ts
        red = cos_sim_text(t, near) if near else 0.0
        return LAMBDA_MID * abs(s - mu) + (1.0 - LAMBDA_MID) * red

    t, s = min(band, key=score_mid)
    return t, float(s)

def select_far(correct: str, near: Optional[str], mid: Optional[str],
               far: List[str], medium: List[str]) -> Optional[str]:
    """far: sim <= FAR_MAX の集合から MMR で1件（無ければ最小simのフォールバック）"""
    sims_f = batch_sims(correct, far)
    far_pool = [t for (t, s) in zip(far, sims_f) if s <= FAR_MAX]
    if not far_pool:
        sims_m = batch_sims(correct, medium)
        far_pool = [t for (t, s) in zip(medium, sims_m) if s <= FAR_MAX]  # 予備
    chosen = mmr_select_one_far(correct, far_pool, [x for x in [near, mid] if x]) if far_pool else None
    if chosen: return chosen
    # 最小simフォールバック
    flat = dedup_semantic((far or []) + (medium or []), 0.90)
    if not flat: return None
    sims = batch_sims(correct, flat)
    return flat[int(np.argmin(sims))]

def pick_choices_1_1_1(correct: str, structured: Dict, axes_row: Optional[Dict[str,str]] = None) -> Dict:
    d = structured.get("distractors", {}) if structured else {}
    close  = dedup_semantic(d.get("close", []) or [], 0.92)
    medium = dedup_semantic(d.get("medium", []) or [], 0.90)
    far    = dedup_semantic(d.get("far", []) or [], 0.88)

    near, sim_near, near_src = select_near(correct, close, medium)
    mid,  sim_mid            = select_mid(correct, near, medium, close, axes_row)
    far1                     = select_far(correct, near, mid, far, medium)

    return {"near": near, "mid": mid, "far": far1,
            "sims": {"near": sim_near, "mid": sim_mid},
            "near_src": near_src}

# ================= Main =================
def main():
    _autodetect_correct_dir()
    print(f"[config] ONLY_FUNC={ONLY_FUNC!r} SKIP_EXISTING={SKIP_EXISTING} MID_SELECTOR={MID_SELECTOR} POOL_DIR={POOL_DIR} CORRECT_JSON_DIR={CORRECT_JSON_DIR or '(none)'}")
    func_df = pd.read_csv(FUNC_CSV, encoding="utf-8-sig")
    # filter
    if ONLY_FUNC:
        if "関数名" in func_df.columns:
            func_df = func_df[func_df["関数名"].astype(str).str.strip() == ONLY_FUNC]
        elif "function_name" in func_df.columns:
            func_df = func_df[func_df["function_name"].astype(str).str.strip() == ONLY_FUNC]
        else:
            func_df = func_df.iloc[0:0]
        print(f"[filter] rows after ONLY_FUNC: {len(func_df)}")

    with open(SRC_FILE, encoding="utf-8") as f:
        src_lines = f.readlines()

    # 擬似TAEC（L3-L5）をロード（taecモード時のみ利用）
    axes_map = load_axes_map() if MID_SELECTOR == "taec" else {}

    for _, row in func_df.iterrows():
        # 名前列
        if "関数名" in row:
            name = str(row["関数名"]).strip()
        elif "function_name" in row:
            name = str(row["function_name"]).strip()
        else:
            print("[WARN] 関数名列が見つからずスキップ"); continue

        out_path = os.path.join(POOL_DIR, f"{name}.json")
        if SKIP_EXISTING and os.path.exists(out_path):
            print(f"[skip] exists: {out_path}")
            continue

        start = int(row["開始行"] if "開始行" in row else row["start_line"]) - 1
        end   = int(row["終了行"] if "終了行" in row else row["end_line"])
        code  = "".join(src_lines[start:end])

        # 正解：legacy優先→無ければLLM
        correct = load_legacy_correct(name)
        if correct:
            save_text(os.path.join(DEBUG_DIR, f"{name}_legacy_correct.txt"), correct)
        else:
            initial = ask_initial_mcqa(code, name)
            if not initial or "choices" not in initial:
                print(f"{name}: 正解抽出失敗 (skip)"); continue
            idx = initial.get("answer_index", 0); idx = (idx[0] if isinstance(idx, list) else idx)
            ch  = initial.get("choices", [])
            if not (0 <= idx < len(ch)):
                print(f"{name}: answer_index範囲外 (skip)"); continue
            correct = (ch[idx] or "").strip()
            if not correct:
                print(f"{name}: 正解が空 (skip)"); continue

        # 誤答候補の生成＋選抜（1–1–1）
        pick_info = None
        for r in range(1, MAX_REGEN + 1):
            structured = ask_structured_distractors(correct, code, name, round_id=r)
            if not structured or "distractors" not in structured: continue
            axes_row = axes_map.get(name, {}) if axes_map else None
            pick_info = pick_choices_1_1_1(correct, structured, axes_row)
            if pick_info["near"] and pick_info["mid"] and pick_info["far"]: break

        near = pick_info.get("near") if pick_info else None
        mid  = pick_info.get("mid")  if pick_info else None
        far1 = pick_info.get("far")  if pick_info else None

        # 最終4択構築
        final = [correct]
        if near: final.append(near)
        if mid:  final.append(mid)
        if far1: final.append(far1)
        while len(final) < 4:
            final.append(f"その他の選択肢{len(final)+1}")
        random.shuffle(final)
        try:
            answer_index = final.index(correct)
        except ValueError:
            final[0] = correct; answer_index = 0

        out = {"question": "関数の目的を最も正しく説明している選択肢は？",
               "choices": final[:4], "answer_index": answer_index}
        save_json(out_path, out)

        # メトリクス
        sims = pick_info.get("sims", {}) if pick_info else {}
        save_json(os.path.join(METRIC_DIR, f"{name}.json"), {
            "mode": MID_SELECTOR,
            "run_tag": RUN_TAG or None,
            "correct": correct,
            "near": near, "mid": mid, "far": far1,
            "near_src": pick_info.get("near_src") if pick_info else None,
            "sim_near": sims.get("near"), "sim_mid": sims.get("mid"),
            "CLOSE_MIN": CLOSE_MIN, "FAR_MAX": FAR_MAX,
            "MID_PRECISE_MIN": MID_PRECISE_MIN, "MID_UNAMBIG_MAX": MID_UNAMBIG_MAX, "MID_EXHAUST_MAX": MID_EXHAUST_MAX,
            "DIVERSITY_RHO": DIVERSITY_RHO, "MMR_LAMBDA": MMR_LAMBDA,
            "legacy_used": bool(load_legacy_correct(name))
        })
        print(f"[MCQT 1-1-1] saved: {out_path}")

if __name__ == "__main__":
    main()
