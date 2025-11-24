# mcqt.py
# 近1・中1・遠1（1–1–1）方式。MIDは guide / taec の切替可能。
# 正解文は外部ファイルから読み込み（最優先：関数別JSONディレクトリ）。LLMからは取得しない。
# Priority: CORRECT_JSON_DIR > CORRECT_JSON > CORRECT_CSV > LEGACY_POOL_DIR
import os, re, json, random, sys
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

# inputs
FUNC_CSV   = os.getenv("FUNC_CSV", "data/csv/functions.csv")
SRC_FILE   = os.getenv("SRC_FILE", "data/original/2048.c")
AXES_CSV   = os.getenv("AXES_CSV", "data/csv/comment_templates.csv")  # taec用（Precise/Unambiguous/Exhaustive）
# gold answers (優先順：DIR > JSON > CSV > LEGACY)
CORRECT_JSON_DIR = os.getenv("CORRECT_JSON_DIR", "").strip()   # 関数別JSON: <dir>/<function>.json
CORRECT_JSON     = os.getenv("CORRECT_JSON", "").strip()       # まとめJSONファイル（obj/arrayどちらでも）
CORRECT_CSV      = os.getenv("CORRECT_CSV", "").strip()        # CSV（function_name,correct 等）
LEGACY_POOL_DIR  = os.getenv("LEGACY_POOL_DIR", "").strip()    # 旧プール（choices+answer_index）
FAIL_ON_MISSING_CORRECT = os.getenv("FAIL_ON_MISSING_CORRECT", "1") == "1"

# thresholds & knobs（1–1–1）
CLOSE_MIN      = float(os.getenv("SIM_CLOSE_MIN", "0.78"))  # near 下限
FAR_MAX        = float(os.getenv("SIM_FAR_MAX", "0.55"))    # far 上限
DIVERSITY_RHO  = float(os.getenv("DIVERSITY_RHO", "0.85"))  # near/mid 等との「似すぎ」上限
MMR_LAMBDA     = float(os.getenv("MMR_LAMBDA", "0.55"))     # far MMR（距離 vs 非冗長）
MID_SELECTOR   = os.getenv("MID_SELECTOR", "guide").lower()  # "guide" or "taec"
LAMBDA_MID     = float(os.getenv("LAMBDA_MID", "0.50"))     # mid の中心距離 vs near類似の重み
# taec（擬似TAEC）検品用しきい値
MID_PRECISE_MIN = float(os.getenv("MID_PRECISE_MIN", "0.65"))
MID_UNAMBIG_MAX = float(os.getenv("MID_UNAMBIG_MAX", "0.55"))
MID_EXHAUST_MAX = float(os.getenv("MID_EXHAUST_MAX", "0.55"))

# Run directories (mode-separated + optional tag)
POOL_DIR_BASE   = os.getenv("POOL_DIR", "data/mcqt/choices")
DEBUG_DIR_BASE  = os.getenv("DEBUG_DIR", "debug_outputs")
METRIC_DIR_BASE = os.getenv("METRIC_DIR", "debug_metrics")
RUN_TAG         = os.getenv("RUN_TAG", "").strip()  # 任意の実験タグを更にサブフォルダに
_sub = [MID_SELECTOR] + ([RUN_TAG] if RUN_TAG else [])
POOL_DIR   = os.path.join(POOL_DIR_BASE,   *[p for p in _sub if p])
DEBUG_DIR  = os.path.join(DEBUG_DIR_BASE,  *[p for p in _sub if p])
METRIC_DIR = os.path.join(METRIC_DIR_BASE, *[p for p in _sub if p])
os.makedirs(POOL_DIR, exist_ok=True); os.makedirs(DEBUG_DIR, exist_ok=True); os.makedirs(METRIC_DIR, exist_ok=True)

# controls
ONLY_FUNC     = (os.getenv("ONLY_FUNC", "") or "").strip()
SKIP_EXISTING = os.getenv("SKIP_EXISTING", "0") == "1"
MAX_REGEN     = int(os.getenv("REGEN_ROUNDS", "3"))
OPENAI_MODEL_DIST = os.getenv("OPENAI_MODEL_DIST", "gpt-4o")

random.seed(42); np.random.seed(42)

# --- Auto-detect CORRECT_JSON_DIR if not explicitly set ---
def _autodetect_correct_dir():
    global CORRECT_JSON_DIR
    if CORRECT_JSON_DIR and os.path.isdir(CORRECT_JSON_DIR):
        return
    cand = []
    # 1) Conventional relative path from CWD
    cand.append(os.path.normpath("data/mcqt/correct_choices"))
    # 2) Relative to this script location
    try:
        _here = os.path.dirname(os.path.abspath(__file__))
        cand.append(os.path.normpath(os.path.join(_here, "..", "..", "data", "mcqt", "correct_choices")))
        cand.append(os.path.normpath(os.path.join(_here, "..", "data", "mcqt", "correct_choices")))
    except Exception:
        pass
    for c in cand:
        if os.path.isdir(c):
            CORRECT_JSON_DIR = c
            print(f"[autodetect] CORRECT_JSON_DIR -> {CORRECT_JSON_DIR}")
            return
    # keep as-is if not found

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

# ================= Gold loaders =================
# LEGACY（旧プール）
def get_correct_from_legacy(name: str) -> Optional[str]:
    if not LEGACY_POOL_DIR:
        return None
    try:
        p = os.path.join(LEGACY_POOL_DIR, f"{name}.json")
        if not os.path.exists(p): return None
        with open(p, "r", encoding="utf-8") as f: obj = json.load(f)
        idx = obj.get("answer_index", 0); idx = (idx[0] if isinstance(idx, list) else idx)
        ch = obj.get("choices", [])
        return (ch[idx] or "").strip() if 0 <= idx < len(ch) else None
    except Exception:
        return None

# 関数別JSONディレクトリ
def get_correct_from_json_dir(name: str) -> Optional[str]:
    base = CORRECT_JSON_DIR
    if not base:
        return None
    path = os.path.join(base, f"{name}.json")
    if not os.path.exists(path):
        print(f"[gold] not found file: {path}")
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            for k in ("correct","gold","label","正解"):
                v = obj.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
            print(f"[gold] keys missing in {path} (expected: correct/gold/label/正解)")
        elif isinstance(obj, str) and obj.strip():
            return obj.strip()
    except Exception as e:
        print(f"[gold] JSON parse error in {path}: {e}")
        return None
    return None

# まとめJSON（オブジェクト/配列両対応）
_correct_map_json_file: Optional[Dict[str,str]] = None
def load_correct_map_from_json_file() -> Dict[str, str]:
    global _correct_map_json_file
    if _correct_map_json_file is not None:
        return _correct_map_json_file
    mp: Dict[str, str] = {}
    if not CORRECT_JSON or not os.path.exists(CORRECT_JSON):
        _correct_map_json_file = mp; return mp
    try:
        with open(CORRECT_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        def set_one(fn, cor):
            fn = (fn or "").strip(); cor = (cor or "").strip()
            if fn and cor: mp[fn] = cor
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, str):
                    set_one(k, v)
                elif isinstance(v, dict):
                    for ck in ("correct","gold","label","正解"):
                        if ck in v and isinstance(v[ck], str):
                            set_one(k, v[ck]); break
        elif isinstance(data, list):
            for row in data:
                if not isinstance(row, dict): continue
                fn = row.get("function_name") or row.get("関数名")
                cor = row.get("correct") or row.get("gold") or row.get("label") or row.get("正解")
                set_one(fn, cor)
    except Exception:
        pass
    _correct_map_json_file = mp
    if mp: print(f"[correct] loaded {len(mp)} entries from CORRECT_JSON: {CORRECT_JSON}")
    return mp

# CSV
_correct_map_csv: Optional[Dict[str,str]] = None
def load_correct_map_from_csv() -> Dict[str, str]:
    global _correct_map_csv
    if _correct_map_csv is not None:
        return _correct_map_csv
    mp: Dict[str,str] = {}
    if not CORRECT_CSV or not os.path.exists(CORRECT_CSV):
        _correct_map_csv = mp; return mp
    try:
        df = pd.read_csv(CORRECT_CSV, encoding="utf-8-sig")
        fn_col = None
        for c in df.columns:
            if c in ("function_name","関数名"):
                fn_col = c; break
        if fn_col is None:
            raise ValueError("CORRECT_CSV に function_name（または 関数名）列が必要です")
        corr_col = None
        for c in df.columns:
            lc = str(c).strip().lower()
            if lc in ("correct","gold","label","正解"):
                corr_col = c; break
        if corr_col is None:
            raise ValueError("CORRECT_CSV に正解列（correct/gold/label/正解）が必要です")
        for fn, cor in zip(df[fn_col].astype(str), df[corr_col].astype(str)):
            mp[str(fn).strip()] = (cor or "").strip()
    except Exception as e:
        print(f"[WARN] CORRECT_CSV 読込で例外: {e}")
    _correct_map_csv = mp
    if mp: print(f"[correct] loaded {len(mp)} entries from CORRECT_CSV: {CORRECT_CSV}")
    return mp

def get_correct(name: str) -> Optional[str]:
    # 優先：DIR > JSON > CSV > LEGACY
    cor = get_correct_from_json_dir(name)
    if cor: return cor
    mpj = load_correct_map_from_json_file()
    cor = mpj.get(name)
    if cor: return cor
    mpc = load_correct_map_from_csv()
    cor = mpc.get(name)
    if cor: return cor
    return get_correct_from_legacy(name)

# ================= LLM calls（正解はLLMから取得しない） =================
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
        model=OPENAI_MODEL_DIST,
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

    # 入力必須チェック
    if not (CORRECT_JSON_DIR or CORRECT_JSON or CORRECT_CSV or LEGACY_POOL_DIR):
        print("[FATAL] CORRECT_JSON_DIR / CORRECT_JSON / CORRECT_CSV / LEGACY_POOL_DIR のいずれかを指定してください。")
        sys.exit(1)

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

    # 擬似TAEC（taecモード時のみ利用）
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

        # 正解は外部からのみ（LLMに依存しない）
        correct = get_correct(name)
        if not correct or not correct.strip():
            msg = f"[ERROR] 正解が見つかりません: {name}"
            if FAIL_ON_MISSING_CORRECT:
                print(msg); sys.exit(2)
            else:
                print(msg + " -> skip"); continue

        # 誤答候補の生成＋選抜（1–1–1）
        pick_info = None
        for r in range(1, MAX_REGEN + 1):
            structured = ask_structured_distractors(correct, code, name, round_id=r)
            if not structured or "distractors" not in structured: continue
            axes_row = axes_map.get(name, {}) if axes_map else None
            pick_info = pick_choices_1_1_1(correct, structured, axes_row)
            if pick_info["near"] and pick_info["mid"] and pick_info["far"]: break

        if not pick_info:
            print(f"[WARN] 候補生成/選抜に失敗: {name}")
            if FAIL_ON_MISSING_CORRECT: sys.exit(3)
            else: continue

        near = pick_info.get("near")
        mid  = pick_info.get("mid")
        far1 = pick_info.get("far")

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
        sims = pick_info.get("sims", {})
        save_json(os.path.join(METRIC_DIR, f"{name}.json"), {
            "mode": MID_SELECTOR, "run_tag": RUN_TAG or None,
            "correct": correct, "near": near, "mid": mid, "far": far1,
            "near_src": pick_info.get("near_src"),
            "sim_near": sims.get("near"), "sim_mid": sims.get("mid"),
            "CLOSE_MIN": CLOSE_MIN, "FAR_MAX": FAR_MAX,
            "MID_PRECISE_MIN": MID_PRECISE_MIN, "MID_UNAMBIG_MAX": MID_UNAMBIG_MAX, "MID_EXHAUST_MAX": MID_EXHAUST_MAX,
            "DIVERSITY_RHO": DIVERSITY_RHO, "MMR_LAMBDA": MMR_LAMBDA,
            "sources": {
                "CORRECT_JSON_DIR": bool(CORRECT_JSON_DIR),
                "CORRECT_JSON": bool(CORRECT_JSON),
                "CORRECT_CSV": bool(CORRECT_CSV),
                "LEGACY_POOL_DIR": bool(LEGACY_POOL_DIR),
            }
        })
        print(f"[MCQT 1-1-1] saved: {out_path}")

if __name__ == "__main__":
    main()
