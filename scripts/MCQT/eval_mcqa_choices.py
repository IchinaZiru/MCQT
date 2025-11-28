
import os, json, glob, argparse, math
from pathlib import Path
import pandas as pd

try:
    from sentence_transformers import SentenceTransformer, util
    _HAS_SBERT = True
except Exception as _e:
    _HAS_SBERT = False
    _SBERT_IMPORT_ERR = str(_e)

# BLEURTは任意（重いため既定OFF）
def _load_bleurt(model_path: str):
    try:
        from bleurt import score
        return score.BleurtScorer(model_path)
    except Exception as e:
        print(f"[WARN] BLEURT読み込みに失敗: {e}")
        return None

def jaccard(a: str, b: str) -> float:
    import re
    ta = set(re.findall(r"\w+", (a or "").lower()))
    tb = set(re.findall(r"\w+", (b or "").lower()))
    if not ta or not tb: 
        return 0.0
    inter = len(ta & tb); uni = len(ta | tb)
    return inter/uni if uni else 0.0

def main():
    ap = argparse.ArgumentParser(description="MCQA/MCQT 選択肢品質評価（guide/taec等に対応）")
    ap.add_argument("--pool_dir", required=True, help="選択肢JSONのルート（例：data/mcqt/choices/guide）")
    ap.add_argument("--out_dir", required=True, help="レポート出力先（例：outputs/mcqa_quality/guide）")
    ap.add_argument("--close_min", type=float, default=float(os.getenv("SIM_CLOSE_MIN", "0.78")), help="近帯域の上限境界（CLOSE_MIN）")
    ap.add_argument("--far_max", type=float, default=float(os.getenv("SIM_FAR_MAX", "0.55")), help="遠帯域の下限境界（FAR_MAX）")
    ap.add_argument("--sbert_model", default=os.getenv("SBERT_MODEL","paraphrase-multilingual-MiniLM-L12-v2"))
    ap.add_argument("--bleurt", type=int, default=int(os.getenv("USE_BLEURT","0")), help="BLEURTを使うなら1（既定0）")
    ap.add_argument("--bleurt_model_path", default=os.getenv("BLEURT_MODEL","models/Bleurt"))
    args = ap.parse_args()

    pool_dir = args.pool_dir
    out_dir = args.out_dir
    close_min = args.close_min
    far_max = args.far_max

    os.makedirs(out_dir, exist_ok=True)

    # ロガー
    print(f"[config] pool_dir={pool_dir} out_dir={out_dir} CLOSE_MIN={close_min} FAR_MAX={far_max}")
    if not _HAS_SBERT:
        print(f"[FATAL] sentence_transformers が読み込めません: {_SBERT_IMPORT_ERR}")
        return

    model = SentenceTransformer(args.sbert_model)
    bleurt = _load_bleurt(args.bleurt_model_path) if args.bleurt else None

    files = sorted(glob.glob(os.path.join(pool_dir, "**", "*.json"), recursive=True))
    print(f"[info] 対象ファイル数: {len(files)}")

    sum_rows, dbg_rows = [], []

    for path in files:
        name = Path(path).stem
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception as e:
            print(f"[WARN] 読込失敗: {path}: {e}")
            continue

        choices = obj.get("choices", [])
        ans_idx = obj.get("answer_index", None)
        if not isinstance(choices, list) or len(choices) < 4 or not isinstance(ans_idx, int) or ans_idx<0 or ans_idx>=len(choices):
            print(f"[WARN] フォーマット不正: {path}")
            continue

        correct = choices[ans_idx]

        # SBERT類似度（正答 vs 各選択肢）
        E = model.encode([correct] + choices, convert_to_tensor=True)
        sims = util.cos_sim(E[0], E[1:]).cpu().numpy()[0].tolist()

        # BLEURT（任意）
        bleurt_scores = None
        if bleurt is not None:
            try:
                bleurt_scores = bleurt.score(references=[correct]*len(choices), candidates=choices)
            except Exception as e:
                print(f"[WARN] BLEURT計算失敗: {name}: {e}")

        # プレースホルダ検出
        def is_placeholder(s: str) -> bool:
            s = (s or "").strip()
            return (s.startswith("その他の選択肢") or s.startswith("ダミー選択肢"))

        # 集計（正答以外のDistractorのみ対象の指標も計算）
        distractor_idx = [i for i in range(len(choices)) if i != ans_idx]
        sims_dist = [sims[i] for i in range(len(sims)) if i != ans_idx]
        placeholders = sum(1 for i in distractor_idx if is_placeholder(choices[i]))

        # 近/中/遠の分布
        near_cnt = sum(1 for s in sims_dist if s >= args.close_min)
        mid_cnt  = sum(1 for s in sims_dist if (s > args.far_max and s < args.close_min))
        far_cnt  = sum(1 for s in sims_dist if s <= args.far_max)

        # 近すぎ/同義っぽさ指標
        near_too_close = sum(1 for s in sims_dist if s >= 0.90)
        # 語彙被り（Jaccard）最大値（参考値）
        try:
            jmax = max([jaccard(correct, choices[i]) for i in distractor_idx]) if distractor_idx else 0.0
        except Exception:
            jmax = 0.0

        sum_rows.append({
            "file": os.path.relpath(path, pool_dir),
            "function_name": name,
            "num_choices": len(choices),
            "placeholders": placeholders,
            "sim_avg_distractors": (sum(sims_dist)/len(sims_dist)) if sims_dist else None,
            "sim_max_distractors": max(sims_dist) if sims_dist else None,
            "sim_min_distractors": min(sims_dist) if sims_dist else None,
            "near_cnt": near_cnt, "mid_cnt": mid_cnt, "far_cnt": far_cnt,
            "near_too_close_cnt": near_too_close,
            "jaccard_max": jmax,
            "CLOSE_MIN": args.close_min,
            "FAR_MAX": args.far_max,
        })

        for i, ch in enumerate(choices):
            dbg_rows.append({
                "file": os.path.relpath(path, pool_dir),
                "function_name": name,
                "choice_index": i,
                "choice_text": ch,
                "is_correct": (i == ans_idx),
                "sim_to_correct": sims[i],
                "bleurt": (None if bleurt_scores is None else float(bleurt_scores[i])),
                "placeholder": is_placeholder(ch),
            })

    # 出力
    df_sum = pd.DataFrame(sum_rows).sort_values("function_name")
    df_dbg = pd.DataFrame(dbg_rows).sort_values(["function_name","choice_index"])

    out_sum = os.path.join(out_dir, "quality_summary.csv")
    out_dbg = os.path.join(out_dir, "quality_debug.csv")
    df_sum.to_csv(out_sum, index=False, encoding="utf-8-sig")
    df_dbg.to_csv(out_dbg, index=False, encoding="utf-8-sig")

    print("[done] 品質評価 完了")
    print(f" - summary: {out_sum}")
    print(f" - debug  : {out_dbg}")

if __name__ == "__main__":
    main()
