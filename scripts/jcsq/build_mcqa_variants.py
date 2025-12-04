from pathlib import Path
import json
import random

RANDOM_SEED = 1234
random.seed(RANDOM_SEED)

ROOT_DIR = Path(__file__).resolve().parents[2]

ALL_DISTRACTORS_JSONL = ROOT_DIR / "data" / "json" / "jcommonsenseqa_with_all_distractors.jsonl"
LLM_MCQT_JSONL        = ROOT_DIR / "data" / "json" / "jcsq_mcqt_llm_mcqt.jsonl"

OUT_VARIANTS_JSONL    = ROOT_DIR / "data" / "json" / "jcsq_mcqa_variants.jsonl"


def load_jsonl(path: Path):
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def build_mcqa_instance(qid: int, variant: str, question: str, correct: str, distractors: list[str]):
    """
    4択問題を1件生成する:
    - choices: 正解 + 誤答3つ（計4つ）をシャッフル
    - answer_index: 正解が choices の何番目か
    """
    if len(distractors) < 3:
        return None  # スキップ

    # 正解＋誤答3つ
    base_choices = [correct] + distractors[:3]

    # 再現性のあるシャッフル
    rnd = random.Random(f"{RANDOM_SEED}_{qid}_{variant}")
    choices = base_choices[:]
    rnd.shuffle(choices)

    answer_index = choices.index(correct)

    return {
        "qid": qid,
        "variant": variant,
        "question": question,
        "choices": choices,
        "answer_index": answer_index,
    }


def main():
    # 1. 全問題 + orig/sciq/mcqt 誤答セット
    all_ex = load_jsonl(ALL_DISTRACTORS_JSONL)
    print(f"loaded {len(all_ex)} questions with original/sciq/mcqt from {ALL_DISTRACTORS_JSONL}")

    # 2. 500問の LLM-MCQT を id で辞書化
    llm_ex_list = load_jsonl(LLM_MCQT_JSONL)
    print(f"loaded {len(llm_ex_list)} LLM-MCQT questions from {LLM_MCQT_JSONL}")
    llm_by_id = {ex["id"]: ex for ex in llm_ex_list}

    out_instances = []

    for ex in all_ex:
        qid = ex["id"]
        q   = ex["question"]
        a   = ex["correct"]

        # --- orig ---
        inst = build_mcqa_instance(
            qid, "orig", q, a, ex["distractors_original4"]
        )
        if inst:
            out_instances.append(inst)

        # --- sciq (SciQ風RF) ---
        if "distractors_sciq_like" in ex:
            inst = build_mcqa_instance(
                qid, "sciq", q, a, ex["distractors_sciq_like"]
            )
            if inst:
                out_instances.append(inst)

        # --- mcqt (辞書版MCQT) ---
        if "distractors_mcqt" in ex:
            inst = build_mcqa_instance(
                qid, "mcqt", q, a, ex["distractors_mcqt"]
            )
            if inst:
                out_instances.append(inst)

        # --- llm_mcqt (本来のMCQT: LLM生成版) ---
        if qid in llm_by_id:
            llm_ex = llm_by_id[qid]
            if "distractors_mcqt_llm" in llm_ex:
                inst = build_mcqa_instance(
                    qid, "llm_mcqt", q, a, llm_ex["distractors_mcqt_llm"]
                )
                if inst:
                    out_instances.append(inst)

    # 保存
    OUT_VARIANTS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUT_VARIANTS_JSONL.open("w", encoding="utf-8") as f:
        for inst in out_instances:
            f.write(json.dumps(inst, ensure_ascii=False) + "\n")

    # variantごとの件数確認
    counts = {}
    for inst in out_instances:
        counts[inst["variant"]] = counts.get(inst["variant"], 0) + 1
    print("instances per variant:", counts)
    print(f"saved {len(out_instances)} MCQA instances to {OUT_VARIANTS_JSONL}")


if __name__ == "__main__":
    main()
