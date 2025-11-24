"""生成済みコメントテンプレートをレベル別でソースコードに挿入するスクリプト
usage: python insert_comments.py --level N
  --level N : 0..7 の整数。0はコメント削除、1〜7は7軸コメントをN個採用して挿入
"""
import os
import csv
import argparse
import pandas as pd

SOURCE_FILE = "data/original/2048.c"
FUNCTIONS_CSV = "data/csv/functions.csv"
TEMPLATES_CSV = "data/csv/comment_templates.csv"

# 7軸の出力順（L1〜L7で先頭からn個を使う）
AXIS_KEYS = [
    "Logical",
    "Precise",
    "Unambiguous",
    "Exhaustive",
    "Troubleshooting",
    "Contextualizing",
    "Condensing",
]

COMMENT_BLOCK_FMT = """/* @doc @function {name} @level L{level}
{body}
*/
"""

def strip_comments_c(text: str) -> str:
    """
    C/C++ 風コメントを安全に除去する。文字列/文字リテラルは保持。
    - // ... \n
    - /* ... */
    ※ ネストは想定しない（C準拠）。エスケープにも対応。
    """
    res = []
    i, n = 0, len(text)
    NORMAL, LINE, BLOCK, STR, CHR = range(5)
    st = NORMAL
    esc = False

    while i < n:
        c = text[i]
        c2 = text[i+1] if i+1 < n else ""

        if st == NORMAL:
            if c == "/" and c2 == "/":
                st = LINE
                i += 2
                continue
            if c == "/" and c2 == "*":
                st = BLOCK
                i += 2
                continue
            if c == '"':
                st = STR
                res.append(c)
                i += 1
                esc = False
                continue
            if c == "'":
                st = CHR
                res.append(c)
                i += 1
                esc = False
                continue
            res.append(c)
            i += 1
            continue

        if st == LINE:
            if c == "\n":
                res.append(c)
                st = NORMAL
            i += 1
            continue

        if st == BLOCK:
            if c == "*" and c2 == "/":
                st = NORMAL
                i += 2
            else:
                i += 1
            continue

        if st == STR:
            res.append(c)
            if not esc and c == "\\":
                esc = True
            elif esc:
                esc = False
            elif c == '"':
                st = NORMAL
            i += 1
            continue

        if st == CHR:
            res.append(c)
            if not esc and c == "\\":
                esc = True
            elif esc:
                esc = False
            elif c == "'":
                st = NORMAL
            i += 1
            continue

    return "".join(res)

def load_templates(path: str) -> dict:
    df = pd.read_csv(path, encoding="utf-8-sig")
    # 列名のゆらぎ吸収
    if "function_name" not in df.columns and "関数名" in df.columns:
        df = df.rename(columns={"関数名": "function_name"})
    df["function_name"] = df["function_name"].astype(str)
    # 軸キーのゆらぎ（先頭小文字など）を吸収
    rename_map = {}
    for col in df.columns:
        lower = col.lower()
        if lower == "logical": rename_map[col] = "Logical"
        if lower == "precise": rename_map[col] = "Precise"
        if lower == "unambiguous": rename_map[col] = "Unambiguous"
        if lower == "exhaustive": rename_map[col] = "Exhaustive"
        if lower == "troubleshooting": rename_map[col] = "Troubleshooting"
        if lower in ("context", "contextualizing", "contextualising"):
            rename_map[col] = "Contextualizing"
        if lower in ("condense", "condensing"):
            rename_map[col] = "Condensing"
    if rename_map:
        df = df.rename(columns=rename_map)
    return df.set_index("function_name").to_dict(orient="index")

def load_functions(path: str):
    df = pd.read_csv(path, encoding="utf-8-sig")
    name_col = "関数名" if "関数名" in df.columns else "function_name"
    start_col = "開始行" if "開始行" in df.columns else "start_line"
    end_col = "終了行" if "終了行" in df.columns else "end_line"
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "name": str(r[name_col]),
            "start": int(r[start_col]),
            "end": int(r[end_col]),
        })
    return rows

def generate_comment_block(name: str, level: int, tmpl_row: dict) -> str:
    # level 個だけ先頭から採用
    keys = AXIS_KEYS[:max(0, min(level, len(AXIS_KEYS)))]
    body_lines = []
    for k in keys:
        if k in tmpl_row and isinstance(tmpl_row[k], str) and tmpl_row[k].strip():
            body_lines.append(f" * @{k.lower()} {tmpl_row[k].strip()}")
        else:
            body_lines.append(f" * @{k.lower()} (TBD)")
    body = "\n".join(body_lines) if body_lines else " * (no content)"
    return COMMENT_BLOCK_FMT.format(name=name, level=level, body=body)

def insert_comments_for_level(level: int):
    with open(SOURCE_FILE, encoding="utf-8") as f:
        src_lines = f.readlines()

    if level == 0:
        # L0: コメントを全削除して保存
        os.makedirs("data/levels/L0", exist_ok=True)
        out_path = "data/levels/L0/2048_L0.c"
        stripped = strip_comments_c("".join(src_lines))
        # ついでに余計な空白行を軽く整える（任意）
        out_text = "\n".join([line.rstrip() for line in stripped.splitlines()])
        with open(out_path, "w", encoding="utf-8") as w:
            w.write(out_text + "\n")
        print(f"[L0] 無コメント版を書き出しました: {out_path}")
        return

    # L1〜L7: 7軸コメントを関数宣言直前に挿入
    templates = load_templates(TEMPLATES_CSV)
    funcs = load_functions(FUNCTIONS_CSV)

    out_dir = f"data/levels/L{level}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/2048_L{level}.c"

    # 元コードから毎回生成（重複挿入を防ぐ）
    out = src_lines[:]
    # 行番号は 1 始まり → インデックスは -1
    # 後ろから挿入するとインデックスがずれにくい
    for fmeta in sorted(funcs, key=lambda x: x["start"], reverse=True):
        name = fmeta["name"]
        insert_at = fmeta["start"] - 1  # 関数宣言行の直前
        tmpl = templates.get(name, {})
        block = generate_comment_block(name, level, tmpl)
        out[insert_at] = block + "\n" + out[insert_at]

    with open(out_path, "w", encoding="utf-8") as w:
        w.write("".join(out))
    print(f"[L{level}] コメント付与版を書き出しました: {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, required=True, help="0..7")
    args = parser.parse_args()
    level = args.level
    if level < 0 or level > 7:
        raise SystemExit("level は 0..7 を指定してください")
    insert_comments_for_level(level)

if __name__ == "__main__":
    main()
