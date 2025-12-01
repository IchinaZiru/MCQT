"""プロンプトを間接的に利用してコメントテンプレートを生成するスクリプト"""

import os
import pandas as pd
import json
import re
from openai import OpenAI
from prompt_templates import build_prompt
from dotenv import load_dotenv

load_dotenv()

# OpenAI クライアント初期化（環境変数から API キー取得）
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 入力ファイルパス
SOURCE_FILE = "data/original/2048.c"
FUNC_CSV = "data/csv/functions.csv"
OUTPUT_CSV = "data/csv/comment_templates.csv"

# 関数コード抽出元ファイルを行ごとに読み込み
with open(SOURCE_FILE, "r", encoding="utf-8") as f:
    all_lines = f.readlines()

# 関数一覧CSVを読み込み
functions_df = pd.read_csv(FUNC_CSV)

# === 日本語カラム対応 ===
columns = {col.strip(): col for col in functions_df.columns}
func_col = columns.get("関数名", "function_name")
start_col = columns.get("開始行", "start_line")
end_col = columns.get("終了行", "end_line")

# GPT出力からJSONブロックだけを抽出するユーティリティ
def extract_json_block(text):
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

# 出力行を蓄積
output_rows = []

# 各関数ごとに処理
for _, row in functions_df.iterrows():
    fname = row[func_col]
    start = int(row[start_col]) - 1
    end = int(row[end_col])
    function_code = "".join(all_lines[start:end])

    prompt = build_prompt(fname, function_code)

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        content = response.choices[0].message.content.strip()
        json_text = extract_json_block(content)
        parsed = json.loads(json_text)

        output_rows.append({"function_name": fname, **parsed})
        print(f"{fname} - コメント生成成功")

    except Exception as e:
        print(f"{fname} - エラー: {e}")
        print(f"出力内容:\n{content}\n")
        continue

# 保存
df_out = pd.DataFrame(output_rows)
df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"\nテンプレート保存完了: {OUTPUT_CSV}")
