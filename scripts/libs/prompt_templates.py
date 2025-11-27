"""7軸に基づくコメント生成のためのプロンプトテンプレート"""

def build_prompt(function_name, function_code):
    return f"""
以下のC言語関数 `{function_name}` に対して、7軸に基づく1〜2文の日本語コメントを作成してください。

【コメント軸】
1. Logical（何をする関数か）
2. Precise（どのように処理しているか）
3. Unambiguous（曖昧さを排除した説明）
4. Exhaustive（すべての戻り値・影響を網羅）
5. Troubleshooting（エラー・例外時の対処）
6. Contextualizing（この関数がどこで使われるか）
7. Condensing（一文でまとめる簡潔な記述）

【関数コード】
```c
{function_code}
【出力形式】
以下の形式でJSONとして返答してください：
{{
"Logical": "…",
"Precise": "…",
"Unambiguous": "…",
"Exhaustive": "…",
"Troubleshooting": "…",
"Contextualizing": "…",
"Condensing": "…"
}}
"""