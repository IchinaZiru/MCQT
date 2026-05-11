# D-MCQA（llm_mcqt）実行・集計 手順書（後輩引き継ぎ用）

このドキュメントは、**加工済みデータセット（variants JSONL）**から **llm_mcqt（D-MCQA）だけ**を取り出して、
ローカルLLM（Ollama）に解かせ、**正答率（accuracy）**を出すための手順書です。

---

## 1. 目的

- 入力データ：`variant` が混在した **variants JSONL**
  - 例：`data/json/jcsq_mcqa_variants-gpt-oss.jsonl`
  - 形式（1行1レコード）：
    ```json
    {"qid": 0, "variant": "llm_mcqt", "question": "...", "choices": ["...","...","...","..."], "answer_index": 1}
    ```
- 実行：`variant == "llm_mcqt"` の行だけを抽出して **Ollamaに解かせる**
- 出力：結果CSV（解答ログ）
- 集計：CSVから **accuracy** を計算して表示

---

## 2. 必要ファイル（最小セット）

### データ
- `data/json/jcsq_mcqa_variants-gpt-oss.jsonl`
  - orig/sciq/mcqt/llm_mcqt が混在していてOK（スクリプト側で llm_mcqt のみ抽出）

### スクリプト
- `scripts/jcsq/run_mcqa_ollama_llm_mcqt_only.py`  … **解かせる（llm_mcqtだけ）**
- `scripts/jcsq/summarize_mcqa_results.py`         … **正答率を出す**

---

## 3. 前提（環境）

### Ollama
- Ollama がインストール済み・起動済み
- モデルがあること（例：`gpt-oss:20b`）

確認：
```powershell
ollama list
```

### Python
- Python 3.9+ 推奨
- 依存：`requests`（入ってなければ `pip install requests`）

---

## 4. フォルダ構成（推奨）

プロジェクトルートで以下の構成になっていること：
```
<PROJECT_ROOT>/
  scripts/jcsq/run_mcqa_ollama_llm_mcqt_only.py
  scripts/jcsq/summarize_mcqa_results.py
  data/json/jcsq_mcqa_variants-gpt-oss.jsonl
  outputs_jcsq/mcqa_results/   （無ければ自動作成されます）
```

---

## 5. 実行方法（llm_mcqtだけ解かせる）

### 5.1 1モデルで実行（推奨：まず動作確認）
```powershell
python scripts\jcsq\run_mcqa_ollama_llm_mcqt_only.py `
  --model gpt-oss:20b `
  --input data\json\jcsq_mcqa_variants-gpt-oss.jsonl `
  --variant llm_mcqt `
  --resume
```

#### 実行ログの見方
例：
- `TOTAL(in)`：入力JSONLの総行数（例：35756 = 8939問×4variant）
- `TOTAL(var)`：llm_mcqt行数（例：8939）
- `processed ... (answered=..., acc=...)`：途中経過の正答率

#### 出力ファイル（CSV）
- `outputs_jcsq/mcqa_results/ollama_<SAFE_MODEL>_llm_mcqt.csv`
  - 例：`outputs_jcsq/mcqa_results/ollama_gpt-oss_20b_llm_mcqt.csv`

---

## 6. 正答率を出す（集計）

```powershell
python scripts\jcsq\summarize_mcqa_results.py `
  --csv outputs_jcsq\mcqa_results\ollama_gpt-oss_20b_llm_mcqt.csv
```

- `llm_mcqt` の `n` と `accuracy` が表示されます。

### accuracy の定義
- **accuracy = 正解数 / answered数**
- answered数は「A/B/C/D を抽出できた行」だけ（抽出できなかった行は分母に入りません）

---

## 7. 途中停止した場合（resume）

長時間実験では途中で止まることがあります。その場合は **同じコマンドを再実行**してください。

- `--resume` が有効なとき、**既にCSVにあるqidはスキップ**して続きから進みます。

---

## 8. 複数モデルで回す（PowerShell例）

```powershell
$models = @(
  "gpt-oss:20b",
  "deepseek-r1:32b",
  "qwen2.5:14b",
  "gemma3:27b"
)

foreach ($m in $models) {
  python scripts\jcsq\run_mcqa_ollama_llm_mcqt_only.py --model $m --input data\json\jcsq_mcqa_variants-gpt-oss.jsonl --variant llm_mcqt --resume

  $safe = ($m -replace "[:/]", "_")
  python scripts\jcsq\summarize_mcqa_results.py --csv ("outputs_jcsq\mcqa_results\ollama_" + $safe + "_llm_mcqt.csv")
}
```

---

## 9. よくあるエラーと対処

### (1) Ollamaに繋がらない
- エラー例：connection refused
- 対処：Ollamaを起動、または `http://localhost:11434` が正しいか確認

確認：
```powershell
curl http://localhost:11434/api/tags
```

### (2) `requests` が無い
- エラー例：`ModuleNotFoundError: requests`
- 対処：
```powershell
pip install requests
```

### (3) `TOTAL(var)` が 0 になる
- 原因：入力JSONLに `variant == "llm_mcqt"` が存在しない、またはスペル違い
- 対処：JSONLを確認（`"llm_mcqt"` になっているか）

---

## 10. 任意：llm_mcqtだけの専用データセットを作る

混在が気になる場合、llm_mcqtだけのJSONLを作れます（任意）。

```powershell
python -c "import json; src=r'data\json\jcsq_mcqa_variants-gpt-oss.jsonl'; dst=r'data\json\jcsq_mcqa_variants-llm_mcqt_only.jsonl'; n=0; 
f=open(src,'r',encoding='utf-8'); g=open(dst,'w',encoding='utf-8');
for line in f:
  o=json.loads(line)
  if o.get('variant')=='llm_mcqt':
    g.write(json.dumps(o,ensure_ascii=False)+'\n'); n+=1
f.close(); g.close(); print('saved',n,'to',dst)"
```

この場合、実行コマンドの `--input` を差し替えるだけです。

---

以上。
