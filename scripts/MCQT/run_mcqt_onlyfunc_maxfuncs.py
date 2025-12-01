
"""
各関数 + 付与コメントごとMCQTを回すスクリプト

usage : 
python scripts/MCQT/run_mcqt_onlyfunc.py --model gpt-4o `
  --pool_root 選択肢参照先 e.g. data/mcqt/choices/guide `
  --levels 回すレベル指定 e.g. 0,7 `
  --code_pattern 検証対象コード参照先 e.g. "data/levels/L{level}/2048_L{level}.c" `
  --out_root 回収した回答保存先 e.g. outputs/mcqt_results `
  --sleep 2.5 `
  --max_retries 回答リトライ上限 e.g. 6 `
  --base_sleep sleep時間指定 e.g. 1.0 `
  --mask_name_for_levels 関数名マスクレベル e.g. 0,1 `
  --dump_prompts プロンプトデバック保存先 e.g. data/debug/prompts `

"""
import os, re, csv, json, time, random, argparse, requests, datetime, hashlib
from pathlib import Path
from typing import List, Tuple, Dict, Set, Optional
from openai import OpenAI, RateLimitError, APIError, APIConnectionError, APITimeoutError, OpenAIError
import anthropic
from dotenv import load_dotenv
load_dotenv()

DEFAULT_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_LLAMACPP_HOST = os.getenv("LLAMACPP_HOST", "http://localhost:8080")

class OllamaClient:
    def __init__(self, host): self.host = host
    def ask(self, model, prompt):
        r = requests.post(f"{self.host}/api/generate", json={"model": model, "prompt": prompt, "stream": False})
        r.raise_for_status()
        return r.json().get("response","").strip()

class LlamacppClient:
    def __init__(self, host): self.host = host
    def ask(self, prompt, n_predict=64):
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
    return ("openai", model, OpenAI(api_key=os.getenv("OPENAI_API_KEY")), None, None, None)

def _sanitize_filename(name: str) -> str:
    s = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in name)
    return s[:120]

def _dump_prompt(base_dir: Path, level: int, func: str, prompt: str) -> Path:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_func = _sanitize_filename(func)
    out_dir = base_dir / f"L{level}"
    out_dir.mkdir(parents=True, exist_ok=True)
    h = hashlib.md5(prompt.encode("utf-8")).hexdigest()[:8]
    path = out_dir / f"{ts}_{safe_func}_{h}.txt"
    path.write_text(prompt, encoding="utf-8")
    return path

def parse_levels(s: str) -> List[int]:
    s = s.strip()
    if "-" in s:
        a,b = s.split("-",1); return list(range(int(a), int(b)+1))
    return [int(x) for x in s.split(",") if x.strip()]

def parse_level_list(s: str) -> Set[int]:
    s = s.strip()
    return set(int(x) for x in s.split(",") if x.strip()!="")

def list_pools_for_level(root: Path, level: int) -> List[Path]:
    lvl = root / f"L{level}"
    return sorted(lvl.glob("*.json")) if lvl.exists() else sorted(root.glob("*.json"))

def load_pool(p: Path) -> Dict:
    with p.open(encoding="utf-8") as f:
        return json.load(f)

def find_func_span_auto(code_text: str, func_name: str) -> Tuple[int, int]:
    sig = rf"""
        (?P<sig>
            (?:[A-Za-z_]\w*[\s\*]+)*
            {re.escape(func_name)}
            \s*\( [^;{{}}]* \)
            \s*\{{                     
        )
    """
    m = re.search(sig, code_text, re.VERBOSE | re.MULTILINE)
    if not m: return (0, 0)
    start_idx = m.start("sig")
    start_line = code_text[:start_idx].count("\n") + 1
    i = m.end("sig") - 1
    depth = 0
    n = len(code_text)
    while i < n:
        ch = code_text[i]
        if ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end_idx = i
                break
        if ch == "/":
            if i+1 < n and code_text[i+1] == "/":
                j = code_text.find("\n", i)
                if j == -1: j = n-1
                i = j
        if ch == '"':
            j = i + 1
            while j < n:
                if code_text[j] == '"' and code_text[j-1] != "\\":
                    i = j
                    break
                j += 1
        i += 1
    else:
        end_idx = min(len(code_text)-1, m.end("sig") + 2000)
    end_line = code_text[:end_idx].count("\n") + 1
    return (start_line, end_line)

def extract_leading_comment_block(lines: List[str], start_line: int) -> int:
    i = start_line - 2
    begin = start_line
    in_block = False
    while i >= 0:
        line = lines[i].rstrip()
        if in_block:
            begin = i + 1
            if "/*" in line: in_block = False
            i -= 1; continue
        if not line.strip(): begin = i + 1; i -= 1; continue
        if line.lstrip().startswith("//"): begin = i + 1; i -= 1; continue
        if "*/" in line:
            in_block = True
            begin = i + 1; i -= 1; continue
        break
    return begin

def slice_func_with_comments(code_text: str, func_name: str, pad_after: int = 0) -> str:
    lines = code_text.splitlines()
    n = len(lines)
    s, e = find_func_span_auto(code_text, func_name)
    if s == 0:
        m = re.search(rf'\b{re.escape(func_name)}\b\s*\(', code_text)
        if not m: return ""
        s = code_text[:m.start()].count("\n") + 1
        e = min(n, s + 60)
    begin = extract_leading_comment_block(lines, s)
    end = min(n, e + max(0, pad_after))
    return "\n".join(lines[begin-1:end])

def _stable_alias(func_name: str, seed: str) -> str:
    h = hashlib.md5((seed + "::" + func_name).encode("utf-8")).hexdigest()[:6]
    return f"FUNC_{h}"

def mask_function_name_in_code(code: str, func_name: str, seed: str) -> tuple[str, str]:
    alias = _stable_alias(func_name, seed)
    pat = re.compile(rf'\b{re.escape(func_name)}\b')
    return pat.sub(alias, code), alias

def mask_funcname_in_question(question: str, func_name: str) -> str:
    q = re.sub(rf'`{re.escape(func_name)}`', "この関数", question)
    q = re.sub(rf'"{re.escape(func_name)}"', "この関数", q)
    q = re.sub(rf"\b{re.escape(func_name)}\b", "この関数", q)
    return q

def drop_signature_line(snippet: str) -> str:
    lines = snippet.splitlines()
    out = []
    dropped = False
    for ln in lines:
        if not dropped and "(" in ln and "{" in ln:
            dropped = True
            continue
        out.append(ln)
    return "\n".join(out) if dropped else ("\n".join(lines[1:]) if lines else "")

def drop_includes_and_defines(code: str) -> str:
    out = []
    for ln in code.splitlines():
        s = ln.strip()
        if s.startswith("#include") or s.startswith("#define"):
            continue
        out.append(ln)
    return "\n".join(out)

KW = set("""if else for while do switch case default return break continue
sizeof struct typedef enum union goto const volatile static inline
void char short int long float double signed unsigned bool true false
uint8_t uint16_t uint32_t size_t FILE NULL""".split())

def anonymize_identifiers(code: str) -> str:
    tokens = re.findall(r'[A-Za-z_]\w+|\W', code, flags=0)
    id_map = {}
    vcnt = 1
    out = []
    for t in tokens:
        if re.match(r'[A-Za-z_]\w+$', t):
            if t in KW:
                out.append(t)
            else:
                if t not in id_map:
                    id_map[t] = f"VAR_{vcnt}"
                    vcnt += 1
                out.append(id_map[t])
        else:
            out.append(t)
    return "".join(out)

def strip_literals(code: str) -> str:
    code = re.sub(r'"(?:\\.|[^"\\])*"', '"STR"', code)
    code = re.sub(r"'(?:\\.|[^'\\])'", "'C'", code)
    code = re.sub(r'\b(?:0x[0-9a-fA-F]+|\d+)\b', 'NUM', code)
    return code

def keep_code_skeleton_only(code: str) -> str:
    return re.sub(r'[A-Za-z0-9_]+', '', code)

def make_prompt(question: str, choices: list, code_snippet: Optional[str] = None, style: str = "base") -> str:
    choice_lines = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
    if style == "anti_name_bias":
        header = (
            "次の四択問題に回答してください。出力は A / B / C / D の**1文字のみ**。\n"
            "【重要】関数名や識別子の名称から推測せず、提示されたコード/コメントの内容**のみ**に基づいて判断してください。\n"
            "名称が意味を示唆しても無視してください。不確定な内容は根拠にしないでください。\n\n"
        )
        code_block = f"```c\n{code_snippet}\n```\n\n" if code_snippet else ""
        guard = "【厳守】コードで明示されない機能は行わないと見なしてください。\n\n" if code_snippet else ""
        return header + code_block + guard + f"問題: {question}\n{choice_lines}"
    else:
        header = (
            "以下の四択問題に答えてください。\n"
            "【重要】出力は必ず以下の中から1文字だけ選んでください：\n"
            "A / B / C / D\n\n"
            "絶対に文章や説明は書かないでください。\n"
            "もしわからない場合でも必ず A〜D の中から最も適切なものを選んでください。\n\n"
        )
        if code_snippet:
            return header + f"```c\n{code_snippet}\n```\n\n問題: {question}\n{choice_lines}"
        else:
            return header + f"問題: {question}\n{choice_lines}"

def extract_answer(raw: str) -> str:
    m = re.search(r"\b([ABCD])\b", raw, re.IGNORECASE)
    if m: return m.group(1).upper()
    n = re.search(r"\b([1-4])\b", raw)
    if n: return {"1":"A","2":"B","3":"C","4":"D"}.get(n.group(1),"?")
    return "?"

def call_openai_with_retry(client, model, prompt, max_tokens=4, max_retries=6, base_sleep=0.5):
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role":"user","content":prompt}],
                temperature=0,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except (RateLimitError, APIConnectionError, APITimeoutError, APIError, OpenAIError) as e:
            wait = base_sleep * (2 ** attempt) * (1.0 + 0.25 * random.random())
            print(f"[ERROR] {type(e).__name__}: {getattr(e, 'message', str(e))}")
            print(f"[RETRY] wait {wait:.2f}s (attempt {attempt+1})")
            time.sleep(wait)
    return "?"

def load_gold_label(root: Path, func: str, choices: list) -> str:
    """Return gold as A/B/C/D.
    Supports schemas:
      - {"answer":"A"} or label/gold/correct_label
      - {"answer_index":0..3} or 1..4
      - {"correct":"<choice text>"}  (matches choices[] to map into A-D)
    """
    p = (root / f"{func}.json")
    if not p.exists():
        return ""
    try:
        with p.open("r", encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        return ""
    # 1) Direct label
    for key in ("answer", "label", "gold", "correct_label"):
        v = d.get(key)
        if isinstance(v, str) and v.strip().upper() in ("A","B","C","D"):
            return v.strip().upper()
    # 2) Index -> label
    for key in ("answer_index","index","gold_index"):
        if key in d:
            try:
                idx = int(d[key])
                if idx in (0,1,2,3):
                    return "ABCD"[idx]
                if idx in (1,2,3,4):
                    return "ABCD"[idx-1]
            except Exception:
                pass
    # 3) Text match -> label
    for key in ("correct","gold_text","text"):
        v = d.get(key)
        if isinstance(v, str):
            try:
                i = choices.index(v)
                return "ABCD"[i] if 0 <= i < 4 else ""
            except ValueError:
                # also try trimmed comparison
                v2 = v.strip()
                trimmed = [c.strip() if isinstance(c,str) else c for c in choices]
                try:
                    i = trimmed.index(v2)
                    return "ABCD"[i] if 0 <= i < 4 else ""
                except ValueError:
                    continue
    return ""
def run_level(mode, model_name, openai_client, ollama_client, llamacpp_client, claude_client,
              level: int, pool_root: Path, out_dir: Path,
              code_text: str, pad_after: int, sleep: float,
              prompt_style: str,
              name_only_lv: Set[int], mask_name_lv: Set[int], sigless_lv: Set[int],
              mask_q_lv: Set[int], drop_inc_lv: Set[int], anon_id_lv: Set[int],
              strip_lit_lv: Set[int], skeleton_lv: Set[int], mask_seed: str,
              dump_dir: Path = None, print_head: int = 0,
              max_retries: int = 6, base_sleep: float = 0.5,
              max_funcs: int = 0, func_pick: str = 'head', func_seed: str = '',
              start_func: str = '', correct_root: Path = Path('data/mcqt/correct_choices')):

    pools = list_pools_for_level(pool_root, level)
    # --- optional: start from a specific function name (filename stem contains) ---
    # Applies only when func_pick == 'head'（順次処理の先頭をずらす）
    if start_func and func_pick == 'head':
        key = start_func.lower()
        start_idx = 0
        for i, pp in enumerate(pools):
            stem = pp.stem.lower()
            if key in stem:
                start_idx = i
                break
        pools = pools[start_idx:]
    # --- limit number of pools (functions) per level if requested ---
    if max_funcs and max_funcs > 0:
        if func_pick == 'head':
            pools = pools[:max_funcs]
        elif func_pick == 'random':
            import random
            rnd = random.Random()
            if func_seed:
                rnd.seed(func_seed)
            rnd.shuffle(pools)
            pools = pools[:max_funcs]
        else:
            pools = pools[:max_funcs]
    # ---------------------------------------------------------------
    if not pools:
        print(f"[WARN] L{level}: プールが見つかりません")
        return None
    out_csv = out_dir / f"L{level}.csv"
    rows = []
    for p in pools:
        data = load_pool(p)
        q = data.get("question","").strip()
        choices = data.get("choices", [])
        if len(choices) != 4:
            print(f"[L{level}] WARN: skip non-4 choices -> {p.name}")
            continue
        func = p.stem
        snippet = slice_func_with_comments(code_text, func, pad_after=pad_after)

        q_text = q
        if level in mask_q_lv:
            q_text = mask_funcname_in_question(q_text, func)

        if level in name_only_lv:
            code_block = None
        else:
            code_block = snippet if snippet else None
            if code_block:
                if level in drop_inc_lv:
                    code_block = drop_includes_and_defines(code_block)
                if level in mask_name_lv:
                    code_block, _ = mask_function_name_in_code(code_block, func, seed=mask_seed)
                if level in sigless_lv:
                    code_block = drop_signature_line(code_block)
                if level in anon_id_lv:
                    code_block = anonymize_identifiers(code_block)
                if level in strip_lit_lv:
                    code_block = strip_literals(code_block)
                if level in skeleton_lv:
                    code_block = keep_code_skeleton_only(code_block)

        prompt = make_prompt(q_text, choices, code_snippet=code_block, style=prompt_style)

        if dump_dir:
            saved = _dump_prompt(dump_dir, level, func, prompt)
            print(f"[L{level}] prompt saved: {saved.as_posix()}")
        if print_head > 0:
            head = prompt[:print_head].replace("\n", "\\n")
            print(f"[L{level}] {func} prompt head: {head}")

        if mode == "openai":
            raw = call_openai_with_retry(openai_client, model_name, prompt,
                                         max_tokens=4, max_retries=max_retries, base_sleep=base_sleep)
        elif mode == "claude":
            resp = claude_client.messages.create(
                model=model_name, max_tokens=10,
                messages=[{"role":"user","content":prompt}]
            )
            raw = resp.content[0].text.strip()
        elif mode == "ollama":
            raw = ollama_client.ask(model_name, prompt)
        elif mode == "llamacpp":
            raw = llamacpp_client.ask(prompt)
        else:
            raw = "?"
        ans = extract_answer(raw)
        idx_map = {"A":0,"B":1,"C":2,"D":3}
        pred_text = ""
        if ans in idx_map and 0 <= idx_map[ans] < len(choices):
            pred_text = choices[idx_map[ans]]
        gold = load_gold_label(correct_root, func, choices)
        print(f"[L{level}] {func}: {ans} | correct = {gold}")
        rows.append({"function": func, "answer": ans, "pred_text": pred_text, "gold": gold})
        if mode in ("openai","claude"):
            time.sleep(sleep)

    out_dir.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["function","answer","pred_text","gold"])
        writer.writeheader(); writer.writerows(rows)
    print(f"[DONE] L{level} -> {out_csv}")
    return out_csv

def main():
    ap = argparse.ArgumentParser(description="AutoSpan+Mask v2: 関数抽出→プロンプト送信（名前バイアス対策＋再試行制御）")
    ap.add_argument("--model", required=True)
    ap.add_argument("--pool_root", default="data/mcqt/choices/guide")
    ap.add_argument("--correct_root", default="data/mcqt/correct_choices")
    ap.add_argument("--levels", default="0-7")
    ap.add_argument("--out_root", default="outputs/mcqa_results")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--code_pattern", help="レベル別コードパターン 例: data/levels/L{level}/xxx_L{level}.c")
    group.add_argument("--code_file", help="共通コードファイル")
    ap.add_argument("--pad_after", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=1.0)
    # --- NEW: limit how many functions (pool JSONs) to run per level ---
    ap.add_argument("--max_funcs", type=int, default=0,
                    help="各レベルで処理する関数（プールJSON）数の上限。0は無制限")
    ap.add_argument("--func_pick", choices=["head","random"], default="head",
                    help="--max_funcs 適用時の選び方：先頭から(head)またはランダム(random)")
    ap.add_argument("--func_seed", default="",
                    help="--func_pick=random のときの乱択シード（任意の文字列）")
    ap.add_argument("--start_func", default="",
                    help="この文字列を含む関数名（プールJSONのstem）から順に処理（func_pick=head時のみ有効）")

    ap.add_argument("--dump_prompts", default="", help="プロンプトを保存するディレクトリ（例: debug/prompts）")
    ap.add_argument("--print_prompt_head", type=int, default=0, help="各プロンプトの先頭N文字を標準出力に表示（0で無効）")

    ap.add_argument("--prompt_style", choices=["base","anti_name_bias"], default="base")

    ap.add_argument("--name_only_levels", default="", help="指定レベルは関数名のみ提示（コードは送らない）。例: 0,1")
    ap.add_argument("--mask_name_for_levels", default="", help="指定レベルはコード内の関数名をマスク。例: 0")
    ap.add_argument("--signatureless_levels", default="", help="指定レベルはシグネチャ1行を落として送る。例: 0")
    ap.add_argument("--mask_question_func_names", default="", help="指定レベルは設問内の関数名を『この関数』に置換。例: 0")
    ap.add_argument("--drop_includes_defines", default="", help="指定レベルは #include/#define を削除。例: 0")
    ap.add_argument("--anonymize_identifiers", default="", help="指定レベルは識別子を VAR_n へ正規化。例: 0")
    ap.add_argument("--strip_literals", default="", help="指定レベルは数値/文字列リテラルをマスク。例: 0")
    ap.add_argument("--keep_code_skeleton_only", default="", help="指定レベルは英数字を除去し構文骨格のみ残す。例: 0")
    ap.add_argument("--mask_seed", default="maskv1", help="関数名マスクの安定化シード（同名→同エイリアス）")

    # 追加: 再試行制御
    ap.add_argument("--max_retries", type=int, default=6)
    ap.add_argument("--base_sleep", type=float, default=0.5)

    args = ap.parse_args()

    mode, model_name, openai_client, ollama_client, llamacpp_client, claude_client = build_clients(args.model)
    levels = parse_levels(args.levels)

    # sanitize model dir (改行や空白/記号を置換)
    model_dir = re.sub(r'[^A-Za-z0-9._-]+', '_', args.model.strip())
    out_dir = Path(args.out_root) / model_dir

    name_only_lv = parse_level_list(args.name_only_levels)
    mask_name_lv = parse_level_list(args.mask_name_for_levels)
    sigless_lv   = parse_level_list(args.signatureless_levels)
    mask_q_lv    = parse_level_list(args.mask_question_func_names)
    drop_inc_lv  = parse_level_list(args.drop_includes_defines)
    anon_id_lv   = parse_level_list(args.anonymize_identifiers)
    strip_lit_lv = parse_level_list(args.strip_literals)
    skeleton_lv  = parse_level_list(args.keep_code_skeleton_only)
    mask_seed    = args.mask_seed

    for lv in levels:
        code_path = Path(args.code_pattern.format(level=lv)) if args.code_pattern else Path(args.code_file)
        if not code_path.exists():
            print(f"[L{lv}] ERROR: コードが見つかりません: {code_path}")
            continue
        code_text = code_path.read_text(encoding="utf-8")
        dump_dir = Path(args.dump_prompts) if args.dump_prompts else None

        run_level(mode, model_name, openai_client, ollama_client, llamacpp_client, claude_client,
                  lv, Path(args.pool_root), out_dir, code_text, args.pad_after, args.sleep,
                  prompt_style=args.prompt_style,
                  name_only_lv=name_only_lv, mask_name_lv=mask_name_lv, sigless_lv=sigless_lv,
                  mask_q_lv=mask_q_lv, drop_inc_lv=drop_inc_lv, anon_id_lv=anon_id_lv,
                  strip_lit_lv=strip_lit_lv, skeleton_lv=skeleton_lv, mask_seed=mask_seed,
                  dump_dir=dump_dir, print_head=args.print_prompt_head,
                  max_retries=args.max_retries, base_sleep=args.base_sleep,
                  max_funcs=args.max_funcs, func_pick=args.func_pick, func_seed=args.func_seed,
                  start_func=args.start_func, correct_root=Path(args.correct_root))

if __name__ == "__main__":
    main()


