"""関数抽出スクリプト"""

import re
import csv

def extract_functions(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    functions = []
    i = 0
    pattern_header = re.compile(r'^\s*(\w[\w\s\*\[\]]+)\s+(\w+)\s*\((.*)\)\s*$')  # ヘッダだけ
    pattern_inline = re.compile(r'^\s*(\w[\w\s\*\[\]]+)\s+(\w+)\s*\((.*)\)\s*\{')  # 一行定義

    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith('#'):
            i += 1
            continue

        match_inline = pattern_inline.match(lines[i])
        if match_inline:
            return_type = match_inline.group(1).strip()
            func_name = match_inline.group(2).strip()
            args = match_inline.group(3).strip()
            start_line = i + 1
            brace_depth = lines[i].count('{') - lines[i].count('}')
            j = i + 1
            while j < len(lines) and brace_depth > 0:
                brace_depth += lines[j].count('{') - lines[j].count('}')
                j += 1
            end_line = j
            line_count = end_line - start_line + 1
            functions.append({
                '関数名': func_name,
                '開始行': start_line,
                '終了行': end_line,
                '行数': line_count,
                '引数': args,
                '戻り値': return_type
            })
            i = j
            continue

        match_header = pattern_header.match(lines[i])
        if match_header and i + 1 < len(lines) and lines[i+1].strip().startswith('{'):
            return_type = match_header.group(1).strip()
            func_name = match_header.group(2).strip()
            args = match_header.group(3).strip()
            start_line = i + 1
            brace_depth = 1
            j = i + 2
            while j < len(lines) and brace_depth > 0:
                brace_depth += lines[j].count('{') - lines[j].count('}')
                j += 1
            end_line = j
            line_count = end_line - start_line + 1
            functions.append({
                '関数名': func_name,
                '開始行': start_line,
                '終了行': end_line,
                '行数': line_count,
                '引数': args,
                '戻り値': return_type
            })
            i = j
            continue

        i += 1

    return functions

def save_csv(functions, path):
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        fieldnames = ['ID', '関数名', '開始行', '終了行', '行数', '引数', '戻り値']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(functions, start=1):
            row_with_id = {'ID': f'F{idx:03d}', **row}
            writer.writerow(row_with_id)

if __name__ == '__main__':
    input_path = 'data/original/2048.c'
    output_path = 'data/csv/functions.csv'
    funcs = extract_functions(input_path)
    save_csv(funcs, output_path)
    print(f'{len(funcs)} 関数を抽出し、CSVに保存しました')
