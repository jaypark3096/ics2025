# -*- coding: utf-8 -*-
"""
prepare_v1_train.py
从 train.txt 取前 N 条化学反应（RXN SMILES）生成 v1-train.txt。
- 默认 N=10000，输入/输出文件名可在脚本顶部常量修改。
- 支持（可选）去除末尾“变化注释串”（形如 12-34;56-78）与去原子映射。
- 仅用标准库，无需安装第三方包。

用法：
    将本脚本与 train.txt 放在同一目录，直接运行：
        python prepare_v1_train.py
"""

import os
import re

# ========= 可按需修改的配置 =========
INPUT_PATH  = "train.txt"     # 输入文件
OUTPUT_PATH = "v1-train.txt"  # 输出文件
MAX_LINES   = 10_000          # 取前多少行

# 可选清洗项（默认不改动原始字符，仅导出原样子集）
STRIP_TAIL_ANNOTATION = False   # 去掉行尾的 “12-34;56-78;...” 注释
REMOVE_ATOM_MAPPING   = False   # 去掉原子映射，如 [C:12] -> [C]
# ==================================

# 尾部变化注释的正则（整段由 "a-b" 及分号分隔组成）
ANN_RE = re.compile(r"^\s*\d+-\d+(?:;\d+-\d+)*\s*$")
# 原子映射（:数字]）匹配
MAP_RE = re.compile(r":\d+\]")

def valid_rxn_line(line: str) -> bool:
    """是否为形如 'LHS >> RHS' 的反应行"""
    return ">>" in line

def split_line(line: str):
    """
    拆分一行 -> (lhs, rhs, ann)
    - 支持行尾可选的“变化注释串”（用最后一个空格与 RHS 分开）
    """
    s = line.strip()
    if ">>" not in s:
        return None, None, None
    lhs, rhs_and_maybe_ann = s.split(">>", 1)
    lhs = lhs.strip()
    rhs_and_maybe_ann = rhs_and_maybe_ann.strip()

    # 尝试把 RHS 与 尾部注释串分开（注释串匹配 ANN_RE）
    last_space = rhs_and_maybe_ann.rfind(" ")
    ann = None
    rhs = rhs_and_maybe_ann
    if last_space != -1:
        maybe_rhs = rhs_and_maybe_ann[:last_space].strip()
        maybe_ann = rhs_and_maybe_ann[last_space + 1:].strip()
        if ANN_RE.match(maybe_ann):
            rhs = maybe_rhs
            ann = maybe_ann
    return lhs, rhs, ann

def strip_tail_annotation(lhs: str, rhs: str, ann: str):
    """根据配置，决定是否移除尾部注释串；默认保留"""
    if STRIP_TAIL_ANNOTATION:
        return lhs, rhs, None
    return lhs, rhs, ann

def remove_atom_mapping(smiles: str) -> str:
    """去除 SMILES 中的原子映射 :<num>]（仅当配置打开时）"""
    if not REMOVE_ATOM_MAPPING or not smiles:
        return smiles
    return MAP_RE.sub("]", smiles)

def count_molecules(smiles_block: str) -> int:
    """通过 '.' 计数分子个数（粗略）"""
    return smiles_block.count(".") + 1 if smiles_block else 0

def has_atom_mapping(smiles_block: str) -> bool:
    return bool(MAP_RE.search(smiles_block or ""))

def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"未找到输入文件：{INPUT_PATH}\n"
            "请将 train.txt 放在脚本同目录，或修改脚本顶部 INPUT_PATH。"
        )

    total_read = 0
    total_written = 0
    invalid = 0
    with_ann = 0
    lhs_map_cnt = 0
    rhs_map_cnt = 0
    lhs_mols_acc = 0
    rhs_mols_acc = 0

    with open(INPUT_PATH, "r", encoding="utf-8", errors="ignore") as fin, \
         open(OUTPUT_PATH, "w", encoding="utf-8") as fout:

        for line in fin:
            if total_written >= MAX_LINES:
                break
            total_read += 1
            line = line.rstrip("\n")

            if not valid_rxn_line(line):
                invalid += 1
                continue

            lhs, rhs, ann = split_line(line)
            if lhs is None or rhs is None:
                invalid += 1
                continue

            # 统计信息（基于原始 lhs/rhs）
            if has_atom_mapping(lhs): lhs_map_cnt += 1
            if has_atom_mapping(rhs): rhs_map_cnt += 1
            lhs_mols_acc += count_molecules(lhs)
            rhs_mols_acc += count_molecules(rhs)
            if ann is not None:
                with_ann += 1

            # 可选清洗
            lhs, rhs, ann = strip_tail_annotation(lhs, rhs, ann)
            lhs = remove_atom_mapping(lhs)
            rhs = remove_atom_mapping(rhs)

            # 还原为一行（如果 ann 被保留则加回去）
            out = f"{lhs} >> {rhs}"
            if ann:
                out += f" {ann}"
            fout.write(out + "\n")
            total_written += 1

    # 输出统计
    denom = max(total_written, 1)
    print("=" * 60)
    print(f"输入文件        : {INPUT_PATH}")
    print(f"输出文件        : {OUTPUT_PATH}")
    print(f"目标行数        : {MAX_LINES}")
    print(f"读取行数        : {total_read}")
    print(f"写出行数        : {total_written}")
    print(f"无效行（无 >>） : {invalid}")
    print("-" * 60)
    print(f"前 {total_written} 行统计（基于写出前的原始 LHS/RHS）：")
    print(f"  含尾部注释行数 : {with_ann}")
    print(f"  LHS 含映射行数 : {lhs_map_cnt}")
    print(f"  RHS 含映射行数 : {rhs_map_cnt}")
    print(f"  平均 LHS 分子数: {lhs_mols_acc / denom:.2f}")
    print(f"  平均 RHS 分子数: {rhs_mols_acc / denom:.2f}")
    print("-" * 60)
    print("可在脚本顶部切换：STRIP_TAIL_ANNOTATION / REMOVE_ATOM_MAPPING")

if __name__ == "__main__":
    main()
