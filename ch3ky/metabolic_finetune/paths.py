# -*- coding: utf-8 -*-
from __future__ import annotations

import os, sys
from pathlib import Path

# —— 基础路径（不依赖 CWD）——
PKG_DIR  = Path(__file__).resolve().parent           # .../metabolic_finetune
ROOT_DIR = PKG_DIR.parent                            # 项目根目录
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# 统一输出目录（可被其它模块复用）
OUT_DIR: Path = ROOT_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 5 折模型与预测结果目录（后续训练 / 测试脚本会用到）
CV_DIR   = OUT_DIR / "cv_models"
PRED_DIR = OUT_DIR / "predictions"
CV_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILENAME = "merged_HMDB_drugbank_reactions.csv"

# ======================
# 数据路径解析
# ======================
def resolve_csv_path(debug: bool = True) -> Path:
    """
    默认：包内 metabolic_finetune/data/merged_HMDB_drugbank_reactions.csv
    覆盖：METAB_DATA_CSV（完整文件）或 METAB_DATA_DIR（目录）
    兜底：项目根目录 data/merged_HMDB_drugbank_reactions.csv
    """
    default_csv = PKG_DIR / "data" / DATA_FILENAME
    env_csv = os.environ.get("METAB_DATA_CSV")
    env_dir = os.environ.get("METAB_DATA_DIR")

    candidates = []
    if env_csv:
        candidates.append(Path(env_csv))
    if env_dir:
        candidates.append(Path(env_dir) / DATA_FILENAME)
    candidates.append(default_csv)
    candidates.append(ROOT_DIR / "data" / DATA_FILENAME)

    for p in candidates:
        if p and p.is_file():
            if debug:
                print(f"[Data] 读取: {p}")
            return p.resolve()

    tried = "\n  - ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"找不到 {DATA_FILENAME}。请将文件放到 metabolic_finetune/data/ 下，"
        "或设置环境变量 METAB_DATA_CSV（完整文件路径）/ METAB_DATA_DIR（目录）。\n"
        "已尝试：\n  - " + tried
    )

# ======================
# 酶特征目录解析
# ======================
def resolve_enzyme_dir(debug: bool = True) -> Path:
    """
    默认：包内 metabolic_finetune/enzyme_features
    覆盖：METAB_ENZ_DIR
    兜底：项目根目录 enzyme_features/
    """
    env_dir = os.environ.get("METAB_ENZ_DIR")
    candidates = []
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.append(PKG_DIR / "enzyme_features")
    candidates.append(ROOT_DIR / "enzyme_features")

    for d in candidates:
        if d and d.is_dir():
            if debug:
                print(f"[Enzyme] 目录: {d}")
            return d.resolve()

    tried = "\n  - ".join(str(d) for d in candidates)
    raise FileNotFoundError(
        "找不到 enzyme_features 目录。请将目录放到 metabolic_finetune/enzyme_features/ 下，"
        "或设置环境变量 METAB_ENZ_DIR 指向该目录。\n已尝试：\n  - " + tried
    )

# ======================
# 共享编码器/解码器权重解析
# ======================
def resolve_shared_ckpt_paths(debug: bool = True) -> tuple[Path, Path]:
    """
    默认优先找包内：
      metabolic_finetune/shared_encoder.pt
      metabolic_finetune/shared_decoder.pt
    覆盖：METAB_SHARED_ENCODER / METAB_SHARED_DECODER
    兜底：项目根目录、outputs/、chemical_pretrain/outputs/ 等常见位置
    """
    env_enc = os.environ.get("METAB_SHARED_ENCODER")
    env_dec = os.environ.get("METAB_SHARED_DECODER")

    enc_names = ["shared_encoder.pt","shared_encoder_finetuned.pt","shared_encoder_meta.pt"]
    dec_names = ["shared_decoder.pt","shared_decoder_finetuned.pt","shared_decoder_meta.pt"]

    enc_candidates, dec_candidates = [], []

    if env_enc:
        enc_candidates.append(Path(env_enc))
    if env_dec:
        dec_candidates.append(Path(env_dec))

    enc_candidates += [PKG_DIR / n for n in enc_names]
    dec_candidates += [PKG_DIR / n for n in dec_names]

    enc_candidates += [ROOT_DIR / n for n in enc_names]
    dec_candidates += [ROOT_DIR / n for n in dec_names]

    enc_candidates += [OUT_DIR / n for n in enc_names]
    dec_candidates += [OUT_DIR / n for n in dec_names]

    chm_out = ROOT_DIR / "chemical_pretrain" / "outputs"
    enc_candidates += [chm_out / n for n in enc_names]
    dec_candidates += [chm_out / n for n in dec_names]

    enc_path = next((p.resolve() for p in enc_candidates if p.is_file()), None)
    dec_path = next((p.resolve() for p in dec_candidates if p.is_file()), None)

    if debug:
        if enc_path:
            print(f"[Shared] Encoder: {enc_path}")
        if dec_path:
            print(f"[Shared] Decoder: {dec_path}")

    if (enc_path is None) or (dec_path is None):
        tried_enc = "\n  - ".join(str(p) for p in enc_candidates[:10]) + (
            "\n  ..." if len(enc_candidates) > 10 else ""
        )
        tried_dec = "\n  - ".join(str(p) for p in dec_candidates[:10]) + (
            "\n  ..." if len(dec_candidates) > 10 else ""
        )
        raise FileNotFoundError(
            "未找到共享权重。可将权重放到 metabolic_finetune/ 目录下，或设置环境变量：\n"
            "  METAB_SHARED_ENCODER, METAB_SHARED_DECODER\n"
            f"[尝试过的 Encoder 路径示例]\n  - {tried_enc}\n"
            f"[尝试过的 Decoder 路径示例]\n  - {tried_dec}"
        )

    return enc_path, dec_path


# —— 导出供其它模块使用 —— #
CSV_PATH  = resolve_csv_path(debug=True)
ENZ_DIR   = resolve_enzyme_dir(debug=True)
ENC_PATH, DEC_PATH = resolve_shared_ckpt_paths(debug=True)
