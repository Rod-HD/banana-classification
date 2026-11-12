"""Hằng số dùng chung: đường dẫn, tên lớp, seed."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
SPLIT_DIR = ROOT / "data" / "splits"
CACHE_DIR = ROOT / "data" / "cache"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
MODELS_DIR = ROOT / "models"
CKPT_DIR = ROOT / "checkpoints"
LOG_DIR = ROOT / "logs"

CLASSES = ["bangla_kola", "champa_kola", "sabri_kola", "sagor_kola"]
CLASS_NAMES = ["Bangla Kola", "Champa Kola", "Sabri Kola", "Sagor Kola"]
SEED = 42
IMG_SIZE = 256          # kích thước chuẩn cho đặc trưng thủ công
N_FOLDS = 5

for d in (CACHE_DIR, RESULTS_DIR, FIG_DIR, MODELS_DIR, CKPT_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = SEED):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
