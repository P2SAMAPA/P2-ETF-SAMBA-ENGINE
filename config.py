# config.py — P2-ETF-SAMBA-ENGINE
# Graph-Mamba ETF Signal Engine

import os

# ── HuggingFace ────────────────────────────────────────────────────────────────
HF_TOKEN        = os.environ.get("HF_TOKEN", "")
HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO", "P2SAMAPA/p2-etf-deepm-data")
HF_MODELS_REPO  = os.environ.get("HF_MODELS_REPO",  "P2SAMAPA/p2-etf-samba-models")

# ── Data files ─────────────────────────────────────────────────────────────────
FILE_MASTER      = "data/master.parquet"

# ── Option A — Fixed Income / Alternatives ─────────────────────────────────────
FI_ETFS = [
    "TLT", "LQD", "HYG", "VNQ",
    "GLD", "SLV", "PFF", "MBB",
]
FI_BENCHMARK = "AGG"

# ── Option B — Equity Sectors ──────────────────────────────────────────────────
EQ_ETFS = [
    "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV",
    "XLI", "XLY", "XLP", "XLU", "GDX", "XME",
]
EQ_BENCHMARK = "SPY"

# ── Macro features ─────────────────────────────────────────────────────────────
MACRO_VARS = ["VIX", "T10Y2Y", "HY_SPREAD", "USD_INDEX", "DTB3"]

# ── Sequence config ────────────────────────────────────────────────────────────
LOOKBACK    = 60      # trading days of history per sample
PRED_HORIZON = 1      # predict next day's best ETF

# ── Mamba config ───────────────────────────────────────────────────────────────
D_MODEL       = 64    # Mamba model dimension
D_STATE       = 16    # SSM state dimension
D_CONV        = 4     # local conv width in Mamba
EXPAND        = 2     # expansion factor in Mamba
N_MAMBA_LAYERS = 2    # number of Mamba blocks per asset encoder

# ── Graph config ───────────────────────────────────────────────────────────────
GRAPH_HIDDEN_DIM = 64
N_ATTN_HEADS     = 2

# ── Macro encoder ──────────────────────────────────────────────────────────────
MACRO_HIDDEN_DIM = 32

# ── Training ───────────────────────────────────────────────────────────────────
TRAIN_SPLIT  = 0.70
VAL_SPLIT    = 0.15
# TEST = remaining 15%

BATCH_SIZE   = 64
MAX_EPOCHS   = 150
PATIENCE     = 15
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
DROPOUT       = 0.2

TRAIN_END    = "2024-12-31"
LIVE_START   = "2025-01-01"

# ── Shrinking windows ──────────────────────────────────────────────────────────
WINDOWS = [
    {"id": 1, "start": "2008-01-01"},
    {"id": 2, "start": "2010-01-01"},
    {"id": 3, "start": "2012-01-01"},
    {"id": 4, "start": "2014-01-01"},
    {"id": 5, "start": "2016-01-01"},
    {"id": 6, "start": "2018-01-01"},
    {"id": 7, "start": "2020-01-01"},
    {"id": 8, "start": "2022-01-01"},
]

# ── Local dirs ─────────────────────────────────────────────────────────────────
MODELS_DIR = "models"
DATA_DIR   = "data"
