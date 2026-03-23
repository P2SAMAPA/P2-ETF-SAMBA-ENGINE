# loader.py — Loads data from shared p2-etf-deepm-data HF dataset

import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import config as cfg


def _load_parquet(filename: str) -> pd.DataFrame:
    path = hf_hub_download(
        repo_id=cfg.HF_DATASET_REPO,
        filename=filename,
        repo_type="dataset",
        token=cfg.HF_TOKEN or None,
        force_download=True,
    )
    df = pd.read_parquet(path)
    if "Date" in df.columns:
        df = df.set_index("Date")
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df.sort_index()


def load_master() -> pd.DataFrame:
    print("[loader] Loading master dataset...")
    df = _load_parquet(cfg.FILE_MASTER)
    print(f"[loader] Master: {df.shape}, {df.index[0].date()} → {df.index[-1].date()}")
    return df


def get_option_data(option: str, master: pd.DataFrame) -> dict:
    tickers   = cfg.FI_ETFS   if option == "A" else cfg.EQ_ETFS
    benchmark = cfg.FI_BENCHMARK if option == "A" else cfg.EQ_BENCHMARK

    # Log returns
    logret_cols = [f"{t}_logret" for t in tickers if f"{t}_logret" in master.columns]
    returns = master[logret_cols].copy()
    returns.columns = [c.replace("_logret", "") for c in returns.columns]

    # Simple returns for portfolio eval
    ret_cols = [f"{t}_ret" for t in tickers if f"{t}_ret" in master.columns]
    simple_ret = master[ret_cols].copy()
    simple_ret.columns = [c.replace("_ret", "") for c in simple_ret.columns]

    # Volatility
    vol_cols = [f"{t}_vol" for t in tickers if f"{t}_vol" in master.columns]
    vol = master[vol_cols].copy() if vol_cols else pd.DataFrame(index=master.index)
    vol.columns = [c.replace("_vol", "") for c in vol.columns]

    # Macro
    macro_cols = [c for c in cfg.MACRO_VARS if c in master.columns]
    macro = master[macro_cols].copy().ffill().fillna(0.0)

    # Macro derived (stress composite etc.)
    derived_cols = [c for c in master.columns if "macro_" in c or "stress" in c.lower()]
    macro_derived = master[derived_cols].copy().ffill().fillna(0.0) \
                    if derived_cols else pd.DataFrame(index=master.index)

    # Cash rate
    cash_rate = master["TBILL_daily"].fillna(0.0) \
                if "TBILL_daily" in master.columns \
                else pd.Series(0.0, index=master.index)

    # Benchmark returns
    bench_ret = master[f"{benchmark}_ret"].fillna(0.0) \
                if f"{benchmark}_ret" in master.columns \
                else pd.Series(0.0, index=master.index)

    print(f"[loader] Option {option} ({len(tickers)} ETFs): "
          f"{len(master)} days, {master.index[0].date()} → {master.index[-1].date()}")

    return {
        "option":       option,
        "tickers":      tickers,
        "benchmark":    benchmark,
        "returns":      returns,
        "simple_ret":   simple_ret,
        "vol":          vol,
        "macro":        macro,
        "macro_derived":macro_derived,
        "cash_rate":    cash_rate,
        "bench_ret":    bench_ret,
    }
