# predict.py — SAMBA daily signal generation
#
# Usage:
#   python predict.py --option both

import argparse
import json
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import torch
from huggingface_hub import HfApi, hf_hub_download   # added for remote history

import config as cfg
import loader
import features as feat
from model import SAMBA

DEVICE = torch.device("cpu")


def next_trading_day(from_date: str = None) -> str:
    nyse = mcal.get_calendar("NYSE")
    base = pd.Timestamp(from_date) if from_date else pd.Timestamp.today()
    schedule = nyse.schedule(
        start_date=base.strftime("%Y-%m-%d"),
        end_date=(base + pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
    )
    days = mcal.date_range(schedule, frequency="1D").normalize().tz_localize(None)
    future = [d for d in days if d > base]
    return str(future[0].date()) if future else str((base + pd.Timedelta(days=1)).date())


def _load_samba(model_path: str, meta: dict) -> SAMBA:
    """Load SAMBA model from checkpoint — auto-detects architecture from meta."""
    cfg_m = meta.get("config", {})
    model = SAMBA(
        n_assets=meta["n_assets"],
        n_asset_feats=meta["n_asset_feats"],
        n_macro_feats=meta["n_macro_feats"],
        d_model=cfg_m.get("d_model", cfg.D_MODEL),
        d_state=cfg_m.get("d_state", cfg.D_STATE),
        d_conv=cfg_m.get("d_conv", cfg.D_CONV),
        expand=cfg_m.get("expand", cfg.EXPAND),
        n_mamba_layers=cfg_m.get("n_mamba_layers", cfg.N_MAMBA_LAYERS),
        macro_hidden_dim=cfg_m.get("macro_hidden_dim", cfg.MACRO_HIDDEN_DIM),
        graph_hidden_dim=cfg_m.get("graph_hidden_dim", cfg.GRAPH_HIDDEN_DIM),
        n_attn_heads=cfg_m.get("n_attn_heads", cfg.N_ATTN_HEADS),
        dropout=0.0,
    ).to(DEVICE)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def _build_inference_input(data: dict, meta: dict) -> tuple:
    """Build last-window inference tensors."""
    lookback = meta.get("lookback", cfg.LOOKBACK)
    tickers  = meta["tickers"]

    asset_feat = feat.build_asset_features(data["returns"], data["vol"])
    macro_feat = feat.build_macro_features(data["macro"], data["macro_derived"])

    common = asset_feat.index.intersection(macro_feat.index)
    af = asset_feat.reindex(common).ffill().fillna(0.0)
    mf = macro_feat.reindex(common).ffill().fillna(0.0)

    asset_col_map = []
    for t in tickers:
        cols = [c for c in af.columns if c.startswith(f"{t}_")]
        asset_col_map.append([af.columns.get_loc(c) for c in cols])

    n_assets      = len(tickers)
    n_asset_feats = len(asset_col_map[0])
    af_w = af.iloc[-lookback:].values
    mf_w = mf.iloc[-lookback:].values

    X_asset = np.zeros((1, n_assets, lookback, n_asset_feats), dtype=np.float32)
    X_macro = mf_w[np.newaxis, :, :].astype(np.float32)

    for a, col_idxs in enumerate(asset_col_map):
        X_asset[0, a] = af_w[:, col_idxs]

    last_date = str(af.index[-1].date())
    return X_asset, X_macro, last_date


def _run_inference(model: SAMBA, scaler, X_asset, X_macro,
                   tickers: list) -> tuple:
    X_asset_s, X_macro_s = scaler.transform(X_asset, X_macro)
    with torch.no_grad():
        weights = model(
            torch.tensor(X_asset_s), torch.tensor(X_macro_s)
        ).numpy()[0]

    pick_idx   = int(np.argmax(weights))
    pick       = tickers[pick_idx]
    conviction = float(weights[pick_idx])
    weights_dict = {tickers[i]: round(float(weights[i]), 4)
                    for i in range(len(tickers))}
    return pick, conviction, weights_dict


def generate_signal(option: str, master: pd.DataFrame) -> dict:
    print(f"\n[predict] Generating fixed split signal for Option {option}...")

    model_path  = os.path.join(cfg.MODELS_DIR, f"samba_option{option}_best.pt")
    meta_path   = os.path.join(cfg.MODELS_DIR, f"meta_option{option}.json")
    scaler_path = os.path.join(cfg.MODELS_DIR, f"scaler_option{option}.pkl")

    if not os.path.exists(model_path):
        print(f"  No model found — run train.py first.")
        return None

    with open(meta_path)    as f: meta   = json.load(f)
    with open(scaler_path, "rb") as f: scaler = pickle.load(f)

    data          = loader.get_option_data(option, master)
    model         = _load_samba(model_path, meta)
    X_asset, X_macro, last_date = _build_inference_input(data, meta)
    pick, conviction, weights   = _run_inference(model, scaler, X_asset,
                                                  X_macro, meta["tickers"])
    signal_date = next_trading_day(last_date)

    rc = data["macro"].iloc[-1]
    regime_context = {
        k: round(float(rc.get(k, 0)), 3)
        for k in ["VIX", "T10Y2Y", "HY_SPREAD", "USD_INDEX"]
    }

    print(f"  Option {option}: {pick} (conviction={conviction:.1%}) for {signal_date}")
    return {
        "option":          option,
        "mode":            "fixed_split",
        "option_name":     "Fixed Income / Alts" if option == "A" else "Equity Sectors",
        "signal_date":     signal_date,
        "last_data_date":  last_date,
        "generated_at":    datetime.utcnow().isoformat(),
        "pick":            pick,
        "conviction":      round(conviction, 4),
        "weights":         weights,
        "regime_context":  regime_context,
        "trained_at":      meta.get("trained_at", ""),
        "winning_loss":    meta.get("winning_loss", ""),
        "test_ann_return": meta.get("test_ann_return", 0),
        "test_ann_vol":    meta.get("test_ann_vol",    0),
        "test_sharpe":     meta.get("test_sharpe",     0),
        "test_max_dd":     meta.get("test_max_dd",     0),
        "test_hit_rate":   meta.get("test_hit_rate",   0),
        "test_start":      meta.get("test_start", ""),
        "model_n_params":  meta.get("n_params", 0),
    }


def generate_window_signal(option: str, master: pd.DataFrame) -> dict:
    print(f"\n[predict] Generating window signal for Option {option}...")

    model_path  = os.path.join(cfg.MODELS_DIR,
                               f"samba_option{option}_window_best.pt")
    meta_path   = os.path.join(cfg.MODELS_DIR,
                               f"meta_option{option}_window.json")
    scaler_path = os.path.join(cfg.MODELS_DIR,
                               f"scaler_option{option}_window.pkl")

    if not os.path.exists(model_path):
        print(f"  No window model found — run train_windows.py first.")
        return None

    with open(meta_path)    as f: meta   = json.load(f)
    with open(scaler_path, "rb") as f: scaler = pickle.load(f)

    data          = loader.get_option_data(option, master)
    model         = _load_samba(model_path, meta)
    X_asset, X_macro, last_date = _build_inference_input(data, meta)
    pick, conviction, weights   = _run_inference(model, scaler, X_asset,
                                                  X_macro, meta["tickers"])
    signal_date = next_trading_day(last_date)

    print(f"  Option {option} window: {pick} (conviction={conviction:.1%}) | "
          f"Window {meta['winning_window']}: "
          f"{meta['winning_train_start']}→{meta['winning_train_end']}")
    return {
        "option":              option,
        "mode":                "shrinking_window",
        "option_name":         "Fixed Income / Alts" if option == "A" else "Equity Sectors",
        "signal_date":         signal_date,
        "last_data_date":      last_date,
        "generated_at":        datetime.utcnow().isoformat(),
        "pick":                pick,
        "conviction":          round(conviction, 4),
        "weights":             weights,
        "trained_at":          meta.get("trained_at", ""),
        "winning_window":      meta.get("winning_window", 0),
        "winning_train_start": meta.get("winning_train_start", ""),
        "winning_train_end":   meta.get("winning_train_end", ""),
        "winning_loss":        meta.get("winning_loss", ""),
        "oos_ann_return":      meta.get("oos_ann_return", 0),
        "oos_ann_vol":         meta.get("oos_ann_vol", 0),
        "oos_sharpe":          meta.get("oos_sharpe", 0),
        "oos_hit_rate":        meta.get("oos_hit_rate", 0),
        "oos_max_dd":          meta.get("oos_max_dd", 0),
    }


def update_history(signal: dict, option: str) -> None:
    """
    Append new signal to history and upload to Hugging Face.
    Downloads existing history from HF first.
    """
    # Local path
    local_path = os.path.join(cfg.MODELS_DIR, f"signal_history_{option}.json")
    history = []

    # Try to download existing history from Hugging Face
    try:
        downloaded = hf_hub_download(
            repo_id=cfg.HF_MODELS_REPO,
            filename=f"signal_history_{option}.json",
            repo_type="dataset",
            token=cfg.HF_TOKEN,
            local_dir=cfg.MODELS_DIR,
            local_dir_use_symlinks=False,
        )
        with open(downloaded) as f:
            history = json.load(f)
        print(f"[predict] Loaded existing history: {len(history)} records for Option {option}")
    except Exception as e:
        print(f"[predict] No existing history found for Option {option} (starting fresh): {e}")

    # Create new record
    record = {
        "signal_date":  signal["signal_date"],
        "pick":         signal["pick"],
        "conviction":   signal["conviction"],
        "generated_at": signal["generated_at"],
    }

    if record["signal_date"] not in {r["signal_date"] for r in history}:
        history.append(record)
        print(f"[predict] Appended new record for {record['signal_date']}")
    else:
        print(f"[predict] Record for {record['signal_date']} already exists – skipping")

    # Save locally
    with open(local_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[predict] History: {len(history)} records for Option {option}")


def _best_signal(sig_fixed: dict, sig_window: dict) -> dict:
    """Return whichever signal has higher return — used for history recording."""
    ret_f = sig_fixed.get("test_ann_return",  -999) if sig_fixed else -999
    ret_w = sig_window.get("oos_ann_return",   -999) if sig_window else -999
    return sig_window if (ret_w > ret_f and sig_window and "pick" in sig_window) \
           else (sig_fixed or {})


def save_signals(sig_A=None, sig_B=None, sig_Aw=None, sig_Bw=None):
    os.makedirs(cfg.MODELS_DIR, exist_ok=True)
    combined = {
        "generated_at":    datetime.utcnow().isoformat(),
        "option_A":        sig_A,
        "option_B":        sig_B,
        "option_A_window": sig_Aw,
        "option_B_window": sig_Bw,
    }
    with open(os.path.join(cfg.MODELS_DIR, "latest_signals.json"), "w") as f:
        json.dump(combined, f, indent=2)

    for sig, name in [
        (sig_A,  "signal_A"),
        (sig_B,  "signal_B"),
        (sig_Aw, "signal_A_window"),
        (sig_Bw, "signal_B_window"),
    ]:
        if sig:
            with open(os.path.join(cfg.MODELS_DIR, f"{name}.json"), "w") as f:
                json.dump(sig, f, indent=2)

    # Record BEST signal (hero pick) in history — matches what app shows
    if sig_A or sig_Aw:
        update_history(_best_signal(sig_A, sig_Aw), "A")
    if sig_B or sig_Bw:
        update_history(_best_signal(sig_B, sig_Bw), "B")

    print("[predict] All signals saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--option", choices=["A", "B", "both"], default="both")
    args = parser.parse_args()

    print("[predict] Loading master dataset...")
    master = loader.load_master()

    sig_A = sig_B = sig_Aw = sig_Bw = None

    if args.option in ("A", "both"):
        sig_A  = generate_signal("A", master)
        sig_Aw = generate_window_signal("A", master)

    if args.option in ("B", "both"):
        sig_B  = generate_signal("B", master)
        sig_Bw = generate_window_signal("B", master)

    save_signals(sig_A, sig_B, sig_Aw, sig_Bw)

    print("\n[predict] Done.")
    for sig, label in [(sig_A, "A fixed"), (sig_B, "B fixed"),
                       (sig_Aw, "A window"), (sig_Bw, "B window")]:
        if sig:
            print(f"  Option {label}: {sig['pick']} on {sig['signal_date']} "
                  f"(conviction={sig['conviction']:.1%})")
