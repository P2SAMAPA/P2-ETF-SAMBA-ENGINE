# train_windows.py — SAMBA shrinking window training
# 8 windows × 2 loss functions, winner = highest OOS ann return
#
# Usage:
#   python train_windows.py --option both

import argparse
import json
import os
import pickle
import shutil
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import config as cfg
import loader
import features as feat
from model import SAMBA, sharpe_loss, evar_loss, count_parameters
from train import train_epoch, eval_epoch, DEVICE

os.makedirs(cfg.MODELS_DIR, exist_ok=True)


def make_window_dataloaders(feat_dict: dict, scaler: feat.FeatureScaler,
                             train_start: str, train_end: str,
                             oos_start: str) -> tuple:
    dates   = feat_dict["dates"]
    X_asset = feat_dict["X_asset"]
    X_macro = feat_dict["X_macro"]
    y       = feat_dict["y"]
    cash    = feat_dict["cash_rate"]

    train_mask = (dates >= np.datetime64(train_start)) & \
                 (dates <= np.datetime64(train_end))
    oos_mask   = dates >= np.datetime64(oos_start)

    if train_mask.sum() < cfg.LOOKBACK * 2:
        raise ValueError(f"Too few training samples: {train_mask.sum()}")

    Xa_tr, Xm_tr = X_asset[train_mask], X_macro[train_mask]
    y_tr, c_tr   = y[train_mask], cash[train_mask]
    Xa_oo, Xm_oo = X_asset[oos_mask], X_macro[oos_mask]
    y_oo, c_oo   = y[oos_mask], cash[oos_mask]

    Xa_tr_s, Xm_tr_s = scaler.fit_transform(Xa_tr, Xm_tr)
    Xa_oo_s, Xm_oo_s = scaler.transform(Xa_oo, Xm_oo)

    def ds(Xa, Xm, yy, cc):
        return TensorDataset(
            torch.tensor(Xa), torch.tensor(Xm),
            torch.tensor(yy), torch.tensor(cc),
        )

    train_dl = DataLoader(ds(Xa_tr_s, Xm_tr_s, y_tr, c_tr),
                          batch_size=cfg.BATCH_SIZE, shuffle=False)
    oos_dl   = DataLoader(ds(Xa_oo_s, Xm_oo_s, y_oo, c_oo),
                          batch_size=cfg.BATCH_SIZE, shuffle=False)

    return train_dl, oos_dl, int(train_mask.sum()), int(oos_mask.sum())


def train_window(window: dict, feat_dict: dict, option: str,
                 loss_fn: str) -> dict:
    wid = window["id"]
    scaler = feat.FeatureScaler()

    try:
        train_dl, oos_dl, n_train, n_oos = make_window_dataloaders(
            feat_dict, scaler,
            train_start=window["start"],
            train_end=cfg.TRAIN_END,
            oos_start=cfg.LIVE_START,
        )
    except ValueError as e:
        print(f"  Window {wid} skipped: {e}")
        return None

    print(f"  Window {wid} ({window['start']}→{cfg.TRAIN_END}) | "
          f"loss={loss_fn} | train={n_train} | oos={n_oos}")

    model = SAMBA(
        n_assets=feat_dict["n_assets"],
        n_asset_feats=feat_dict["n_asset_feats"],
        n_macro_feats=feat_dict["n_macro_feats"],
        d_model=cfg.D_MODEL, d_state=cfg.D_STATE,
        d_conv=cfg.D_CONV, expand=cfg.EXPAND,
        n_mamba_layers=cfg.N_MAMBA_LAYERS,
        macro_hidden_dim=cfg.MACRO_HIDDEN_DIM,
        graph_hidden_dim=cfg.GRAPH_HIDDEN_DIM,
        n_attn_heads=cfg.N_ATTN_HEADS,
        dropout=cfg.DROPOUT,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    model_path = os.path.join(
        cfg.MODELS_DIR, f"samba_option{option}_w{wid}_{loss_fn}.pt"
    )
    best_oos = -float("inf")
    patience = 0

    for epoch in range(1, cfg.MAX_EPOCHS + 1):
        train_epoch(model, train_dl, optimizer, loss_fn)
        oos_ann_ret, oos_sharpe = eval_epoch(model, oos_dl, loss_fn)
        scheduler.step(oos_ann_ret)

        if oos_ann_ret > best_oos:
            best_oos = oos_ann_ret
            patience = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience += 1
        if patience >= cfg.PATIENCE:
            break

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    oos_ann_ret, oos_sharpe = eval_epoch(model, oos_dl, loss_fn)

    # Full OOS metrics
    model.eval()
    all_rets = []
    with torch.no_grad():
        for Xa, Xm, y_b, c_b in oos_dl:
            w = model(Xa, Xm)
            all_rets.append((w * y_b).sum(dim=1).numpy())
    r = np.concatenate(all_rets)
    ann_vol  = float(r.std() * np.sqrt(252))
    curve    = np.cumprod(1 + r)
    max_dd   = float(((curve - np.maximum.accumulate(curve)) /
                      np.maximum.accumulate(curve)).min())
    hit_rate = float((r > 0).mean())

    print(f"  Window {wid} result: OOS={oos_ann_ret*100:.2f}% | "
          f"Sharpe={oos_sharpe:.3f}")

    return {
        "window_id":     wid,
        "train_start":   window["start"],
        "train_end":     cfg.TRAIN_END,
        "loss_fn":       loss_fn,
        "oos_ann_return":round(oos_ann_ret, 4),
        "oos_ann_vol":   round(ann_vol, 4),
        "oos_sharpe":    round(oos_sharpe, 4),
        "oos_hit_rate":  round(hit_rate, 4),
        "oos_max_dd":    round(max_dd, 4),
        "model_path":    model_path,
        "scaler":        scaler,
    }


def train_windows_option(option: str) -> dict:
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"SAMBA Shrinking Windows — Option {'A (FI)' if option == 'A' else 'B (Equity)'}")
    print(f"{'='*60}")

    master    = loader.load_master()
    data      = loader.get_option_data(option, master)
    feat_dict = feat.prepare_features(data)

    all_results  = []
    best_result  = None
    best_return  = -float("inf")

    for window in cfg.WINDOWS:
        for loss_fn in ["sharpe", "evar"]:
            result = train_window(window, feat_dict, option, loss_fn)
            if result is None:
                continue
            all_results.append({k: v for k, v in result.items()
                                 if k not in ("scaler", "model_path")})
            if result["oos_ann_return"] > best_return:
                best_return = result["oos_ann_return"]
                best_result = result

    if best_result is None:
        raise RuntimeError("All windows failed.")

    canonical = os.path.join(cfg.MODELS_DIR,
                             f"samba_option{option}_window_best.pt")
    scaler_p  = os.path.join(cfg.MODELS_DIR,
                             f"scaler_option{option}_window.pkl")
    shutil.copy2(best_result["model_path"], canonical)
    with open(scaler_p, "wb") as f:
        pickle.dump(best_result["scaler"], f)

    summary = {
        "option":              option,
        "trained_at":          datetime.utcnow().isoformat(),
        "elapsed_sec":         round(time.time() - t0, 1),
        "winning_window":      best_result["window_id"],
        "winning_train_start": best_result["train_start"],
        "winning_train_end":   best_result["train_end"],
        "winning_loss":        best_result["loss_fn"],
        "oos_ann_return":      best_result["oos_ann_return"],
        "oos_ann_vol":         best_result["oos_ann_vol"],
        "oos_sharpe":          best_result["oos_sharpe"],
        "oos_hit_rate":        best_result["oos_hit_rate"],
        "oos_max_dd":          best_result["oos_max_dd"],
        "n_assets":            feat_dict["n_assets"],
        "tickers":             feat_dict["tickers"],
        "n_asset_feats":       feat_dict["n_asset_feats"],
        "n_macro_feats":       feat_dict["n_macro_feats"],
        "all_windows":         all_results,
        "config": {
            "lookback":         cfg.LOOKBACK,
            "d_model":          cfg.D_MODEL,
            "d_state":          cfg.D_STATE,
            "d_conv":           cfg.D_CONV,
            "expand":           cfg.EXPAND,
            "n_mamba_layers":   cfg.N_MAMBA_LAYERS,
            "macro_hidden_dim": cfg.MACRO_HIDDEN_DIM,
            "graph_hidden_dim": cfg.GRAPH_HIDDEN_DIM,
            "n_attn_heads":     cfg.N_ATTN_HEADS,
        },
    }

    meta_path = os.path.join(cfg.MODELS_DIR,
                             f"meta_option{option}_window.json")
    with open(meta_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Winner: Window {best_result['window_id']} "
          f"({best_result['train_start']}→{best_result['train_end']}) "
          f"| loss={best_result['loss_fn']} "
          f"| OOS={best_result['oos_ann_return']*100:.2f}%")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--option", choices=["A", "B", "both"], default="both")
    args = parser.parse_args()

    options = ["A", "B"] if args.option == "both" else [args.option]
    for opt in options:
        train_windows_option(opt)
