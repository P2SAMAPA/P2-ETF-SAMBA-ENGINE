# train.py — SAMBA fixed split training (70/15/15)
#
# Usage:
#   python train.py --option A --loss both
#   python train.py --option both --loss both

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

os.makedirs(cfg.MODELS_DIR, exist_ok=True)
DEVICE = torch.device("cpu")


# ── Data helpers ───────────────────────────────────────────────────────────────

def make_dataloaders(feat_dict: dict, scaler: feat.FeatureScaler) -> tuple:
    X_asset = feat_dict["X_asset"]
    X_macro = feat_dict["X_macro"]
    y       = feat_dict["y"]
    cash    = feat_dict["cash_rate"]
    dates   = feat_dict["dates"]

    n       = len(dates)
    n_train = int(n * cfg.TRAIN_SPLIT)
    n_val   = int(n * cfg.VAL_SPLIT)

    Xa_tr, Xm_tr, y_tr, c_tr = X_asset[:n_train], X_macro[:n_train], y[:n_train], cash[:n_train]
    Xa_va, Xm_va, y_va, c_va = X_asset[n_train:n_train+n_val], X_macro[n_train:n_train+n_val], \
                                y[n_train:n_train+n_val], cash[n_train:n_train+n_val]
    Xa_te, Xm_te, y_te, c_te = X_asset[n_train+n_val:], X_macro[n_train+n_val:], \
                                y[n_train+n_val:], cash[n_train+n_val:]

    Xa_tr_s, Xm_tr_s = scaler.fit_transform(Xa_tr, Xm_tr)
    Xa_va_s, Xm_va_s = scaler.transform(Xa_va, Xm_va)
    Xa_te_s, Xm_te_s = scaler.transform(Xa_te, Xm_te)

    def ds(Xa, Xm, yy, cc):
        return TensorDataset(
            torch.tensor(Xa), torch.tensor(Xm),
            torch.tensor(yy), torch.tensor(cc),
        )

    train_dl = DataLoader(ds(Xa_tr_s, Xm_tr_s, y_tr, c_tr),
                          batch_size=cfg.BATCH_SIZE, shuffle=False)
    val_dl   = DataLoader(ds(Xa_va_s, Xm_va_s, y_va, c_va),
                          batch_size=cfg.BATCH_SIZE, shuffle=False)
    test_dl  = DataLoader(ds(Xa_te_s, Xm_te_s, y_te, c_te),
                          batch_size=cfg.BATCH_SIZE, shuffle=False)

    test_dates = dates[n_train + n_val:]
    return train_dl, val_dl, test_dl, test_dates


# ── Epoch helpers ──────────────────────────────────────────────────────────────

def train_epoch(model, dl, optimizer, loss_fn_name):
    model.train()
    total = 0.0
    for Xa, Xm, y_b, c_b in dl:
        optimizer.zero_grad()
        w = model(Xa, Xm)
        loss = sharpe_loss(w, y_b, c_b) if loss_fn_name == "sharpe" \
               else evar_loss(w, y_b, c_b)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    return total / len(dl)


def eval_epoch(model, dl, loss_fn_name):
    model.eval()
    all_rets = []
    with torch.no_grad():
        for Xa, Xm, y_b, c_b in dl:
            w = model(Xa, Xm)
            port_ret = (w * y_b).sum(dim=1)
            all_rets.append(port_ret.numpy())
    r        = np.concatenate(all_rets)
    ann_ret  = float(r.mean() * 252)
    ann_vol  = float(r.std() * np.sqrt(252) + 1e-8)
    sharpe   = ann_ret / ann_vol
    curve    = np.cumprod(1 + r)
    max_dd   = float(((curve - np.maximum.accumulate(curve)) /
                      np.maximum.accumulate(curve)).min())
    hit_rate = float((r > 0).mean())
    return ann_ret, sharpe, ann_vol, max_dd, hit_rate


# ── Train one option one loss ──────────────────────────────────────────────────

def train_one(option: str, loss_fn: str, feat_dict: dict,
              data: dict) -> dict:
    print(f"\n  Training Option {option} | loss={loss_fn}")

    scaler    = feat.FeatureScaler()
    train_dl, val_dl, test_dl, test_dates = make_dataloaders(feat_dict, scaler)

    model = SAMBA(
        n_assets=feat_dict["n_assets"],
        n_asset_feats=feat_dict["n_asset_feats"],
        n_macro_feats=feat_dict["n_macro_feats"],
        d_model=cfg.D_MODEL,
        d_state=cfg.D_STATE,
        d_conv=cfg.D_CONV,
        expand=cfg.EXPAND,
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

    model_path = os.path.join(cfg.MODELS_DIR,
                              f"samba_option{option}_{loss_fn}_best.pt")
    best_val   = -float("inf")
    patience   = 0

    for epoch in range(1, cfg.MAX_EPOCHS + 1):
        train_loss          = train_epoch(model, train_dl, optimizer, loss_fn)
        val_ann_ret, val_sh, _, _, _ = eval_epoch(model, val_dl, loss_fn)
        scheduler.step(val_ann_ret)

        if val_ann_ret > best_val:
            best_val = val_ann_ret
            patience = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience += 1

        if patience >= cfg.PATIENCE:
            print(f"    Early stop at epoch {epoch}")
            break

        if epoch % 20 == 0:
            print(f"    Epoch {epoch:3d} | train_loss={train_loss:.4f} "
                  f"| val_ann_ret={val_ann_ret*100:.2f}% | val_sh={val_sh:.3f}")

    # Final test eval
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    test_ann_ret, test_sharpe, test_vol, test_dd, test_hr = eval_epoch(model, test_dl, loss_fn)

    n_params = count_parameters(model)
    print(f"  Result: test_ann_ret={test_ann_ret*100:.2f}% | "
          f"test_sharpe={test_sharpe:.3f} | params={n_params:,}")

    return {
        "loss_fn":       loss_fn,
        "test_ann_ret":  round(test_ann_ret, 4),
        "test_sharpe":   round(test_sharpe, 4),
        "test_ann_vol":  round(test_vol, 4),
        "test_max_dd":   round(test_dd, 4),
        "test_hit_rate": round(test_hr, 4),
        "model_path":    model_path,
        "scaler":        scaler,
        "n_params":      n_params,
        "test_dates":    test_dates,
    }


# ── Train option (both loss functions) ────────────────────────────────────────

def train_option(option: str) -> dict:
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"SAMBA Fixed Split — Option {'A (FI)' if option == 'A' else 'B (Equity)'}")
    print(f"{'='*60}")

    master    = loader.load_master()
    data      = loader.get_option_data(option, master)
    feat_dict = feat.prepare_features(data)

    results = {}
    for loss_fn in ["sharpe", "evar"]:
        results[loss_fn] = train_one(option, loss_fn, feat_dict, data)

    # Pick winner by test ann return
    winner = max(results, key=lambda k: results[k]["test_ann_ret"])
    best   = results[winner]

    canonical = os.path.join(cfg.MODELS_DIR, f"samba_option{option}_best.pt")
    scaler_p  = os.path.join(cfg.MODELS_DIR, f"scaler_option{option}.pkl")
    shutil.copy2(best["model_path"], canonical)
    with open(scaler_p, "wb") as f:
        pickle.dump(best["scaler"], f)

    # Test period dates
    n_total = len(feat_dict["dates"])
    n_test  = int(n_total * (1 - cfg.TRAIN_SPLIT - cfg.VAL_SPLIT))
    test_start = str(feat_dict["dates"][-n_test])[:10]

    summary = {
        "option":        option,
        "trained_at":    datetime.utcnow().isoformat(),
        "elapsed_sec":   round(time.time() - t0, 1),
        "winning_loss":    winner,
        "test_ann_return": best["test_ann_ret"],
        "test_ann_vol":    best["test_ann_vol"],
        "test_sharpe":     best["test_sharpe"],
        "test_max_dd":     best["test_max_dd"],
        "test_hit_rate":   best["test_hit_rate"],
        "test_start":      test_start,
        "n_params":      best["n_params"],
        "n_assets":      feat_dict["n_assets"],
        "tickers":       feat_dict["tickers"],
        "n_asset_feats": feat_dict["n_asset_feats"],
        "n_macro_feats": feat_dict["n_macro_feats"],
        "lookback":      cfg.LOOKBACK,
        "config": {
            "d_model":          cfg.D_MODEL,
            "d_state":          cfg.D_STATE,
            "d_conv":           cfg.D_CONV,
            "expand":           cfg.EXPAND,
            "n_mamba_layers":   cfg.N_MAMBA_LAYERS,
            "macro_hidden_dim": cfg.MACRO_HIDDEN_DIM,
            "graph_hidden_dim": cfg.GRAPH_HIDDEN_DIM,
            "n_attn_heads":     cfg.N_ATTN_HEADS,
        },
        "all_results": {
            k: {kk: vv for kk, vv in v.items()
                if kk not in ("scaler", "model_path", "test_dates")}
            for k, v in results.items()
        },
    }

    meta_path = os.path.join(cfg.MODELS_DIR, f"meta_option{option}.json")
    with open(meta_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Winner: {winner} | test_ann_ret={best['test_ann_ret']*100:.2f}% "
          f"| test_sharpe={best['test_sharpe']:.3f}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--option", choices=["A", "B", "both"], default="both")
    args = parser.parse_args()

    options = ["A", "B"] if args.option == "both" else [args.option]
    for opt in options:
        train_option(opt)
