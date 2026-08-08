# aggregate_windows.py — combine per-window SAMBA results produced by parallel
# CI matrix jobs (`train_windows.py --option X --window N`) into the same
# canonical model + meta_option{X}_window.json that train_windows_option()
# used to produce when all 8 windows ran serially in a single job.
#
# Usage (after downloading all `samba-{option}-window-*` artifacts into models/):
#   python aggregate_windows.py --option B

import argparse
import glob
import json
import os
import shutil
import time
from datetime import datetime

import config as cfg
import loader
import features as feat


def aggregate(option: str) -> dict:
    t0 = time.time()

    pattern = os.path.join(cfg.MODELS_DIR, f"result_option{option}_w*_*.json")
    result_files = sorted(glob.glob(pattern))

    if not result_files:
        raise RuntimeError(
            f"No per-window result files found for option {option} "
            f"(looked for {pattern}). Check that the matrix training jobs "
            f"ran and their artifacts were downloaded into {cfg.MODELS_DIR}/ "
            f"before this script runs."
        )

    all_results = []
    best = None
    best_return = -float("inf")

    for path in result_files:
        with open(path) as f:
            r = json.load(f)
        all_results.append(r)
        if r["oos_ann_return"] > best_return:
            best_return = r["oos_ann_return"]
            best = r

    print(f"Loaded {len(all_results)} window/loss result(s) for option {option}:")
    for r in sorted(all_results, key=lambda x: -x["oos_ann_return"]):
        print(f"  window={r['window_id']:>2} loss={r['loss_fn']:<6} "
              f"OOS={r['oos_ann_return']*100:7.2f}%  Sharpe={r['oos_sharpe']:.3f}")

    wid, loss_fn = best["window_id"], best["loss_fn"]
    model_src  = os.path.join(cfg.MODELS_DIR, f"samba_option{option}_w{wid}_{loss_fn}.pt")
    scaler_src = os.path.join(cfg.MODELS_DIR, f"scaler_option{option}_w{wid}_{loss_fn}.pkl")

    canonical  = os.path.join(cfg.MODELS_DIR, f"samba_option{option}_window_best.pt")
    scaler_dst = os.path.join(cfg.MODELS_DIR, f"scaler_option{option}_window.pkl")

    if not os.path.exists(model_src):
        raise RuntimeError(
            f"Winning model file missing: {model_src}. The matrix job for "
            f"window {wid} either failed to upload its artifact, or its "
            f"artifact wasn't downloaded into {cfg.MODELS_DIR}/ before this "
            f"script ran — check the workflow's download-artifact step."
        )

    shutil.copy2(model_src, canonical)
    if os.path.exists(scaler_src):
        shutil.copy2(scaler_src, scaler_dst)
    else:
        print(f"  Warning: no scaler found at {scaler_src} for the winning "
              f"window/loss combo — predict.py may fail to run inference.")

    # Cheap (no training) reload just to keep meta_option{X}_window.json's
    # schema identical to what train_windows_option() used to write, since
    # downstream code (predict.py / the dashboard) may depend on these keys.
    master    = loader.load_master()
    data      = loader.get_option_data(option, master)
    feat_dict = feat.prepare_features(data)

    summary = {
        "option":              option,
        "trained_at":          datetime.utcnow().isoformat(),
        "elapsed_sec":         round(time.time() - t0, 1),  # aggregator's own runtime only
        "winning_window":      best["window_id"],
        "winning_train_start": best["train_start"],
        "winning_train_end":   best["train_end"],
        "winning_loss":        best["loss_fn"],
        "oos_ann_return":      best["oos_ann_return"],
        "oos_ann_vol":         best["oos_ann_vol"],
        "oos_sharpe":          best["oos_sharpe"],
        "oos_hit_rate":        best["oos_hit_rate"],
        "oos_max_dd":          best["oos_max_dd"],
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

    meta_path = os.path.join(cfg.MODELS_DIR, f"meta_option{option}_window.json")
    with open(meta_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWinner: window {wid} loss={loss_fn} "
          f"OOS={best['oos_ann_return']*100:.2f}% Sharpe={best['oos_sharpe']:.3f}")
    print(f"Wrote {canonical}\nWrote {scaler_dst}\nWrote {meta_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--option", choices=["A", "B"], required=True)
    args = parser.parse_args()
    aggregate(args.option)
