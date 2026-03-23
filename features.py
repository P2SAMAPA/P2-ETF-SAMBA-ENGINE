# features.py — Feature engineering for SAMBA

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import config as cfg


def build_asset_features(returns: pd.DataFrame, vol: pd.DataFrame) -> pd.DataFrame:
    """
    Per-asset features: log return, rolling momentum, volatility-scaled return.
    Returns flat DataFrame with columns like TLT_logret, TLT_mom5, TLT_volscale etc.
    """
    feats = {}
    for ticker in returns.columns:
        r = returns[ticker]
        v = vol[ticker] if ticker in vol.columns else r.rolling(21).std() * np.sqrt(252)

        feats[f"{ticker}_ret"]      = r
        feats[f"{ticker}_mom5"]     = r.rolling(5).sum()
        feats[f"{ticker}_mom21"]    = r.rolling(21).sum()
        feats[f"{ticker}_mom63"]    = r.rolling(63).sum()
        feats[f"{ticker}_vol"]      = v
        feats[f"{ticker}_volscale"] = (r / (v.replace(0, np.nan) + 1e-8)).fillna(0.0)

    df = pd.DataFrame(feats, index=returns.index)
    return df.ffill().fillna(0.0)


def build_macro_features(macro: pd.DataFrame,
                         macro_derived: pd.DataFrame) -> pd.DataFrame:
    """Macro features: raw + z-scores + stress composite."""
    parts = [macro]
    if not macro_derived.empty:
        parts.append(macro_derived)
    df = pd.concat(parts, axis=1)
    df = df.loc[:, ~df.columns.duplicated()]
    return df.ffill().fillna(0.0)


class FeatureScaler:
    """RobustScaler per feature — fit on train, transform on all."""

    def __init__(self):
        self.asset_scaler = RobustScaler()
        self.macro_scaler = RobustScaler()
        self._fitted = False

    def fit_transform(self, X_asset: np.ndarray,
                      X_macro: np.ndarray) -> tuple:
        B, A, L, F = X_asset.shape
        Xa = X_asset.reshape(-1, F)
        Xm = X_macro.reshape(-1, X_macro.shape[-1])

        Xa_s = self.asset_scaler.fit_transform(Xa).reshape(B, A, L, F)
        Xm_s = self.macro_scaler.fit_transform(Xm).reshape(
            X_macro.shape[0], X_macro.shape[1], -1)
        self._fitted = True
        return Xa_s.astype(np.float32), Xm_s.astype(np.float32)

    def transform(self, X_asset: np.ndarray,
                  X_macro: np.ndarray) -> tuple:
        if not self._fitted:
            return self.fit_transform(X_asset, X_macro)
        B, A, L, F = X_asset.shape
        Xa = X_asset.reshape(-1, F)
        Xm = X_macro.reshape(-1, X_macro.shape[-1])

        Xa_s = self.asset_scaler.transform(Xa).reshape(B, A, L, F)
        Xm_s = self.macro_scaler.transform(Xm).reshape(
            X_macro.shape[0], X_macro.shape[1], -1)
        return Xa_s.astype(np.float32), Xm_s.astype(np.float32)


def build_sequences(asset_feat: pd.DataFrame, macro_feat: pd.DataFrame,
                    tickers: list, lookback: int = 60) -> dict:
    """
    Build rolling window sequences for training.
    Returns X_asset (N, A, L, F_a), X_macro (N, L, F_m),
            dates (N,), n_asset_feats, n_macro_feats.
    """
    common = asset_feat.index.intersection(macro_feat.index)
    af = asset_feat.reindex(common).ffill().fillna(0.0)
    mf = macro_feat.reindex(common).ffill().fillna(0.0)

    n_assets      = len(tickers)
    asset_col_map = []
    for t in tickers:
        cols = [c for c in af.columns if c.startswith(f"{t}_")]
        asset_col_map.append([af.columns.get_loc(c) for c in cols])

    n_asset_feats = len(asset_col_map[0])
    n_macro_feats = mf.shape[1]
    n_samples     = len(common) - lookback

    X_asset = np.zeros((n_samples, n_assets, lookback, n_asset_feats), dtype=np.float32)
    X_macro = np.zeros((n_samples, lookback, n_macro_feats), dtype=np.float32)
    dates   = []

    af_arr = af.values
    mf_arr = mf.values

    for i in range(n_samples):
        window = af_arr[i: i + lookback]
        for a, col_idxs in enumerate(asset_col_map):
            X_asset[i, a] = window[:, col_idxs]
        X_macro[i] = mf_arr[i: i + lookback]
        dates.append(common[i + lookback])

    return {
        "X_asset":       X_asset,
        "X_macro":       X_macro,
        "dates":         np.array(dates),
        "n_assets":      n_assets,
        "n_asset_feats": n_asset_feats,
        "n_macro_feats": n_macro_feats,
        "tickers":       tickers,
        "asset_col_map": asset_col_map,
    }


def build_labels(returns: pd.DataFrame, tickers: list,
                 cash_rate: pd.Series, lookback: int = 60) -> tuple:
    """
    Build return labels aligned with sequences.
    Returns y (N, n_assets), cash (N,).
    """
    common = returns.index
    n      = len(common) - lookback
    y      = returns[tickers].iloc[lookback:].values.astype(np.float32)
    cash   = cash_rate.reindex(common).ffill().fillna(0.0).iloc[lookback:].values.astype(np.float32)
    return y, cash


def prepare_features(data: dict) -> dict:
    """Full feature preparation pipeline."""
    asset_feat = build_asset_features(data["returns"], data["vol"])
    macro_feat = build_macro_features(data["macro"], data["macro_derived"])

    seq_dict   = build_sequences(
        asset_feat, macro_feat, data["tickers"], cfg.LOOKBACK
    )
    y, cash    = build_labels(
        data["returns"], data["tickers"], data["cash_rate"], cfg.LOOKBACK
    )

    seq_dict["y"]         = y
    seq_dict["cash_rate"] = cash

    # Align dates with simple returns for backtest
    seq_dict["simple_ret"] = data["simple_ret"].reindex(
        seq_dict["dates"]
    ).ffill().fillna(0.0)

    return seq_dict
