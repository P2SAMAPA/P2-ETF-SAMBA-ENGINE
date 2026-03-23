# P2-ETF-SAMBA-ENGINE

**Graph-Mamba ETF Signal Engine**

SAMBA replaces the LSTM encoder in DeePM with Mamba's selective state space model (SSM), achieving better long-range sequence modelling with lower memory footprint. Cross-asset relationships are captured via a dynamic graph layer conditioned on macro context.

---

## Architecture

```
x_asset (B, n_assets, lookback, n_asset_feats)
x_macro (B, lookback, n_macro_feats)
        ↓
MambaAssetEncoder (selective SSM, 2 layers) → (B, A, D_MODEL=64)
MacroEncoder (linear + mean pool)           → (B, 32)
DynamicGraphLayer (macro-gated attention)   → (B, A, 64)
PortfolioHead (MLP 128→64→A, softmax)       → weights (B, A)
```

**Key difference from DeePM:**
- DeePM uses LSTM → Mamba uses selective SSM (handles longer sequences, less memory)
- DeePM uses fixed macro graph prior → SAMBA learns dynamic adjacency per forward pass
- No CASH output — model always picks an ETF

---

## ETF Universe

### Option A — Fixed Income / Alternatives (benchmark: AGG)
TLT · LQD · HYG · VNQ · GLD · SLV · PFF · MBB

### Option B — Equity Sectors (benchmark: SPY)
SPY · QQQ · XLK · XLF · XLE · XLV · XLI · XLY · XLP · XLU · GDX · XME

---

## Training

- Fixed split: 70% train / 15% val / 15% test
- Shrinking windows: 8 windows (2008→2024 down to 2022→2024), all OOS 2025-01-01→today
- Both EVaR and Sharpe loss trained, winner = highest OOS annualised return
- MAX_EPOCHS=150, early stopping PATIENCE=15

---

## Setup

### Secrets

| Secret | Value |
|--------|-------|
| `HF_TOKEN` | HuggingFace write token |
| `HF_DATASET_REPO` | `P2SAMAPA/p2-etf-deepm-data` |
| `HF_MODELS_REPO` | `P2SAMAPA/p2-etf-samba-models` |

### First run

```
GitHub Actions → SAMBA Train and Predict → Run workflow
```

Training takes ~60-90 minutes on CPU (Mamba SSM is more efficient than LSTM).

---

## Disclaimer

Research and educational purposes only. Not financial advice.
