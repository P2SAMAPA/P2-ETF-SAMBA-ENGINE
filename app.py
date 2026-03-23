# app.py — P2-ETF-SAMBA-ENGINE Streamlit Dashboard
# Graph-Mamba ETF Signal Engine
# Two tabs: Option A (FI) | Option B (Equity)

import json
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from huggingface_hub import hf_hub_download

import config as cfg

st.set_page_config(
    page_title="SAMBA — Graph-Mamba ETF Engine",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  .stApp { background-color: #ffffff; }

  .hero-card {
    background: #faf5ff; border: 1px solid #e9d5ff;
    border-radius: 14px; padding: 28px 32px 22px 32px; margin-bottom: 24px;
  }
  .hero-ticker { font-size: 64px; font-weight: 700; color: #1a1a2e; line-height: 1.1; }
  .hero-conv   { font-size: 28px; font-weight: 500; color: #7c3aed; margin-top: 6px; }
  .hero-date   { font-size: 15px; color: #6b7280; margin-top: 8px; }
  .hero-source { font-size: 14px; color: #6d28d9; font-weight: 600;
                 background: #ede9fe; border-radius: 20px;
                 padding: 3px 12px; display: inline-block; margin-top: 8px; }
  .runner-up   { font-size: 18px; color: #374151; margin-top: 14px;
                 padding-top: 14px; border-top: 1px solid #e5e7eb; }

  .label-fixed  { display:inline-block; font-size:14px; font-weight:700;
                  color:#374151; text-transform:uppercase; letter-spacing:.07em;
                  background:#f3f4f6; border-radius:6px;
                  padding:5px 14px; margin-bottom:12px; }
  .label-window { display:inline-block; font-size:14px; font-weight:700;
                  color:#6d28d9; text-transform:uppercase; letter-spacing:.07em;
                  background:#ede9fe; border-radius:6px;
                  padding:5px 14px; margin-bottom:12px; }
  .window-badge { font-size:14px; color:#6d28d9; background:#ede9fe;
                  border:1px solid #c4b5fd; border-radius:20px;
                  padding:4px 14px; display:inline-block; margin-bottom:12px; }
  .period-badge { font-size:14px; color:#374151; background:#f3f4f6;
                  border:1px solid #e5e7eb; border-radius:20px;
                  padding:4px 14px; display:inline-block; margin-bottom:12px; }

  .metric-row { display:flex; gap:12px; margin:10px 0 16px 0; }
  .metric-box { flex:1; background:#fff; border:1px solid #e5e7eb;
                border-radius:10px; padding:14px 10px; text-align:center; }
  .metric-label { font-size:12px; color:#6b7280; text-transform:uppercase;
                  letter-spacing:.05em; margin-bottom:6px; }
  .metric-value { font-size:24px; font-weight:600; color:#111827; }
  .pos { color:#059669; } .neg { color:#dc2626; }

  .pill   { display:inline-block; padding:5px 14px; border-radius:20px;
            font-size:15px; font-weight:500; margin:3px 3px 3px 0; }
  .pill-g { background:#d1fae5; color:#065f46; }
  .pill-a { background:#fef3c7; color:#92400e; }
  .pill-r { background:#fee2e2; color:#991b1b; }

  .hit-line { font-size:16px; color:#374151; margin-bottom:10px; }
  .fn       { font-size:13px; color:#9ca3af; margin-top:8px; }
  .sec-hdr  { font-size:20px; font-weight:700; color:#1a1a2e; margin:28px 0 12px 0; }
</style>
""", unsafe_allow_html=True)


# ── Data loading ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=1800)
def load_signals() -> dict:
    try:
        url = (f"https://huggingface.co/datasets/{cfg.HF_MODELS_REPO}"
               f"/resolve/main/models/latest_signals.json")
        headers = {"Authorization": f"Bearer {cfg.HF_TOKEN}"} if cfg.HF_TOKEN else {}
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        raw = r.json()
        return {
            "A":  raw.get("option_A")        or {},
            "B":  raw.get("option_B")        or {},
            "Aw": raw.get("option_A_window") or {},
            "Bw": raw.get("option_B_window") or {},
        }
    except Exception as e:
        st.error(f"Could not load signals: {e}")
        return {"A": {}, "B": {}, "Aw": {}, "Bw": {}}


@st.cache_data(ttl=3600)
def load_master() -> pd.DataFrame:
    try:
        path = hf_hub_download(
            repo_id=cfg.HF_DATASET_REPO,
            filename=cfg.FILE_MASTER,
            repo_type="dataset",
            token=cfg.HF_TOKEN or None,
            force_download=True,
        )
        df = pd.read_parquet(path)
        if "Date" in df.columns:
            df = df.set_index("Date")
        df.index = pd.to_datetime(df.index)
        return df.sort_index()
    except Exception as e:
        st.error(f"Could not load master dataset: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=1800)
def load_history(option: str) -> pd.DataFrame:
    try:
        url = (f"https://huggingface.co/datasets/{cfg.HF_MODELS_REPO}"
               f"/resolve/main/models/signal_history_{option}.json")
        headers = {"Authorization": f"Bearer {cfg.HF_TOKEN}"} if cfg.HF_TOKEN else {}
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        return pd.DataFrame(r.json())
    except Exception:
        return pd.DataFrame()


# ── Helpers ────────────────────────────────────────────────────────────────────

def pill(label, val, lo, hi):
    cls = "pill-g" if val < lo else ("pill-r" if val > hi else "pill-a")
    return f'<span class="pill {cls}">{label}: {val}</span>'


def best_signal(sig_fixed: dict, sig_window: dict) -> tuple:
    ret_fixed  = sig_fixed.get("test_ann_return",  -999) if sig_fixed else -999
    ret_window = sig_window.get("oos_ann_return",   -999) if sig_window else -999
    if ret_window > ret_fixed and sig_window and "pick" in sig_window:
        return sig_window, "Shrinking Window"
    elif sig_fixed and "pick" in sig_fixed:
        return sig_fixed, "Fixed Split"
    return sig_fixed or {}, "—"


def build_bt(pick: str, master: pd.DataFrame, option: str,
             start_date: str = None) -> dict:
    if not pick or master.empty:
        return {}
    benchmark    = cfg.FI_BENCHMARK if option == "A" else cfg.EQ_BENCHMARK
    period_start = start_date or cfg.LIVE_START
    oos = master[master.index >= period_start].copy()
    if oos.empty:
        return {}

    bench_ret = oos.get(f"{benchmark}_ret",
                        pd.Series(0.0, index=oos.index)).fillna(0.0)
    pick_rets = oos.get(f"{pick}_ret",
                        pd.Series(0.0, index=oos.index)).fillna(0.0)
    sc = (1 + pick_rets).cumprod()
    bc = (1 + bench_ret).cumprod()

    return {
        "dates": oos.index, "sc": sc, "bc": bc,
        "pick": pick, "benchmark": benchmark,
    }


# ── UI components ──────────────────────────────────────────────────────────────

def render_hero(sig_fixed: dict, sig_window: dict, option: str):
    best, source = best_signal(sig_fixed, sig_window)
    if not best or "pick" not in best:
        st.info("Signal not available yet — run the training workflow first.")
        return

    tickers     = cfg.FI_ETFS if option == "A" else cfg.EQ_ETFS
    w           = best.get("weights", {})
    picks       = sorted([(t, w.get(t, 0.0)) for t in tickers],
                         key=lambda x: x[1], reverse=True)

    pick       = best["pick"]
    conviction = best.get("conviction", 0)
    sig_date   = best.get("signal_date", "—")
    gen        = best.get("generated_at", "")
    try:
        gen = datetime.fromisoformat(gen).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        pass

    t2 = picks[1] if len(picks) > 1 else None
    t3 = picks[2] if len(picks) > 2 else None
    runner = ""
    if t2: runner += f"<span style='color:#6b7280'>2nd:</span> <b>{t2[0]}</b> {t2[1]*100:.1f}%"
    if t3: runner += f"&nbsp;&nbsp;<span style='color:#6b7280'>3rd:</span> <b>{t3[0]}</b> {t3[1]*100:.1f}%"

    rc    = best.get("regime_context", {})
    pills = ""
    if rc.get("VIX"):             pills += pill("VIX",    rc["VIX"],    15,   25)
    if rc.get("T10Y2Y") is not None: pills += pill("T10Y2Y", rc["T10Y2Y"], -0.5, 0.5)
    if rc.get("HY_SPREAD"):       pills += pill("HY Spr", rc["HY_SPREAD"], 300, 500)

    st.markdown(f"""
    <div class="hero-card">
      <div class="hero-ticker">{pick}</div>
      <div class="hero-conv">{conviction*100:.1f}% conviction</div>
      <div class="hero-date">Signal for {sig_date} &nbsp;·&nbsp; Generated {gen}</div>
      <div class="hero-source">Source: {source}</div>
      <div class="runner-up">{runner}</div>
      <div style="margin-top:16px">{pills}</div>
    </div>
    """, unsafe_allow_html=True)


def render_metrics(m: dict):
    if not m:
        st.caption("Metrics available after first training run.")
        return
    fp = lambda v: f"{v*100:.1f}%"
    c  = lambda v: "pos" if v >= 0 else "neg"
    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-box">
        <div class="metric-label">Ann Return</div>
        <div class="metric-value {c(m.get('ar',0))}">{fp(m.get('ar',0))}</div>
      </div>
      <div class="metric-box">
        <div class="metric-label">Ann Vol</div>
        <div class="metric-value">{fp(m.get('av',0))}</div>
      </div>
      <div class="metric-box">
        <div class="metric-label">Sharpe</div>
        <div class="metric-value {c(m.get('sh',0))}">{m.get('sh',0):.2f}</div>
      </div>
      <div class="metric-box">
        <div class="metric-label">Max DD (peak→trough)</div>
        <div class="metric-value neg">{fp(m.get('dd',0))}</div>
      </div>
      <div class="metric-box">
        <div class="metric-label">Hit Rate</div>
        <div class="metric-value">{fp(m.get('hr',0))}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_curve(bt: dict, key: str = ""):
    if not bt:
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=bt["dates"], y=bt["sc"].values,
        name=f"SAMBA ({bt['pick']})",
        line=dict(color="#7c3aed", width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=bt["dates"], y=bt["bc"].values,
        name=bt["benchmark"],
        line=dict(color="#9ca3af", width=1.5, dash="dot"),
    ))
    fig.update_layout(
        height=290, margin=dict(l=0, r=0, t=8, b=0),
        paper_bgcolor="white", plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0, font=dict(size=13)),
        xaxis=dict(showgrid=True, gridcolor="#f3f4f6", tickfont=dict(size=12)),
        yaxis=dict(showgrid=True, gridcolor="#f3f4f6", tickfont=dict(size=12),
                   tickformat=".2f"),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True,
                    config={"displayModeBar": False}, key=f"curve_{key}")


def render_footnote(signal: dict, window: bool = False):
    if not signal:
        return
    trained = signal.get("trained_at", "—")
    try:
        trained = datetime.fromisoformat(trained).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        pass

    if window:
        wid  = signal.get("winning_window", "?")
        ws   = signal.get("winning_train_start", "?")
        we   = signal.get("winning_train_end", "?")
        loss = signal.get("winning_loss", "—")
        ret  = signal.get("oos_ann_return", 0)
        shr  = signal.get("oos_sharpe", 0)
        detail = (f"Window {wid} ({ws}→{we}) &nbsp;·&nbsp; Loss: {loss} &nbsp;·&nbsp; "
                  f"OOS Return: {ret*100:.2f}% &nbsp;·&nbsp; Sharpe: {shr:.3f}")
    else:
        loss = signal.get("winning_loss", "—")
        ret  = signal.get("test_ann_return", 0)
        shr  = signal.get("test_sharpe", 0)
        start = signal.get("test_start", "")
        detail = (f"Loss: {loss} &nbsp;·&nbsp; "
                  f"Test Return: {ret*100:.2f}% &nbsp;·&nbsp; Sharpe: {shr:.3f}"
                  + (f" &nbsp;·&nbsp; Test from: {start}" if start else ""))

    st.markdown(
        f"<div class='fn'>Trained {trained} &nbsp;·&nbsp; {detail}</div>",
        unsafe_allow_html=True,
    )


def render_history(hist_df: pd.DataFrame, master: pd.DataFrame):
    if hist_df.empty:
        st.info("Signal history will appear after the first training run.")
        return

    if "actual_return" not in hist_df.columns and not master.empty:
        def get_ret(row):
            try:
                date = pd.Timestamp(row["signal_date"])
                col  = f"{row['pick']}_ret"
                if col in master.columns and date in master.index:
                    return master.loc[date, col]
            except Exception:
                pass
            return np.nan
        hist_df["actual_return"] = hist_df.apply(get_ret, axis=1)

    if "hit" not in hist_df.columns and "actual_return" in hist_df.columns:
        hist_df["hit"] = hist_df["actual_return"].apply(
            lambda x: "✓" if (not np.isnan(x) and x > 0)
                      else ("✗" if not np.isnan(x) else "—")
        )

    disp = hist_df.sort_values("signal_date", ascending=False).copy()
    col_map = {
        "signal_date":   "Date",
        "pick":          "Pick",
        "conviction":    "Conviction",
        "actual_return": "Actual Return",
        "hit":           "Hit",
    }
    cols = [c for c in col_map if c in disp.columns]
    disp = disp[cols].rename(columns=col_map)

    if "Conviction" in disp.columns:
        disp["Conviction"] = disp["Conviction"].apply(lambda x: f"{x*100:.1f}%")
    if "Actual Return" in disp.columns:
        disp["Actual Return"] = disp["Actual Return"].apply(
            lambda x: f"{x*100:.2f}%" if not np.isnan(x) else "—"
        )

    if "Hit" in disp.columns:
        hits  = (disp["Hit"] == "✓").sum()
        total = disp["Hit"].isin(["✓", "✗"]).sum()
        hr    = hits / total if total > 0 else 0
        st.markdown(
            f"<div class='hit-line'>Hit rate: <b>{hr:.1%}</b>"
            f" &nbsp;({hits}/{total} signals)</div>",
            unsafe_allow_html=True,
        )

    st.dataframe(disp, use_container_width=True, hide_index=True)


# ── Option renderer ────────────────────────────────────────────────────────────

def render_option(option: str, signals: dict, master: pd.DataFrame):
    sig  = signals.get(option,        {})
    sigw = signals.get(f"{option}w",  {})
    hist = load_history(option)

    # Hero — best of fixed split vs shrinking window
    render_hero(sig, sigw, option)

    # Compute periods
    n_total    = len(master) if not master.empty else 4582
    n_test     = int(n_total * (1 - cfg.TRAIN_SPLIT - cfg.VAL_SPLIT))
    test_start = sig.get("test_start") or \
                 (str(master.index[-n_test].date()) if not master.empty else "2023-01-01")
    oos_start  = cfg.LIVE_START
    oos_end    = str(master.index[-1].date()) if not master.empty else "today"

    # Metrics from signal JSON (no app-side recompute)
    fw_metrics = {
        "ar": sig.get("test_ann_return", 0),
        "av": 0,   # not stored in fixed split meta yet
        "sh": sig.get("test_sharpe", 0),
        "dd": 0,
        "hr": 0,
    }
    sw_metrics = {
        "ar": sigw.get("oos_ann_return", 0),
        "av": sigw.get("oos_ann_vol", 0),
        "sh": sigw.get("oos_sharpe", 0),
        "dd": sigw.get("oos_max_dd", 0),
        "hr": sigw.get("oos_hit_rate", 0),
    }

    bt_f = build_bt(sig.get("pick",  ""), master, option, start_date=test_start)
    bt_w = build_bt(sigw.get("pick", ""), master, option, start_date=oos_start)

    col_f, col_w = st.columns(2, gap="large")

    with col_f:
        st.markdown("<div class='label-fixed'>Fixed Split (70/15/15)</div>",
                    unsafe_allow_html=True)
        st.markdown(
            f"<div class='period-badge'>Test: {test_start} → {oos_end}</div>",
            unsafe_allow_html=True,
        )
        render_metrics(fw_metrics)
        render_curve(bt_f, key=f"{option}_fixed")
        render_footnote(sig, window=False)

    with col_w:
        st.markdown("<div class='label-window'>Shrinking Window</div>",
                    unsafe_allow_html=True)
        if sigw and "winning_window" in sigw:
            st.markdown(
                f"<div class='window-badge'>"
                f"Window {sigw['winning_window']}: "
                f"{sigw.get('winning_train_start','?')} → {sigw.get('winning_train_end','?')}"
                f" &nbsp;·&nbsp; OOS: {oos_start} → {oos_end}"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div class='window-badge' style='visibility:hidden;'>placeholder</div>",
                unsafe_allow_html=True,
            )
        render_metrics(sw_metrics)
        render_curve(bt_w, key=f"{option}_window")
        render_footnote(sigw, window=True)

    # Signal history
    st.markdown("<div class='sec-hdr'>Signal History</div>",
                unsafe_allow_html=True)
    render_history(hist, master)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    st.markdown(
        "<h2 style='margin-bottom:2px;color:#1a1a2e;font-size:34px;'>"
        "SAMBA — Graph-Mamba ETF Engine</h2>"
        "<p style='color:#6b7280;font-size:16px;margin-top:0;'>"
        "Selective State Space &nbsp;·&nbsp; Dynamic Graph Adjacency &nbsp;·&nbsp; "
        "EVaR + Sharpe objective</p>",
        unsafe_allow_html=True,
    )

    with st.spinner("Loading signals and data..."):
        signals = load_signals()
        master  = load_master()

    tab_a, tab_b = st.tabs([
        "🌊  Option A — Fixed Income / Alts",
        "🌊  Option B — Equity Sectors",
    ])

    with tab_a:
        render_option("A", signals, master)

    with tab_b:
        render_option("B", signals, master)

    st.markdown(
        "<div style='margin-top:40px;padding-top:16px;border-top:1px solid #e5e7eb;"
        "font-size:13px;color:#9ca3af;text-align:center;'>"
        "P2-ETF-SAMBA-ENGINE &nbsp;·&nbsp; Research only &nbsp;·&nbsp; Not financial advice"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
