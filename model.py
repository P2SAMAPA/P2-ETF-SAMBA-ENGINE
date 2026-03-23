# model.py — SAMBA Graph-Mamba Architecture
#
# Architecture:
#   x_asset  (B, n_assets, lookback, n_asset_feats)
#   x_macro  (B, lookback, n_macro_feats)
#       ↓
#   MambaAssetEncoder  — per-asset Mamba SSM (replaces LSTM in DeePM)
#       → asset_emb    (B, n_assets, D_MODEL)
#   MacroEncoder       — linear projection + mean pool
#       → macro_ctx    (B, MACRO_HIDDEN_DIM)
#   DynamicGraphLayer  — learned adjacency + multi-head attention
#       → graph_emb    (B, n_assets, GRAPH_HIDDEN_DIM)
#   PortfolioHead      — MLP → softmax over n_assets
#       → weights      (B, n_assets)

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg


# ── Mamba Block ────────────────────────────────────────────────────────────────

class MambaBlock(nn.Module):
    """
    Simplified Mamba-style selective state space block.
    Full Mamba uses hardware-aware CUDA kernels; this is a clean
    PyTorch implementation suitable for CPU/small GPU training.

    Input:  (B, L, D)
    Output: (B, L, D)
    """

    def __init__(self, d_model: int, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d_model  = d_model
        self.d_inner  = d_model * expand
        self.d_state  = d_state
        self.d_conv   = d_conv

        # Input projection
        self.in_proj  = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Causal depthwise conv
        self.conv1d   = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )

        # SSM parameters
        self.x_proj   = nn.Linear(self.d_inner, d_state * 2 + self.d_inner, bias=False)
        self.dt_proj  = nn.Linear(self.d_inner, self.d_inner, bias=True)

        # A matrix (log parameterised for stability)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log    = nn.Parameter(torch.log(A))
        self.D        = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm     = nn.LayerNorm(d_model)
        self.dropout  = nn.Dropout(dropout)

    def ssm(self, x: torch.Tensor) -> torch.Tensor:
        """Selective state space model scan."""
        B, L, D = x.shape
        A = -torch.exp(self.A_log)                           # (D, N)

        x_dbl = self.x_proj(x)                              # (B, L, N*2+D)
        delta, B_param, C = x_dbl.split(
            [self.d_inner, self.d_state, self.d_state], dim=-1
        )
        delta  = F.softplus(self.dt_proj(delta))            # (B, L, D)

        # Discretise A and B
        dA     = torch.exp(delta.unsqueeze(-1) * A)         # (B, L, D, N)
        dB     = delta.unsqueeze(-1) * B_param.unsqueeze(2) # (B, L, D, N)

        # Sequential scan (efficient for short sequences)
        h = torch.zeros(B, D, self.d_state, device=x.device)
        ys = []
        for i in range(L):
            h = dA[:, i] * h + dB[:, i] * x[:, i:i+1, :].transpose(1, 2)
            y = (h * C[:, i:i+1, :].unsqueeze(1)).sum(-1)  # (B, D)
            ys.append(y)
        y = torch.stack(ys, dim=1)                          # (B, L, D)
        return y + x * self.D.unsqueeze(0).unsqueeze(0)    # broadcast (1,1,D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_norm   = self.norm(x)

        # Split into two branches
        xz       = self.in_proj(x_norm)
        x_br, z  = xz.chunk(2, dim=-1)

        # Causal conv
        x_conv   = self.conv1d(x_br.transpose(1, 2))[..., :x_br.shape[1]].transpose(1, 2)
        x_conv   = F.silu(x_conv)

        # SSM
        y        = self.ssm(x_conv)

        # Gate
        y        = y * F.silu(z)

        # Output
        out      = self.out_proj(y)
        return self.dropout(out) + residual


# ── Asset Encoder (Mamba) ──────────────────────────────────────────────────────

class MambaAssetEncoder(nn.Module):
    """
    Encodes each asset's time series independently using stacked Mamba blocks.
    Input:  (B, n_assets, L, n_feats)
    Output: (B, n_assets, D_MODEL)  — last timestep representation
    """

    def __init__(self, n_feats: int, d_model: int, n_layers: int = 2,
                 d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Linear(n_feats, d_model)
        self.layers     = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand, dropout)
            for _ in range(n_layers)
        ])
        self.norm       = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, A, L, F = x.shape
        x = x.view(B * A, L, F)
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x[:, -1, :])          # take last timestep
        return x.view(B, A, -1)             # (B, A, D_MODEL)


# ── Macro Encoder ──────────────────────────────────────────────────────────────

class MacroEncoder(nn.Module):
    """
    Encodes macro time series into a context vector.
    Input:  (B, L, n_macro_feats)
    Output: (B, macro_hidden_dim)
    """

    def __init__(self, n_macro_feats: int, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(n_macro_feats, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)            # (B, L, hidden_dim)
        x = x.mean(dim=1)           # mean pool over time
        return self.norm(x)         # (B, hidden_dim)


# ── Dynamic Graph Layer ────────────────────────────────────────────────────────

class DynamicGraphLayer(nn.Module):
    """
    Learns dynamic asset adjacency conditioned on macro context.
    Uses multi-head attention as graph aggregation.

    Input:  asset_emb  (B, A, D_MODEL)
            macro_ctx  (B, macro_hidden_dim)
    Output: (B, A, graph_hidden_dim)
    """

    def __init__(self, d_model: int, macro_hidden_dim: int,
                 graph_hidden_dim: int, n_heads: int = 2, dropout: float = 0.2):
        super().__init__()

        # Project macro context to modulate graph attention
        self.macro_gate = nn.Sequential(
            nn.Linear(macro_hidden_dim, d_model),
            nn.Sigmoid(),
        )

        # Multi-head self-attention as graph aggregation
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Output projection
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, graph_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(graph_hidden_dim)

    def forward(self, asset_emb: torch.Tensor,
                macro_ctx: torch.Tensor) -> torch.Tensor:
        # Macro-conditioned gating
        gate = self.macro_gate(macro_ctx).unsqueeze(1)  # (B, 1, D_MODEL)
        x    = asset_emb * gate                          # (B, A, D_MODEL)

        # Graph attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out                                 # residual

        # Project to graph hidden dim
        out = self.out_proj(x)
        return self.norm2(out)                           # (B, A, graph_hidden_dim)


# ── Portfolio Head ─────────────────────────────────────────────────────────────

class PortfolioHead(nn.Module):
    """
    Maps graph embeddings + macro context to portfolio weights.
    Input:  graph_emb (B, A, graph_hidden_dim)
            macro_ctx (B, macro_hidden_dim)
    Output: weights   (B, A)  — softmax, long-only
    """

    def __init__(self, n_assets: int, graph_hidden_dim: int,
                 macro_hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        combined = graph_hidden_dim + macro_hidden_dim

        self.head = nn.Sequential(
            nn.Linear(combined, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_assets),
        )

    def forward(self, graph_emb: torch.Tensor,
                macro_ctx: torch.Tensor) -> torch.Tensor:
        # Pool graph embeddings and concat macro
        graph_pool = graph_emb.mean(dim=1)               # (B, graph_hidden_dim)
        combined   = torch.cat([graph_pool, macro_ctx], dim=-1)
        logits     = self.head(combined)                 # (B, n_assets)
        return F.softmax(logits, dim=-1)


# ── Full SAMBA Model ───────────────────────────────────────────────────────────

class SAMBA(nn.Module):
    """
    SAMBA: State-space Attention Mamba Bidirectional Adjacency model.

    Replaces DeePM's LSTM encoder with Mamba SSM blocks for better
    long-range sequence modelling with lower memory footprint.
    """

    def __init__(
        self,
        n_assets: int,
        n_asset_feats: int,
        n_macro_feats: int,
        d_model: int         = 64,
        d_state: int         = 16,
        d_conv: int          = 4,
        expand: int          = 2,
        n_mamba_layers: int  = 2,
        macro_hidden_dim: int = 32,
        graph_hidden_dim: int = 64,
        n_attn_heads: int    = 2,
        dropout: float       = 0.2,
    ):
        super().__init__()
        self.n_assets = n_assets

        self.asset_encoder = MambaAssetEncoder(
            n_feats=n_asset_feats,
            d_model=d_model,
            n_layers=n_mamba_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )
        self.macro_encoder = MacroEncoder(
            n_macro_feats=n_macro_feats,
            hidden_dim=macro_hidden_dim,
            dropout=dropout,
        )
        self.graph_layer = DynamicGraphLayer(
            d_model=d_model,
            macro_hidden_dim=macro_hidden_dim,
            graph_hidden_dim=graph_hidden_dim,
            n_heads=n_attn_heads,
            dropout=dropout,
        )
        self.portfolio_head = PortfolioHead(
            n_assets=n_assets,
            graph_hidden_dim=graph_hidden_dim,
            macro_hidden_dim=macro_hidden_dim,
            dropout=dropout,
        )

    def forward(self, x_asset: torch.Tensor,
                x_macro: torch.Tensor) -> torch.Tensor:
        asset_emb = self.asset_encoder(x_asset)    # (B, A, D_MODEL)
        macro_ctx = self.macro_encoder(x_macro)    # (B, macro_hidden_dim)
        graph_emb = self.graph_layer(asset_emb, macro_ctx)  # (B, A, graph_hidden_dim)
        weights   = self.portfolio_head(graph_emb, macro_ctx)  # (B, A)
        return weights


# ── Loss Functions ─────────────────────────────────────────────────────────────

def sharpe_loss(weights: torch.Tensor, returns: torch.Tensor,
                cash_rate: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    port_ret = (weights * returns).sum(dim=1)
    excess   = port_ret - cash_rate
    return -(excess.mean() / (excess.std() + eps)) * math.sqrt(252)


def evar_loss(weights: torch.Tensor, returns: torch.Tensor,
              cash_rate: torch.Tensor, beta: float = 0.95,
              eps: float = 1e-6) -> torch.Tensor:
    port_ret  = (weights * returns).sum(dim=1)
    excess    = port_ret - cash_rate
    mean_ret  = excess.mean()

    # EVaR: exponential CVaR approximation
    t         = torch.tensor(1.0, requires_grad=False)
    evar_val  = t * torch.log(
        torch.mean(torch.exp(-excess / (t + eps))) + eps
    ) + t * math.log(1.0 / (1.0 - beta))

    return evar_val - mean_ret * 0.5


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
