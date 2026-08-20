"""Standard plotly figures for a factor analysis -- the "tearsheet".

The same five views for every factor, so the eye learns where to look:
quantile spread, L/S equity and drawdown, binned monotonicity, rolling Sharpe,
and the parameter grid.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLORS_Q = ["#c0392b", "#e67e22", "#7f8c8d", "#27ae60", "#2980b9"]
ACCENT = "#8e44ad"


def fig_ls_focus(qr, title="", periods_per_year=365):
    """L/S focus: LSiv (rank x inverse-vol) vs LS (rank only), log equity plus a
    drawdown panel. Headline stats for LSiv in the title."""
    r_iv, r_ls = qr["LSiv"].dropna(), qr["LS"].dropna()
    eq_iv, eq_ls = (1 + r_iv).cumprod(), (1 + r_ls).cumprod()
    dd = (eq_iv / eq_iv.cummax() - 1) * 100
    ann = r_iv.mean() * periods_per_year * 100
    vol = r_iv.std() * np.sqrt(periods_per_year) * 100
    sh = r_iv.mean() / r_iv.std() * np.sqrt(periods_per_year) if r_iv.std() > 0 else np.nan
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.75, 0.25], vertical_spacing=0.04)
    fig.add_trace(go.Scatter(x=eq_iv.index, y=eq_iv, mode="lines",
                             name="LSiv (rank x inv-vol)",
                             line=dict(color=ACCENT, width=2.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=eq_ls.index, y=eq_ls, mode="lines", name="LS (rank only)",
                             line=dict(color="black", width=1.2, dash="dot")), row=1, col=1)
    fig.add_hline(y=1, line_color="gray", line_dash="dot", row=1, col=1)
    fig.add_trace(go.Scatter(x=dd.index, y=dd, mode="lines", fill="tozeroy",
                             line=dict(color="#c0392b", width=1), showlegend=False),
                  row=2, col=1)
    fig.update_yaxes(type="log", title_text="equity (base 1, log)", row=1, col=1)
    fig.update_yaxes(title_text="LSiv drawdown (%)", row=2, col=1)
    fig.update_layout(
        title=f"{title}, L/S rank x inverse-vol | ann={ann:+.1f}%  vol={vol:.1f}%  "
              f"sharpe={sh:.2f}  maxDD={dd.min():.1f}%",
        height=560, template="plotly_white")
    return fig


def fig_quantile_equity(qr, title=""):
    """Equity per quantile + LS + LSiv. Monotone ordering of Q1..Q5 is the
    visual signature of a factor that is not driven by one bucket."""
    fig = go.Figure()
    qcols = [c for c in qr.columns if c.startswith("Q")]
    for k, c in enumerate(qcols):
        r = qr[c].fillna(0)
        fig.add_trace(go.Scatter(x=r.index, y=(1 + r).cumprod(), mode="lines",
                                 line=dict(color=COLORS_Q[k % len(COLORS_Q)], width=1.5), name=c))
    for c, color, dash in [("LS", "black", "solid"), ("LSiv", ACCENT, "dash")]:
        if c in qr:
            r = qr[c].fillna(0)
            fig.add_trace(go.Scatter(x=r.index, y=(1 + r).cumprod(), mode="lines",
                                     line=dict(color=color, width=2.5, dash=dash), name=c))
    fig.add_hline(y=1, line_color="gray", line_dash="dot")
    fig.update_layout(title=title, yaxis_type="log",
                      yaxis_title="equity (base 1, log)", height=480,
                      template="plotly_white")
    return fig


def fig_alpha_by_quantile(qr, mkt, title=""):
    """Cumulative demeaned alpha per quantile -- strips out the market leg so
    you see the cross-sectional spread on its own."""
    fig = go.Figure()
    qcols = [c for c in qr.columns if c.startswith("Q")]
    for k, c in enumerate(qcols):
        a = (qr[c] - mkt).fillna(0)
        fig.add_trace(go.Scatter(x=a.index, y=(1 + a).cumprod(), mode="lines",
                                 line=dict(color=COLORS_Q[k % len(COLORS_Q)], width=1.8), name=c))
    fig.add_hline(y=1, line_color="gray", line_dash="dot")
    fig.update_layout(title=title, yaxis_type="log",
                      yaxis_title="growth of 1.0 (log)", height=480,
                      template="plotly_white")
    return fig


def fig_binned(curves, title="", n_bins=10):
    """Binned percentile -> forward return; curves = {label: Series per bin}."""
    fig = go.Figure()
    for label, g in curves.items():
        fig.add_trace(go.Scatter(x=(g.index + 0.5) * (100 / n_bins), y=g.values,
                                 mode="lines+markers", name=label))
    fig.add_hline(y=0, line_color="gray")
    fig.update_layout(title=title, xaxis_title="cross-sectional factor percentile",
                      yaxis_title="mean relative forward return (%)", height=420,
                      template="plotly_white")
    return fig


def fig_rolling_sharpe(r, ref_px=None, window=90, title="", periods_per_year=365):
    """Rolling annualised Sharpe + a reference price on a log overlay, to see
    whether the edge is really regime-dependent or just noisy."""
    r = r.dropna()
    rs = r.rolling(window).mean() / r.rolling(window).std() * np.sqrt(periods_per_year)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=rs.index, y=rs, mode="lines", name=f"sharpe {window}p",
                             line=dict(color=ACCENT, width=2)))
    fig.add_hline(y=0, line_color="gray", line_dash="dot")
    if ref_px is not None:
        ref = ref_px.reindex(rs.index)
        fig.add_trace(go.Scatter(x=ref.index, y=ref, mode="lines",
                                 name=getattr(ref_px, "name", None) or "ref",
                                 line=dict(color="#f39c12", width=1)),
                      secondary_y=True)
        fig.update_yaxes(type="log", title_text="reference price (log)",
                         secondary_y=True, showgrid=False)
    fig.update_yaxes(title_text=f"rolling {window}-step sharpe", secondary_y=False)
    fig.update_layout(title=title, height=420, template="plotly_white")
    return fig


def fig_regime(r, cond, n_bins=3, labels=None, title="", cond_name="regime"):
    """PnL by regime: equity 'gated' to days where cond sits in each bucket
    (flat otherwise), with the conditioning variable below, coloured by bucket."""
    both = pd.concat({"r": r, "c": cond}, axis=1).dropna()
    b = pd.qcut(both["c"], n_bins, labels=False, duplicates="drop")
    n_eff = int(b.max()) + 1
    names = labels if labels and len(labels) == n_eff \
        else [f"T{k+1}" for k in range(n_eff)]
    colors = ["#c0392b", "#7f8c8d", "#27ae60", "#2980b9"]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.4], vertical_spacing=0.05)
    eq_all = (1 + both["r"]).cumprod()
    fig.add_trace(go.Scatter(x=eq_all.index, y=eq_all, mode="lines", name="total",
                             line=dict(color="black", width=1, dash="dot")),
                  row=1, col=1)
    for k in range(n_eff):
        color = colors[k % len(colors)]
        eq = (1 + both["r"].where(b == k, 0)).cumprod()
        fig.add_trace(go.Scatter(x=eq.index, y=eq, mode="lines",
                                 name=f"{cond_name} {names[k]}",
                                 line=dict(color=color, width=2)), row=1, col=1)
        seg = both["c"].where(b == k)
        fig.add_trace(go.Scatter(x=seg.index, y=seg, mode="lines", showlegend=False,
                                 line=dict(color=color, width=1.5),
                                 connectgaps=False), row=2, col=1)
    fig.add_hline(y=1, line_color="gray", line_dash="dot", row=1, col=1)
    fig.update_yaxes(type="log", title_text="gated equity (base 1, log)", row=1, col=1)
    fig.update_yaxes(title_text=cond_name, row=2, col=1)
    fig.update_layout(title=title, height=620, template="plotly_white")
    return fig


def fig_grid_heatmap(grid, title="", x_prefix="EMA"):
    """Sharpe grid heatmap -- parameter robustness at a glance."""
    z = grid.values.astype(float)
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"{x_prefix}{s}" for s in grid.columns], y=grid.index.tolist(),
        colorscale="RdYlGn", zmid=0, text=np.round(z, 2), texttemplate="%{text}",
        colorbar=dict(title="sharpe")))
    fig.update_layout(title=title, height=380, width=820, template="plotly_white")
    return fig
