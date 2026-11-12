#!/usr/bin/env python3
"""
06_make_figures.py — Figures of the manuscript
==============================================

Figure 1: error structure (Allan deviation, sigma(c), decomposition).
Figure 2: resolution limit against instrument limit.
Figure 3: consequence for the annual PM2.5 mean and the ozone design statistic.
Figure S1: hourly against daily agreement.

The palette uses muted tones from Paul Tol, checked for lightness band, chroma
floor, colour-blind separation (worst adjacent pair dE 9.0 under
deuteranopia), normal-vision floor and contrast against the background. Colour
follows the **instrument family**, not the series order, so a family keeps its
colour across all figures.

Usage
-----
  python3 06_make_figures.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SURFACE = "#fcfcfb"
INK, INK2, GRID, AXIS = "#1a1a18", "#5c5b57", "#ededea", "#6f6e69"
# Fixed order. One colour per family, the same in every figure.
COL = {"T640 optical": "#3273b3", "Beta attenuation": "#CC6677",
       "SHARP": "#117733", "TEOM FDMS": "#882255",
       "TEOM/R&P": "#882255", "UV photometric": "#999933"}
# Display names where the label needs escaping for matplotlib mathtext.
NAME = {"TEOM/R&P": "TEOM/R\\&P"}
W1, W15, W2 = 3.46, 4.72, 7.09
RC = {"font.size": 8.0, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
      "xtick.labelsize": 7.2, "ytick.labelsize": 7.2, "legend.fontsize": 7.2,
      "figure.dpi": 400, "savefig.dpi": 400,
      "xtick.direction": "in", "ytick.direction": "in",
      "xtick.top": True, "ytick.right": True,
      "axes.edgecolor": AXIS, "axes.linewidth": 0.7,
      "text.color": INK, "axes.labelcolor": INK, "axes.titlecolor": INK,
      "xtick.color": AXIS, "ytick.color": AXIS,
      "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
      "savefig.facecolor": SURFACE, "legend.frameon": False,
      "lines.linewidth": 1.1, "font.family": "sans-serif"}


def clean(ax) -> None:
    """All four spines visible, recessive grid, minor ticks."""
    for s in ax.spines.values():
        s.set_visible(True)
    ax.grid(True, color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.minorticks_on()
    ax.tick_params(which="minor", length=1.8, color=AXIS)
    ax.tick_params(which="major", length=3.2, color=AXIS)


def panel(ax, letter: str, dy: float = 1.12) -> None:
    ax.text(0.0, dy, letter, transform=ax.transAxes, fontsize=9,
            fontweight="bold", color=INK, va="bottom", ha="left")


def fig01(A: pd.DataFrame, out: Path) -> None:
    """Dumbbell: hourly CV -> daily CV, by family and pollutant."""
    g = A[A.cv_hourly.notna() & A.cv_daily.notna()]
    T = g.groupby(["variable", "family"]).agg(
        n=("site", "size"), sites=("site", "nunique"),
        cvh=("cv_hourly", "median"), cvd=("cv_daily", "median"),
        # Median of the per-pair ratios, not ratio of the medians: each pair
        # averages in its own way, and that distribution is what is
        # summarised. It is also the figure printed by step 3.
        ratio=("ratio", "median")).reset_index()
    T = T[T.sites >= 2].sort_values("cvh")
    fig, ax = plt.subplots(figsize=(W15, 2.9))
    clean(ax)
    y = np.arange(len(T))
    for i, r in enumerate(T.itertuples()):
        c = COL.get(r.family, INK2)
        ax.plot([r.cvd, r.cvh], [i, i], color=c, lw=1.6, solid_capstyle="round",
                zorder=2)
        # 2-px ring in the background colour where markers overlap
        ax.plot(r.cvd, i, "o", ms=5.5, color=SURFACE, mec=c, mew=1.3, zorder=3)
        ax.plot(r.cvh, i, "o", ms=5.5, color=c, mec=SURFACE, mew=0.8, zorder=3)
        ax.text(r.cvh + 1.1, i, f"\u00d7{r.ratio:.2f}", va="center",
                fontsize=6.8, color=INK2)
    ax.set_yticks(y)
    POL = {"pm25": "PM$_{2.5}$", "pm10": "PM$_{10}$", "o3": "O$_3$"}
    ax.set_yticklabels([f"{NAME.get(r.family, r.family)} · "
                        f"{POL.get(r.variable, r.variable)} ({r.sites})"
                        for r in T.itertuples()])
    ax.set_xlabel("Coefficient of variation, upper bound (%)")
    ax.set_xlim(-1.5, 36)
    ax.set_ylim(-0.6, len(T) - 0.4)
    ax.plot([], [], "o", ms=5.5, color=INK2, mec=SURFACE, label="1 hour")
    ax.plot([], [], "o", ms=5.5, color=SURFACE, mec=INK2, mew=1.3,
            label="24 hours")
    ax.legend(loc="lower right", handletextpad=0.4)
    fig.tight_layout()
    fig.savefig(out / "figS1_hourly_vs_daily.png", bbox_inches="tight")
    plt.close(fig)


def fig02(R: pd.DataFrame, out: Path) -> None:
    """Resolution limit against instrument limit."""
    T = R.groupby(["variable", "family"]).agg(
        sites=("site", "nunique"), sd=("sd_steps", "median"),
        # The three fractions are averaged, not medians: three medians do not
        # sum to one and the stacked bar would fall short.
        h0=("frac_0", "mean"), h1=("frac_1", "mean"),
        h2=("frac_ge2", "mean")).reset_index()
    T = T[T.sites >= 2].sort_values("sd")
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.7),
                             gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    clean(ax)
    y = np.arange(len(T))
    for i, r in enumerate(T.itertuples()):
        ax.barh(i, r.sd, height=0.58, color=COL.get(r.family, INK2),
                edgecolor=SURFACE, lw=0.8, zorder=2)
        ax.text(r.sd * 1.14, i, f"{r.sd:.1f}", va="center", fontsize=6.8,
                color=INK2)
    # Two series without instrument error, rounded to the same increment, give
    # a difference with standard deviation 1/sqrt(6) increments (Sheppard).
    ax.axvline(1 / np.sqrt(6), color=INK, lw=0.9, ls=(0, (3, 2)), zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(0.25, 90)
    POL2 = {"pm25": "PM$_{2.5}$", "pm10": "PM$_{10}$", "o3": "O$_3$"}
    ax.set_yticks(y)
    ax.set_yticklabels([f"{NAME.get(r.family, r.family)} · "
                        f"{POL2.get(r.variable, r.variable)}"
                        for r in T.itertuples()])
    ax.set_xlabel("SD of paired difference (reporting increments)")
    ax.set_ylim(-0.6, len(T) + 0.25)
    ax.text(0.45, len(T) - 0.15, "rounding alone", ha="left",
            fontsize=6.6, color=INK2, style="italic", va="center")
    panel(ax, "(a)", 1.12)

    ax = axes[1]
    clean(ax)
    left = np.zeros(len(T))
    shades = [0.30, 0.62, 1.0]
    labels = ["0", "1", "$\\geq$2 increments"]
    for j, (col, lab) in enumerate(zip(["h0", "h1", "h2"], labels)):
        v = T[col].to_numpy()
        for i, r in enumerate(T.itertuples()):
            c = COL.get(r.family, INK2)
            ax.barh(i, v[i], left=left[i], height=0.58, color=c,
                    alpha=shades[j], edgecolor=SURFACE, lw=0.8, zorder=2)
        left += v
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of concurrent hours")
    ax.set_ylim(-0.6, len(T) + 0.25)
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=INK2, alpha=shades[j], edgecolor=SURFACE,
                     label=lab) for j, lab in enumerate(labels)]
    ax.legend(handles=handles, loc="lower right",
              bbox_to_anchor=(1.0, 1.0), ncol=3, handlelength=1.0,
              handletextpad=0.3, columnspacing=0.8, borderaxespad=0.2)
    panel(ax, "(b)", 1.12)
    fig.tight_layout()
    fig.savefig(out / "fig02_resolution_limit.png", bbox_inches="tight")
    plt.close(fig)


def fig03(D: pd.DataFrame, out: Path) -> None:
    """Consequence: PM2.5 annual mean (a) and ozone fourth-highest MDA8 (b)."""
    P = D[D.variable == "pm25"].copy()
    P["level"] = (P.annual_x + P.annual_y) / 2
    P["absdiff"] = P.annual_abs.abs()
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.9),
                             gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    clean(ax)
    ax.axvline(9.0, color=INK, lw=0.9, ls=(0, (3, 2)), zorder=3)
    for f, g in P.groupby("family"):
        c = COL.get(f, INK2)
        ax.plot(g.level, g.absdiff, "o", ms=4.6, color=c, mec=SURFACE,
                mew=0.6, alpha=0.9, zorder=2,
                label=f"{NAME.get(f, f)} ({g.site.nunique()})")
    dis = P[P.disagree_annual_9 == 1]
    ax.plot(dis.level, dis.absdiff, "o", ms=9.5, mfc="none", mec=INK,
            mew=0.9, zorder=4, label="disagree on 9 $\\mu$g m$^{-3}$")
    ax.set_xlabel("Pair mean annual PM$_{2.5}$ ($\\mu$g m$^{-3}$)")
    ax.set_ylabel("Annual mean difference ($\\mu$g m$^{-3}$)")
    ax.set_xlim(3.5, 14.5)
    ax.set_ylim(-0.1, 3.5)
    ax.text(9.15, 3.35, "9 $\\mu$g m$^{-3}$", fontsize=6.8, color=INK2,
            ha="left", va="center")
    ax.legend(loc="upper left", handletextpad=0.4)
    panel(ax, "(a)", 1.03)

    ax = axes[1]
    clean(ax)
    O = D[(D.variable == "o3") & D.mda8_4th_x.notna()].copy()
    O["level"] = (O.mda8_4th_x + O.mda8_4th_y) / 2 * 1000
    O["absdiff"] = O.mda8_4th_abs.abs() * 1000
    # Values are whole ppb: a small jitter keeps coincident markers from
    # hiding one another.
    jit = np.random.default_rng(0).uniform(-0.12, 0.12, len(O))
    c = COL["UV photometric"]
    ax.axvline(70.0, color=INK, lw=0.9, ls=(0, (3, 2)), zorder=3)
    ax.plot(O.level, O.absdiff + jit, "o", ms=4.2, color=c, mec=SURFACE,
            mew=0.6, alpha=0.9, zorder=2,
            label=f"UV photometric ({O.site.nunique()})")
    dis = O[O.disagree_mda8_4th == 1]
    ax.plot(dis.level, dis.absdiff + jit[O.disagree_mda8_4th.to_numpy() == 1],
            "o", ms=9, mfc="none", mec=INK, mew=0.9, zorder=4,
            label="disagree on 70 ppb")
    ax.set_xlabel("Pair fourth-highest MDA8 O$_3$ (ppb)")
    ax.set_ylabel("Difference (ppb)")
    ax.set_ylim(-0.5, 7)
    ax.legend(loc="upper left", handletextpad=0.4)
    panel(ax, "(b)", 1.03)
    fig.tight_layout()
    fig.savefig(out / "fig03_annual_consequence.png", bbox_inches="tight")
    plt.close(fig)


def fig04(AL: pd.DataFrame, BN: pd.DataFrame, FT: pd.DataFrame,
          SM: pd.DataFrame, out: Path) -> None:
    """Error structure: Allan deviation, sigma(c) and decomposition."""
    fams = [("pm25", "T640 optical"), ("pm25", "SHARP"),
            ("pm25", "Beta attenuation"), ("o3", "UV photometric")]
    POL = {"pm25": "PM$_{2.5}$", "o3": "O$_3$"}
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.65),
                             gridspec_kw={"width_ratios": [1, 1, 0.9]})

    # (a) Allan deviation normalised by its value at 1 h: shape, no units
    ax = axes[0]
    clean(ax)
    taus = np.array(sorted(AL.tau.unique()), dtype=float)
    ax.plot(taus, taus ** -0.5, color=INK2, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.text(120, 120 ** -0.5 * 1.35, r"$\tau^{-1/2}$", fontsize=7,
            color=INK2, ha="center")
    for var, fam in fams:
        g = AL[(AL.variable == var) & (AL.family == fam)]
        m = g.groupby("tau").adev.median()
        ax.plot(m.index, m / m.loc[1], "o-", ms=2.8, color=COL[fam],
                mec=SURFACE, mew=0.4, label=f"{NAME.get(fam, fam)} · {POL[var]}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Averaging time $\tau$ (h)")
    ax.set_ylabel(r"Allan deviation / value at 1 h")
    ax.set_ylim(0.02, 1.6)
    ax.legend(loc="lower left", handletextpad=0.4, fontsize=6.0,
              borderaxespad=0.3, labelspacing=0.25)
    panel(ax, "(a)", 1.03)

    # (b) sigma per instrument against concentration, PM2.5
    ax = axes[1]
    clean(ax)
    cgrid = np.linspace(0.5, 40, 200)
    for var, fam in fams[:3]:
        g = BN[(BN.variable == var) & (BN.family == fam)]
        cur = g.groupby(["lo", "hi"]).agg(c=("c", "median"),
                                          sd=("sd", "median"),
                                          q1=("sd", lambda s: s.quantile(.25)),
                                          q3=("sd", lambda s: s.quantile(.75)),
                                          n=("sd", "size")).reset_index()
        cur = cur[cur.n >= 3]
        col = COL[fam]
        ax.errorbar(cur.c, cur.sd, yerr=[cur.sd - cur.q1, cur.q3 - cur.sd],
                    fmt="o", ms=3.2, color=col, mec=SURFACE, mew=0.4,
                    elinewidth=0.7, capsize=0, zorder=3)
        f = FT[(FT.variable == var) & (FT.family == fam)].iloc[0]
        ax.plot(cgrid, np.sqrt(f.a ** 2 + (f.b * cgrid) ** 2), color=col,
                lw=1.0, zorder=2)
    ax.set_xlabel(r"Concentration ($\mu$g m$^{-3}$)")
    ax.set_ylabel(r"SD per instrument ($\mu$g m$^{-3}$)")
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 4.6)
    panel(ax, "(b)", 1.03)

    # (c) share of the mean square of the hourly difference
    ax = axes[2]
    clean(ax)
    ax.grid(False, axis="y")
    shares = []
    for var, fam in fams:
        g = SM[(SM.variable == var) & (SM.family == fam)]
        b2 = np.median(g.bias ** 2)
        s2 = np.median(g.sd_slow ** 2)
        f2 = np.median(g.sd_fast ** 2)
        tot = b2 + s2 + f2
        shares.append((b2 / tot, s2 / tot, f2 / tot))
    shares = np.array(shares)
    left = np.zeros(len(fams))
    alph = [1.0, 0.62, 0.30]
    labs = ["offset", "slow", "fast"]
    for j in range(3):
        for i, (var, fam) in enumerate(fams):
            ax.barh(i, shares[i, j], left=left[i], height=0.6,
                    color=COL[fam], alpha=alph[j], edgecolor=SURFACE,
                    lw=0.8, zorder=2)
        left += shares[:, j]
    ax.set_yticks(range(len(fams)))
    ax.set_yticklabels([f"{NAME.get(f, f)}\n{POL[v]}" for v, f in fams],
                       fontsize=6.6)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(fams) - 0.4)
    ax.invert_yaxis()
    ax.set_xlabel("Share of mean sq. difference")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=INK2, alpha=alph[j], edgecolor=SURFACE,
                             label=labs[j]) for j in range(3)],
              loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3,
              fontsize=6.0, handlelength=0.9, handletextpad=0.3,
              columnspacing=0.6, borderaxespad=0.2)
    panel(ax, "(c)", 1.03)
    fig.tight_layout(w_pad=0.8)
    fig.savefig(out / "fig01_error_structure.png", bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> int:
    here = Path(__file__).resolve().parent
    root = here.parent
    p = argparse.ArgumentParser(description="Figures of the manuscript.")
    p.add_argument("--res-dir", type=Path,
                   default=root / "Results_Collocated")
    p.add_argument("--designations", type=Path,
                   default=root / "Data_Collocated"
                   / "method_designations.csv")
    p.add_argument("--fig-dir", type=Path,
                   default=root / "Figures_Collocated")
    a = p.parse_args(argv)
    a.fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(RC)
    plt.rcParams["text.usetex"] = False

    F = pd.read_csv(a.designations, dtype=str)
    F["method_code"] = F.method_code.str.zfill(3)
    fam = dict(zip(F.method_code, F.family))
    A = pd.read_parquet(a.res_dir / "agreement_precision.parquet")
    A["family"] = A.method.map(fam)
    fig01(A, a.fig_dir)
    fig02(pd.read_parquet(a.res_dir / "resolution_limit.parquet"), a.fig_dir)
    fig03(pd.read_parquet(a.res_dir / "derived.parquet"), a.fig_dir)
    if (a.res_dir / "error_allan.parquet").exists():
        fig04(pd.read_parquet(a.res_dir / "error_allan.parquet"),
              pd.read_parquet(a.res_dir / "error_bins.parquet"),
              pd.read_parquet(a.res_dir / "error_model_fit.parquet"),
              pd.read_parquet(a.res_dir / "error_summary.parquet"),
              a.fig_dir)
    for f in sorted(a.fig_dir.glob("fig*.png")):
        print(f"  {f.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
