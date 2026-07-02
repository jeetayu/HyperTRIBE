#!/usr/bin/env python3
"""
PUF60 Mutant vs WT — Multi-angle editing change visualizations

Four complementary views of how editing changes in S161 and S206 mutants:

  Figure 1 (wt_vs_mutant_site_freq.pdf)
      Site-frequency scatter — for sites shared by all 3 conditions, plot WT
      edit_freq vs mutant edit_freq. ~80% of points above the diagonal.

  Figure 2 (editing_freq_distribution.pdf)
      Density / violin of per-site edit_freq across all three conditions.
      Three sub-panels: all sites, shared sites only, new mutant sites only.

  Figure 3 (ma_plot.pdf)
      Delta plot at gene level: x = WT site count, y = Δ sites (mut − WT).
      Shows absolute editing gains concentrated at already highly-bound genes.

  Figure 4 (waterfall_top_genes.pdf)
      Ranked waterfall of genes by absolute delta (sites gained/lost).
      Coloured by CH/leukemia category.

  Figure 5 (site_composition_bars.pdf)
      Stacked bar: fraction of sites shared vs condition-unique, per condition.
      Quantifies how much of each dataset is "new" vs conserved.

  Figure 6 (ch_gene_heatmap.pdf)
      Gene × condition heatmap (WT / S161 / S206) of n_sites with colour
      scale diverging from WT — for all CH/leukemia genes with ≥5 sites.

Usage (via sbatch wrapper):
  python plot_editing_changes.py \\
    --wt   .../wt_concat_strict/annotated_editing_sites.bed \\
    --s161 .../s161_concat_strict/annotated_editing_sites.bed \\
    --s206 .../s206_concat_strict/annotated_editing_sites.bed \\
    --outdir figures/
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "svg.fonttype": "none",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "figure.dpi": 150, "savefig.dpi": 300,
})

# Colour palette
PALETTE = {"WT": "#2980b9", "S161": "#e74c3c", "S206": "#27ae60"}

# CH / leukemia gene sets
CH_CORE = {
    "TET2", "DNMT3A", "ASXL1", "JAK2", "MPL", "CALR", "TP53",
    "PPM1D", "ATM", "CHEK2", "SRSF2", "U2AF1", "ZRSR2", "SF3B1",
}
CH_MYELOID = {
    "CBL", "KMT2A", "RUNX1", "CEBPA", "ETV6", "GATA2", "FLT3", "KIT",
    "NRAS", "KRAS", "PTPN11", "IDH1", "IDH2", "EZH2", "KDM6A",
    "STAG2", "RAD21", "SMC1A", "SMC3", "SETBP1", "BCORL1",
    "CCND1", "CCND2", "CCND3", "RB1", "PTEN", "NF1", "NF2", "ATM",
    "GNB1", "GNAS", "STAT3",
}
PUF60_KNOWN = {
    "QSER1", "TEAD1", "NF2", "DCAF7", "DYRK1A", "DYRK1B",
    "EIF4EBP2", "LARP1", "SP1", "CBL", "MAP1B",
}
ALL_HIGHLIGHTED = CH_CORE | CH_MYELOID | PUF60_KNOWN

def gene_color(g):
    if g in CH_CORE:    return "#c0392b"
    if g in CH_MYELOID: return "#e67e22"
    if g in PUF60_KNOWN: return "#2980b9"
    return "#95a5a6"


# ── Data loading ───────────────────────────────────────────────────────────────
def load_full(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", comment=None, low_memory=False)
    df.columns = [c.lstrip("#") for c in df.columns]
    df["site_id"] = df["chr"].astype(str) + ":" + df["start"].astype(str)
    return df


def load_pc_only(path: str) -> pd.DataFrame:
    df = load_full(path)
    return df[(df["gene_type"] == "protein_coding") &
              (~df["gene"].isin([".", ""]))].copy()


def gene_stats(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby("gene")
              .agg(n_sites=("gene", "count"),
                   median_freq=("edit_freq", "median"),
                   mean_freq=("edit_freq", "mean"))
              .reset_index())


# ── Figure 1: Site-frequency scatter at shared sites ─────────────────────────
def plot_site_freq_scatter(ax, wt_df, mut_df, mut_label, shared_sites,
                           sample_n=60_000, seed=42):
    rng = np.random.default_rng(seed)

    wt_freq  = (wt_df[wt_df["site_id"].isin(shared_sites)]
                .groupby("site_id")["edit_freq"].mean())
    mut_freq = (mut_df[mut_df["site_id"].isin(shared_sites)]
                .groupby("site_id")["edit_freq"].mean())
    merged = wt_freq.rename("wt").to_frame().join(mut_freq.rename("mut"), how="inner")

    # Subsample for rendering speed if large
    if len(merged) > sample_n:
        merged = merged.sample(n=sample_n, random_state=seed)

    frac_above = (merged["mut"] > merged["wt"]).mean()
    x, y = merged["wt"].values, merged["mut"].values

    # Density-aware coloring
    ax.scatter(x, y, s=2, alpha=0.18, color=PALETTE[mut_label], linewidths=0, rasterized=True)

    # Diagonal
    lim = max(x.max(), y.max()) * 1.05
    ax.plot([0, lim], [0, lim], "--", color="#7f8c8d", linewidth=0.8, alpha=0.7)

    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Edit frequency — WT PUF60 (%)", fontsize=8)
    ax.set_ylabel(f"Edit frequency — {mut_label} PUF60 (%)", fontsize=8)
    ax.set_title(
        f"WT vs {mut_label}   (shared sites n={len(merged):,})\n"
        f"{frac_above:.1%} of sites have higher frequency in {mut_label}",
        fontsize=9, fontweight="bold"
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7)

    # Annotate medians
    ax.axvline(np.median(x), color=PALETTE["WT"],       linestyle=":", linewidth=0.8, alpha=0.8)
    ax.axhline(np.median(y), color=PALETTE[mut_label], linestyle=":", linewidth=0.8, alpha=0.8)
    ax.text(np.median(x) + 0.5, lim * 0.98,
            f"WT med {np.median(x):.1f}%", fontsize=6, color=PALETTE["WT"], va="top")
    ax.text(lim * 0.02, np.median(y) + 0.5,
            f"{mut_label} med {np.median(y):.1f}%", fontsize=6, color=PALETTE[mut_label])

    return frac_above


# ── Figure 2: Edit-frequency distributions ────────────────────────────────────
def plot_freq_distributions(axes, wt_df, s161_df, s206_df):
    """Three violin panels: all sites / shared sites / new mutant sites."""
    shared_all = (set(wt_df["site_id"]) & set(s161_df["site_id"]) &
                  set(s206_df["site_id"]))
    new_s161 = set(s161_df["site_id"]) - set(wt_df["site_id"])
    new_s206 = set(s206_df["site_id"]) - set(wt_df["site_id"])
    new_both = new_s161 & new_s206

    panels = [
        ("All sites\n(each condition independently)",
         {
             "WT":   wt_df.groupby("site_id")["edit_freq"].mean().values,
             "S161": s161_df.groupby("site_id")["edit_freq"].mean().values,
             "S206": s206_df.groupby("site_id")["edit_freq"].mean().values,
         }),
        (f"Shared sites (n={len(shared_all):,})\n(present in all 3 conditions)",
         {
             "WT":   wt_df[wt_df["site_id"].isin(shared_all)].groupby("site_id")["edit_freq"].mean().values,
             "S161": s161_df[s161_df["site_id"].isin(shared_all)].groupby("site_id")["edit_freq"].mean().values,
             "S206": s206_df[s206_df["site_id"].isin(shared_all)].groupby("site_id")["edit_freq"].mean().values,
         }),
        (f"New sites in both mutants (n={len(new_both):,})\n(absent from WT)",
         {
             "S161": s161_df[s161_df["site_id"].isin(new_both)].groupby("site_id")["edit_freq"].mean().values,
             "S206": s206_df[s206_df["site_id"].isin(new_both)].groupby("site_id")["edit_freq"].mean().values,
         }),
    ]

    for ax, (title, data_dict) in zip(axes, panels):
        keys = list(data_dict.keys())
        vdata = [data_dict[k] for k in keys]
        colors = [PALETTE[k] for k in keys]

        parts = ax.violinplot(vdata, positions=range(len(keys)),
                              showmedians=True, showextrema=False)
        for pc, col in zip(parts["bodies"], colors):
            pc.set_facecolor(col)
            pc.set_alpha(0.65)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.5)

        # Overlay boxplot IQR
        ax.boxplot(vdata, positions=range(len(keys)), widths=0.08,
                   patch_artist=False, manage_ticks=False,
                   medianprops=dict(color="none"),
                   boxprops=dict(color="#2c3e50", linewidth=0.8),
                   whiskerprops=dict(color="none"),
                   capprops=dict(color="none"),
                   flierprops=dict(marker="", markersize=0))

        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(keys, fontsize=8)
        ax.set_ylabel("Edit frequency per site (%)", fontsize=8)
        ax.set_title(title, fontsize=8, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=7)

        # Annotate medians
        for i, (k, vals) in enumerate(zip(keys, vdata)):
            med = np.median(vals)
            ax.text(i, med + 0.5, f"{med:.1f}%", ha="center", fontsize=6.5,
                    color="black", fontweight="bold")


# ── Figure 3: Delta plot (gene level, absolute change) ────────────────────────
def plot_ma(ax, wt_df, mut_df, mut_label, min_sites=5):
    """Δ sites (mut - WT) vs WT site count — shows absolute gain at each gene."""
    wt_g  = gene_stats(wt_df)
    mut_g = gene_stats(mut_df)
    m = wt_g.rename(columns={"n_sites": "n_wt"}).merge(
        mut_g.rename(columns={"n_sites": "n_mut"}), on="gene", how="outer"
    ).fillna(0)
    m = m[(m["n_wt"] >= min_sites) | (m["n_mut"] >= min_sites)]
    m["delta"] = m["n_mut"] - m["n_wt"]   # absolute change in editing sites
    m["highlighted"] = m["gene"].isin(ALL_HIGHLIGHTED)

    bg = m[~m["highlighted"]]
    ax.scatter(bg["n_wt"], bg["delta"], s=5, alpha=0.2, color="#bdc3c7",
               linewidths=0, zorder=2)

    hl = m[m["highlighted"]]
    for _, row in hl.iterrows():
        ax.scatter(row["n_wt"], row["delta"], s=22, color=gene_color(row["gene"]),
                   alpha=0.85, linewidths=0.4, edgecolors="#333", zorder=4)

    ax.axhline(0, color="#7f8c8d", linewidth=0.8, linestyle="--", alpha=0.7)

    ax.set_xlabel("WT PUF60 site count", fontsize=8)
    ax.set_ylabel(f"Δ editing sites  ({mut_label} − WT)", fontsize=8)
    ax.set_title(f"Editing Site Change — {mut_label} vs WT\n"
                 f"(protein-coding genes, ≥{min_sites} sites in either condition)",
                 fontsize=9, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7)

    # Label top gainers by delta among highlighted
    for _, row in hl.nlargest(12, "delta").iterrows():
        ax.annotate(row["gene"],
                    xy=(row["n_wt"], row["delta"]),
                    xytext=(3, 2), textcoords="offset points",
                    fontsize=5.5, color=gene_color(row["gene"]), fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color="#ccc", lw=0.3))

    frac_up = (m["delta"] > 0).mean()
    ax.text(0.03, 0.97, f"{frac_up:.1%} of genes gained sites",
            transform=ax.transAxes, fontsize=7, va="top",
            color=PALETTE[mut_label], fontweight="bold")
    return ax


# ── Figure 4: Waterfall ────────────────────────────────────────────────────────
def plot_waterfall(ax, wt_df, mut_df, mut_label, top_n=35, min_sites=5):
    """Ranked bar of genes by delta (sites gained – sites lost)."""
    wt_g  = gene_stats(wt_df).rename(columns={"n_sites": "n_wt"})
    mut_g = gene_stats(mut_df).rename(columns={"n_sites": "n_mut"})
    m = wt_g.merge(mut_g, on="gene", how="outer").fillna(0)
    m = m[(m["n_wt"] >= min_sites) | (m["n_mut"] >= min_sites)]
    m["delta"] = m["n_mut"] - m["n_wt"]

    # Top gainers + top losers
    top = pd.concat([
        m.nlargest(top_n, "delta"),
        m.nsmallest(top_n // 3, "delta"),
    ]).drop_duplicates("gene").sort_values("delta")

    colors = [gene_color(g) if g in ALL_HIGHLIGHTED else
              ("#e74c3c" if d > 0 else "#3498db")
              for g, d in zip(top["gene"], top["delta"])]

    y = np.arange(len(top))
    bars = ax.barh(y, top["delta"], height=0.7, color=colors, alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(top["gene"], fontsize=6.5)
    ax.axvline(0, color="#2c3e50", linewidth=0.8)
    ax.set_xlabel(f"Δ editing sites  ({mut_label} – WT)", fontsize=8)
    ax.set_title(f"Waterfall: Top Changed Genes — {mut_label} vs WT", fontsize=9,
                 fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7)

    # Annotate WT and mutant counts on the bars
    for bar, (_, row) in zip(bars, top.iterrows()):
        dx = row["delta"]
        ax.text(dx + (0.5 if dx >= 0 else -0.5), bar.get_y() + bar.get_height() / 2,
                f"WT:{row['n_wt']:.0f}→{mut_label}:{row['n_mut']:.0f}",
                va="center", ha="left" if dx >= 0 else "right",
                fontsize=4.5, color="#2c3e50")

    return ax


# ── Figure 5: Site composition stacked bar ────────────────────────────────────
def plot_site_composition(ax, wt_df, s161_df, s206_df):
    sites_wt   = set(wt_df["site_id"])
    sites_s161 = set(s161_df["site_id"])
    sites_s206 = set(s206_df["site_id"])

    shared_all  = sites_wt & sites_s161 & sites_s206
    wt_s161_only = (sites_wt & sites_s161) - sites_s206
    wt_s206_only = (sites_wt & sites_s206) - sites_s161
    wt_unique   = sites_wt - sites_s161 - sites_s206
    s161_unique = sites_s161 - sites_wt - sites_s206
    s206_unique = sites_s206 - sites_wt - sites_s161
    new_both    = (sites_s161 & sites_s206) - sites_wt

    conditions = ["WT", "S161", "S206"]
    totals = [len(sites_wt), len(sites_s161), len(sites_s206)]

    # Per condition: which fraction falls into each category?
    cats = {
        "Shared all 3":       [len(shared_all)] * 3,
        "Shared WT+S161 only": [len(wt_s161_only), len(wt_s161_only), 0],
        "Shared WT+S206 only": [len(wt_s206_only), 0, len(wt_s206_only)],
        "Shared S161+S206 (new)": [0, len(new_both), len(new_both)],
        "Condition-unique":   [len(wt_unique), len(s161_unique), len(s206_unique)],
    }

    cat_colors = {
        "Shared all 3":           "#2ecc71",
        "Shared WT+S161 only":    "#3498db",
        "Shared WT+S206 only":    "#9b59b6",
        "Shared S161+S206 (new)": "#e67e22",
        "Condition-unique":       "#e74c3c",
    }

    x = np.arange(3)
    bottoms = np.zeros(3)
    for cat, counts in cats.items():
        fracs = [c / t for c, t in zip(counts, totals)]
        ax.bar(x, fracs, bottom=bottoms, color=cat_colors[cat], label=cat, alpha=0.85)
        for xi, (frac, cnt, bot) in enumerate(zip(fracs, counts, bottoms)):
            if frac > 0.04:
                ax.text(xi, bot + frac / 2, f"{cnt:,}\n({frac:.1%})",
                        ha="center", va="center", fontsize=5.5, color="white",
                        fontweight="bold")
        bottoms = bottoms + np.array(fracs)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n(n={t:,})" for c, t in zip(conditions, totals)], fontsize=8)
    ax.set_ylabel("Fraction of sites", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title("Site Composition — Shared vs Condition-Unique", fontsize=9, fontweight="bold")
    ax.legend(fontsize=6.5, loc="lower right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7)


# ── Figure 6: CH gene heatmap ─────────────────────────────────────────────────
def plot_ch_heatmap(ax, wt_df, s161_df, s206_df, min_wt_sites=5):
    """Heatmap: CH driver gene × condition — absolute Δ sites vs WT."""
    wt_g  = gene_stats(wt_df)[["gene", "n_sites"]].rename(columns={"n_sites": "WT"})
    s1_g  = gene_stats(s161_df)[["gene", "n_sites"]].rename(columns={"n_sites": "S161"})
    s2_g  = gene_stats(s206_df)[["gene", "n_sites"]].rename(columns={"n_sites": "S206"})

    mat = wt_g.merge(s1_g, on="gene", how="outer").merge(s2_g, on="gene", how="outer").fillna(0)
    mat = mat[mat["gene"].isin(ALL_HIGHLIGHTED)].copy()
    mat = mat[(mat["WT"] >= min_wt_sites) | (mat["S161"] >= min_wt_sites) |
              (mat["S206"] >= min_wt_sites)].copy()

    # Absolute delta vs WT
    mat["S161_delta"] = mat["S161"] - mat["WT"]
    mat["S206_delta"] = mat["S206"] - mat["WT"]

    # Sort by S161 delta descending
    mat = mat.sort_values("S161_delta", ascending=False)

    heat_data = mat[["S161_delta", "S206_delta"]].values.T  # shape (2, n_genes)

    vmax = max(abs(heat_data).max(), 5)
    im = ax.imshow(heat_data, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["S161\nΔ sites", "S206\nΔ sites"], fontsize=8)
    ax.set_xticks(range(len(mat)))
    ax.set_xticklabels(mat["gene"], rotation=55, ha="right", fontsize=6.5)

    for tick, gene in zip(ax.get_xticklabels(), mat["gene"]):
        tick.set_color(gene_color(gene))

    ax.set_title(
        "CH/Leukemia Driver Genes — Δ editing sites vs WT PUF60\n"
        "Red = more sites in mutant · Blue = fewer sites",
        fontsize=9, fontweight="bold"
    )

    # Annotate cells with delta value
    for i in range(heat_data.shape[0]):
        for j in range(heat_data.shape[1]):
            v = heat_data[i, j]
            txt_color = "white" if abs(v) > vmax * 0.55 else "#222"
            ax.text(j, i, f"{v:+.0f}", ha="center", va="center",
                    fontsize=5, color=txt_color)

    # Add raw WT site count below x-axis
    for j, (_, row) in enumerate(mat.iterrows()):
        ax.text(j, 1.8, f"WT:{row['WT']:.0f}", ha="center", va="bottom",
                fontsize=4, color="#2c3e50", rotation=55)

    plt.colorbar(im, ax=ax, shrink=0.6, label="Δ editing sites vs WT", pad=0.02)
    return ax


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wt",    required=True)
    p.add_argument("--s161",  required=True)
    p.add_argument("--s206",  required=True)
    p.add_argument("--outdir", default=".")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    log.info("Loading BED files (all sites)...")
    wt_all   = load_full(args.wt)
    s161_all = load_full(args.s161)
    s206_all = load_full(args.s206)

    log.info("Loading BED files (protein-coding only)...")
    wt_pc   = load_pc_only(args.wt)
    s161_pc = load_pc_only(args.s161)
    s206_pc = load_pc_only(args.s206)

    sites_wt   = set(wt_all["site_id"])
    sites_s161 = set(s161_all["site_id"])
    sites_s206 = set(s206_all["site_id"])
    shared_all = sites_wt & sites_s161 & sites_s206
    log.info(f"Sites — WT:{len(sites_wt):,}  S161:{len(sites_s161):,}  "
             f"S206:{len(sites_s206):,}  shared:{len(shared_all):,}")

    # ── Figure 1: Site-frequency scatter ──────────────────────────────────────
    log.info("Figure 1: site-frequency scatter...")
    fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(14, 6))
    plot_site_freq_scatter(ax1a, wt_all, s161_all, "S161", shared_all)
    plot_site_freq_scatter(ax1b, wt_all, s206_all, "S206", shared_all)
    fig1.suptitle(
        "Per-Site Edit Frequency at Shared Sites — WT vs Mutant PUF60\n"
        "Each point = one A→G editing site present in all three conditions",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out1 = outdir / "wt_vs_mutant_site_freq.pdf"
    plt.savefig(out1, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved: {out1}")

    # ── Figure 2: Edit-frequency distributions ────────────────────────────────
    log.info("Figure 2: frequency distributions...")
    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5.5))
    plot_freq_distributions(axes2, wt_all, s161_all, s206_all)
    fig2.suptitle(
        "Distribution of Per-Site A→G Edit Frequency by Condition\n"
        "New sites in mutants have lower frequency than shared sites",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    out2 = outdir / "editing_freq_distribution.pdf"
    plt.savefig(out2, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved: {out2}")

    # ── Figure 3: MA plot ─────────────────────────────────────────────────────
    log.info("Figure 3: MA plot...")
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 6))
    plot_ma(ax3a, wt_pc, s161_pc, "S161")
    plot_ma(ax3b, wt_pc, s206_pc, "S206")

    legend_els = [
        mpatches.Patch(color="#c0392b", label="Core CH driver"),
        mpatches.Patch(color="#e67e22", label="Myeloid/leukemia gene"),
        mpatches.Patch(color="#2980b9", label="Known PUF60 target"),
        mpatches.Patch(color="#bdc3c7", label="Other protein-coding"),
    ]
    fig3.legend(handles=legend_els, loc="lower center", ncol=4,
                fontsize=7.5, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))
    fig3.suptitle(
        "MA Plot — Gene-Level Site Count Fold Change vs WT PUF60\n"
        "Gains are enriched at already highly-bound genes (right side of x-axis)",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0.06, 1, 0.94])
    out3 = outdir / "ma_plot.pdf"
    plt.savefig(out3, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved: {out3}")

    # ── Figure 4: Waterfall ────────────────────────────────────────────────────
    log.info("Figure 4: waterfall plots...")
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(16, 10),
                                       gridspec_kw={"wspace": 0.45})
    plot_waterfall(ax4a, wt_pc, s161_pc, "S161", top_n=35)
    plot_waterfall(ax4b, wt_pc, s206_pc, "S206", top_n=35)

    legend_els4 = [
        mpatches.Patch(color="#c0392b", label="Core CH driver"),
        mpatches.Patch(color="#e67e22", label="Myeloid/leukemia gene"),
        mpatches.Patch(color="#2980b9", label="Known PUF60 target"),
        mpatches.Patch(color="#e74c3c", alpha=0.6, label="Other gainer"),
        mpatches.Patch(color="#3498db", alpha=0.6, label="Other loser"),
    ]
    fig4.legend(handles=legend_els4, loc="lower center", ncol=5,
                fontsize=7.5, framealpha=0.9, bbox_to_anchor=(0.5, -0.01))
    fig4.suptitle(
        "Waterfall — Top Genes by Editing Site Count Change (Mutant vs WT PUF60)",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    out4 = outdir / "waterfall_top_genes.pdf"
    plt.savefig(out4, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved: {out4}")

    # ── Figure 5: Site composition ────────────────────────────────────────────
    log.info("Figure 5: site composition bars...")
    fig5, ax5 = plt.subplots(figsize=(7, 5.5))
    plot_site_composition(ax5, wt_all, s161_all, s206_all)
    fig5.suptitle(
        "Site-Level Composition — Shared vs Condition-Unique A→G Sites\n"
        "(all biotypes)",
        fontsize=10, fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out5 = outdir / "site_composition_bars.pdf"
    plt.savefig(out5, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved: {out5}")

    # ── Figure 6: CH gene heatmap ─────────────────────────────────────────────
    log.info("Figure 6: CH gene heatmap...")
    fig6, ax6 = plt.subplots(figsize=(16, 4))
    plot_ch_heatmap(ax6, wt_pc, s161_pc, s206_pc)
    plt.tight_layout()
    out6 = outdir / "ch_gene_heatmap.pdf"
    plt.savefig(out6, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved: {out6}")

    log.info("\nAll 6 figures complete.")
    for out in [out1, out2, out3, out4, out5, out6]:
        log.info(f"  {out}")


if __name__ == "__main__":
    main()
