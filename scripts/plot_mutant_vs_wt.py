#!/usr/bin/env python3
"""
PUF60 Mutant vs WT — site-count scatter and CH driver analysis

For each protein-coding gene, computes the number of TRIBE editing sites
(as a proxy for PUF60 occupancy depth) in WT, S161, and S206 conditions.

Outputs:
  wt_vs_mutant_scatter.pdf    — WT vs S161 and WT vs S206 scatter plots
  ch_driver_delta.pdf         — ranked delta chart for CH/leukemia driver genes
  mutant_specific_genes.pdf   — summary of 285 S161∩S206-not-WT genes
  mutant_specific_genes.tsv   — full table of all genes with site counts + deltas

Usage (via sbatch wrapper):
  python plot_mutant_vs_wt.py \
    --wt   wt_concat_strict/annotated_editing_sites.bed \
    --s161 s161_concat_strict/annotated_editing_sites.bed \
    --s206 s206_concat_strict/annotated_editing_sites.bed \
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
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
import matplotlib.cm as cm

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

# ── CH / leukemia gene annotation ─────────────────────────────────────────────
# Categorised for color coding in the scatter
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

# Color map: core CH = red, myeloid/leukemia = orange, PUF60 known = blue
def gene_color(g):
    if g in CH_CORE:
        return "#c0392b"
    if g in CH_MYELOID:
        return "#e67e22"
    if g in PUF60_KNOWN:
        return "#2980b9"
    return None


# ── Data loading ───────────────────────────────────────────────────────────────
def load_gene_stats(bed_path: str) -> pd.DataFrame:
    df = pd.read_csv(bed_path, sep="\t", comment=None, low_memory=False)
    df.columns = [c.lstrip("#") for c in df.columns]
    df = df[df["gene_type"] == "protein_coding"].copy()
    df = df[~df["gene"].isin([".", ""])].copy()
    df = df.dropna(subset=["gene", "edit_freq"])
    agg = (df.groupby("gene")
             .agg(n_sites=("gene", "count"),
                  median_freq=("edit_freq", "median"),
                  mean_freq=("edit_freq", "mean"))
             .reset_index())
    return agg


def build_comparison(wt_df, mut_df, mut_label):
    m = wt_df.rename(columns={"n_sites": "n_wt", "median_freq": "freq_wt"}).merge(
        mut_df.rename(columns={"n_sites": "n_mut", "median_freq": "freq_mut"}),
        on="gene", how="outer"
    ).fillna(0)
    m["delta"] = m["n_mut"] - m["n_wt"]
    m["log2fc"] = np.log2((m["n_mut"] + 1) / (m["n_wt"] + 1))
    m["highlighted"] = m["gene"].isin(ALL_HIGHLIGHTED)
    m["color_cat"] = m["gene"].apply(gene_color)
    return m


# ── Figure 1: Scatter WT vs mutant ────────────────────────────────────────────
def plot_scatter(ax, df, title, mut_col="n_mut", wt_col="n_wt",
                 min_sites=5, top_n=12):
    """WT (x) vs mutant (y) site count scatter, labeled top changers."""
    sig = df[(df[wt_col] >= min_sites) | (df[mut_col] >= min_sites)].copy()

    xmax = max(sig[wt_col].max(), sig[mut_col].max()) * 1.15

    # Grey background
    bg = sig[~sig["highlighted"]]
    ax.scatter(bg[wt_col], bg[mut_col], s=12, alpha=0.25,
               color="#bdc3c7", linewidths=0, zorder=2)

    # Highlighted genes (CH/known)
    for _, row in sig[sig["highlighted"]].iterrows():
        c = row["color_cat"] or "#7f8c8d"
        ax.scatter(row[wt_col], row[mut_col], s=30, color=c,
                   alpha=0.85, linewidths=0.4, edgecolors="#333", zorder=4)

    # y=x diagonal
    diag = np.linspace(0, xmax, 200)
    ax.plot(diag, diag, "--", color="#7f8c8d", linewidth=0.8, alpha=0.6, zorder=1)

    # 2x gain / 2x loss guide lines
    ax.plot(diag, diag * 2, ":", color="#e74c3c", linewidth=0.6, alpha=0.4, zorder=1)
    ax.plot(diag * 2, diag, ":", color="#3498db", linewidth=0.6, alpha=0.4, zorder=1)

    ax.set_xlim(0, xmax)
    ax.set_ylim(0, xmax)
    ax.set_xlabel("WT PUF60 — sites per gene", fontsize=8)
    ax.set_ylabel(f"{title} — sites per gene", fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7)

    # Label: top gainers + losers + highlighted CH genes
    top_gainers = sig.nlargest(top_n, "delta")
    top_losers  = sig.nsmallest(top_n // 2, "delta")
    top_ch = sig[sig["highlighted"]].nlargest(top_n, "delta")
    to_label = pd.concat([top_gainers, top_losers, top_ch]).drop_duplicates("gene")

    for _, row in to_label.iterrows():
        c = row["color_cat"] or "#2c3e50"
        fw = "bold" if row["highlighted"] else "normal"
        ax.annotate(
            row["gene"],
            xy=(row[wt_col], row[mut_col]),
            xytext=(4, 3), textcoords="offset points",
            fontsize=5.5, color=c, fontweight=fw, zorder=6,
            arrowprops=dict(arrowstyle="-", color="#ccc", lw=0.35),
        )

    # Text annotations for guide lines
    ax.text(xmax * 0.96, xmax * 0.96, "y=x", fontsize=6, color="#7f8c8d", ha="right")
    ax.text(xmax * 0.5, xmax * 0.98, "2× gain", fontsize=5.5,
            color="#e74c3c", ha="center", alpha=0.7)
    ax.text(xmax * 0.98, xmax * 0.5, "2× loss", fontsize=5.5,
            color="#3498db", ha="right", alpha=0.7)

    return ax


# ── Figure 2: CH driver delta bar chart ──────────────────────────────────────
def plot_ch_delta(ax, df_s161, df_s206, highlight_genes, min_sites=1,
                  title="CH/Leukemia Driver Gene — Site Count Change vs WT"):
    """Grouped horizontal bar: delta(S161 - WT) and delta(S206 - WT) per CH gene."""
    # Merge both comparisons on gene
    ch161 = df_s161[df_s161["gene"].isin(highlight_genes)].set_index("gene")[["n_wt", "n_mut", "delta"]]
    ch206 = df_s206[df_s206["gene"].isin(highlight_genes)].set_index("gene")[["n_mut", "delta"]]
    ch = ch161.join(ch206, how="outer", lsuffix="_s161", rsuffix="_s206").fillna(0)
    ch = ch.rename(columns={"n_wt_s161": "n_wt"})  # WT same for both
    ch = ch[(ch["n_wt"] >= min_sites) | (ch["n_mut_s161"] >= min_sites) |
            (ch["n_mut_s206"] >= min_sites)]
    ch = ch.sort_values("delta_s161", ascending=True)

    genes = list(ch.index)
    y = np.arange(len(genes))
    bar_h = 0.35

    ax.barh(y + bar_h / 2, ch["delta_s161"], height=bar_h,
            color=["#e74c3c" if v >= 0 else "#3498db" for v in ch["delta_s161"]],
            alpha=0.8, label="S161")
    ax.barh(y - bar_h / 2, ch["delta_s206"], height=bar_h,
            color=["#e67e22" if v >= 0 else "#27ae60" for v in ch["delta_s206"]],
            alpha=0.8, label="S206")

    ax.set_yticks(y)
    ax.set_yticklabels(genes, fontsize=7)
    ax.axvline(0, color="#2c3e50", linewidth=0.8)
    ax.set_xlabel("Δ editing sites vs WT PUF60", fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7)

    legend_elements = [
        mpatches.Patch(color="#e74c3c", alpha=0.8, label="S161 gain"),
        mpatches.Patch(color="#3498db", alpha=0.8, label="S161 loss"),
        mpatches.Patch(color="#e67e22", alpha=0.8, label="S206 gain"),
        mpatches.Patch(color="#27ae60", alpha=0.8, label="S206 loss"),
    ]
    ax.legend(handles=legend_elements, fontsize=6.5, loc="lower right")
    return ax


# ── Figure 3: Mutant-specific genes (S161∩S206 not WT) ───────────────────────
def plot_mutant_specific(ax_main, ax_ch, s161_df, s206_df, wt_genes, all_highlighted):
    """Panel showing the 285 S161∩S206-not-WT genes."""
    s161_genes = set(s161_df["gene"])
    s206_genes = set(s206_df["gene"])
    shared_not_wt = (s161_genes & s206_genes) - wt_genes

    log.info(f"S161∩S206 not-WT: {len(shared_not_wt)} genes")

    s161_sub = s161_df[s161_df["gene"].isin(shared_not_wt)].copy()
    s206_sub = s206_df[s206_df["gene"].isin(shared_not_wt)].copy()
    merged = s161_sub.merge(s206_sub, on="gene", suffixes=("_s161", "_s206"))
    merged["combined"] = merged["n_sites_s161"] + merged["n_sites_s206"]
    merged = merged.sort_values("combined", ascending=False)
    merged["highlighted"] = merged["gene"].isin(all_highlighted)

    # Scatter: n_sites in S161 vs S206 (colored by highlighted)
    bg = merged[~merged["highlighted"]]
    ax_main.scatter(bg["n_sites_s161"], bg["n_sites_s206"],
                    s=20, alpha=0.4, color="#95a5a6", linewidths=0, zorder=2)
    hl = merged[merged["highlighted"]]
    if not hl.empty:
        colors = [gene_color(g) or "#8e44ad" for g in hl["gene"]]
        ax_main.scatter(hl["n_sites_s161"], hl["n_sites_s206"],
                        s=50, color=colors, alpha=0.9,
                        linewidths=0.5, edgecolors="#333", zorder=4)
        for _, row in hl.iterrows():
            ax_main.annotate(row["gene"],
                             xy=(row["n_sites_s161"], row["n_sites_s206"]),
                             xytext=(3, 2), textcoords="offset points",
                             fontsize=6, color=gene_color(row["gene"]) or "#8e44ad",
                             fontweight="bold")

    # Diagonal for reference
    dmax = max(merged["n_sites_s161"].max(), merged["n_sites_s206"].max()) + 2
    ax_main.plot([0, dmax], [0, dmax], "--", color="#bdc3c7", linewidth=0.7, alpha=0.6)
    ax_main.set_xlabel("Sites in S161 (not in WT)", fontsize=8)
    ax_main.set_ylabel("Sites in S206 (not in WT)", fontsize=8)
    ax_main.set_title(f"S161∩S206 Mutant-Specific Targets\n"
                      f"({len(shared_not_wt)} genes absent in WT PUF60)", fontsize=9, fontweight="bold")
    ax_main.spines["top"].set_visible(False)
    ax_main.spines["right"].set_visible(False)
    ax_main.tick_params(labelsize=7)

    # Side panel: top 20 by combined count
    top20 = merged.head(20)
    colors_bar = [gene_color(g) or "#95a5a6" for g in top20["gene"]]
    y = np.arange(len(top20))
    ax_ch.barh(y, top20["n_sites_s161"], height=0.4, color="#e74c3c", alpha=0.75, label="S161")
    ax_ch.barh(y - 0.4, top20["n_sites_s206"], height=0.4, color="#e67e22", alpha=0.75, label="S206")
    ax_ch.set_yticks(y - 0.2)
    ax_ch.set_yticklabels(top20["gene"], fontsize=7)
    ax_ch.set_xlabel("Sites (absent in WT)", fontsize=7)
    ax_ch.set_title("Top mutant-specific genes", fontsize=8, fontweight="bold")
    ax_ch.legend(fontsize=6.5)
    ax_ch.spines["top"].set_visible(False)
    ax_ch.spines["right"].set_visible(False)
    ax_ch.tick_params(labelsize=7)

    return shared_not_wt


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wt",   required=True, help="WT annotated_editing_sites.bed")
    p.add_argument("--s161", required=True, help="S161 annotated_editing_sites.bed")
    p.add_argument("--s206", required=True, help="S206 annotated_editing_sites.bed")
    p.add_argument("--outdir", default=".", help="Output directory")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    log.info("Loading WT...")
    wt_df = load_gene_stats(args.wt)
    log.info(f"  {len(wt_df):,} genes")
    log.info("Loading S161...")
    s161_df = load_gene_stats(args.s161)
    log.info(f"  {len(s161_df):,} genes")
    log.info("Loading S206...")
    s206_df = load_gene_stats(args.s206)
    log.info(f"  {len(s206_df):,} genes")

    wt_genes = set(wt_df["gene"])

    # Build comparison tables
    cmp_s161 = build_comparison(wt_df, s161_df, "S161")
    cmp_s206 = build_comparison(wt_df, s206_df, "S206")

    # ── Figure 1: WT vs mutant scatter ────────────────────────────────────────
    log.info("Plotting Figure 1: WT vs mutant scatter...")
    fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(16, 7))

    plot_scatter(ax1a, cmp_s161, "S161 PUF60", top_n=15)
    plot_scatter(ax1b, cmp_s206, "S206 PUF60", top_n=15)

    # Shared legend
    legend_elements = [
        mpatches.Patch(color="#c0392b", label="Core CH driver (TET2, DNMT3A, ASXL1…)"),
        mpatches.Patch(color="#e67e22", label="Myeloid/leukemia gene (KMT2A, CBL, STAG2…)"),
        mpatches.Patch(color="#2980b9", label="Known PUF60 target (QSER1, TEAD1, SP1…)"),
        mpatches.Patch(color="#bdc3c7", label="Other protein-coding gene"),
    ]
    fig1.legend(handles=legend_elements, loc="lower center", ncol=2,
                fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))
    fig1.suptitle(
        "PUF60 Mutant vs WT — Editing Site Occupancy per Gene\n"
        "Genes above diagonal = increased binding in mutant PUF60",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0.06, 1, 0.97])
    out1 = outdir / "wt_vs_mutant_scatter.pdf"
    plt.savefig(out1, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {out1}")

    # ── Figure 2: CH driver delta ──────────────────────────────────────────────
    log.info("Plotting Figure 2: CH driver delta bar chart...")
    ch_genes_in_data = ALL_HIGHLIGHTED & (set(cmp_s161["gene"]) | set(cmp_s206["gene"]))

    fig2, ax2 = plt.subplots(figsize=(10, max(6, len(ch_genes_in_data) * 0.28)))
    plot_ch_delta(ax2, cmp_s161, cmp_s206, ch_genes_in_data)

    fig2.suptitle(
        "CH/Leukemia Driver Genes — PUF60 Binding Change in Mutants vs WT\n"
        "Positive = more sites (increased occupancy) in mutant PUF60",
        fontsize=10, fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out2 = outdir / "ch_driver_delta.pdf"
    plt.savefig(out2, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {out2}")

    # ── Figure 3: Mutant-specific genes ──────────────────────────────────────
    log.info("Plotting Figure 3: mutant-specific genes...")
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 7))
    shared_not_wt = plot_mutant_specific(ax3a, ax3b, s161_df, s206_df,
                                         wt_genes, ALL_HIGHLIGHTED)
    fig3.suptitle(
        "Mutant-Specific PUF60 Targets: Genes Bound in S161∩S206 but Absent in WT",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out3 = outdir / "mutant_specific_genes.pdf"
    plt.savefig(out3, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {out3}")

    # ── TSV output ─────────────────────────────────────────────────────────────
    log.info("Writing full gene table...")
    full = wt_df.rename(columns={"n_sites": "n_wt", "median_freq": "freq_wt", "mean_freq": "meanfreq_wt"}).merge(
        s161_df.rename(columns={"n_sites": "n_s161", "median_freq": "freq_s161", "mean_freq": "meanfreq_s161"}),
        on="gene", how="outer"
    ).merge(
        s206_df.rename(columns={"n_sites": "n_s206", "median_freq": "freq_s206", "mean_freq": "meanfreq_s206"}),
        on="gene", how="outer"
    ).fillna(0)
    full["delta_s161"] = full["n_s161"] - full["n_wt"]
    full["delta_s206"] = full["n_s206"] - full["n_wt"]
    full["in_wt"]       = full["gene"].isin(wt_genes)
    full["in_s161"]     = full["gene"].isin(set(s161_df["gene"]))
    full["in_s206"]     = full["gene"].isin(set(s206_df["gene"]))
    full["mutant_specific"] = (full["in_s161"] & full["in_s206"] & ~full["in_wt"])
    full["category"]   = full["gene"].apply(
        lambda g: "core_CH" if g in CH_CORE else
                  "myeloid_leukemia" if g in CH_MYELOID else
                  "puf60_known" if g in PUF60_KNOWN else "other"
    )
    full = full.sort_values("delta_s161", ascending=False)
    out_tsv = outdir / "mutant_specific_genes.tsv"
    full.to_csv(out_tsv, sep="\t", index=False)
    log.info(f"Saved: {out_tsv}")

    # ── Summary to stdout ─────────────────────────────────────────────────────
    log.info("\n" + "="*60)
    log.info("SUMMARY")
    log.info("="*60)
    log.info(f"WT genes:   {len(wt_genes):,}")
    log.info(f"S161 genes: {len(set(s161_df['gene'])):,}")
    log.info(f"S206 genes: {len(set(s206_df['gene'])):,}")
    log.info(f"S161∩S206 not in WT: {len(shared_not_wt):,}")
    log.info("")
    log.info("Top 20 S161 gainers:")
    sig = full[(full["n_wt"] >= 5) | (full["n_s161"] >= 5)]
    print(sig.nlargest(20, "delta_s161")[
        ["gene", "n_wt", "n_s161", "delta_s161", "category"]
    ].to_string())
    log.info("")
    log.info("Top 20 S206 gainers:")
    sig2 = full[(full["n_wt"] >= 5) | (full["n_s206"] >= 5)]
    print(sig2.nlargest(20, "delta_s206")[
        ["gene", "n_wt", "n_s206", "delta_s206", "category"]
    ].to_string())
    log.info("")
    log.info("CH driver genes — binding changes:")
    ch_tbl = full[full["gene"].isin(ALL_HIGHLIGHTED)].sort_values("delta_s161", ascending=False)
    print(ch_tbl[["gene", "n_wt", "n_s161", "n_s206",
                  "delta_s161", "delta_s206", "category"]].to_string())


if __name__ == "__main__":
    main()
