#!/usr/bin/env python3
"""
proximity_filter.py — proximity-based replicate support for TRIBE editing sites.

Standard TRIBE reproducibility requires the EXACT same nucleotide to be edited
in ≥2 replicates. This script adds a complementary criterion: if site A (rep i)
and site B (rep j) are within d nucleotides on the same chromosome (and strand,
when available) and both pass the control frequency filter, they form a cluster
that counts as mutually supporting evidence for binding at that locus.

Biological rationale: ADAR edits the most accessible adenosine in an RBP-tethered
dsRNA region. RNA secondary structure is dynamic, so different replicates may
present slightly different adenosines. Abruzzi et al. (RNA 2023, PMID 37169395)
demonstrated the same principle for APOBEC-based STAMP, using a 100 bp window.
Their data showed ~17% of ADAR sites have a proximal partner within 100 bp even
in high-reproducibility TRIBE experiments.

Algorithm:
  1. Load raw_editing_sites.bed (one row per site × replicate; rep_idx column required)
  2. For each (chromosome, strand) group, sort sites by position
  3. Chain-merge positions into clusters: consecutive sites within distance d are
     linked regardless of replicate. A single cluster may span > d total if bridged
     by intermediate sites (DBSCAN-like chaining).
  4. For each cluster:
       - Require ≥ min_replicates distinct rep_idx values
       - Require ALL sites have control_G/control_A ≤ max_ctrl_freq
  5. Classify each cluster as:
       - "exact"    — cluster contains ≥1 position called in ≥min_replicates
                      (would be caught by standard filter_replicates.py)
       - "proximal" — cluster is supported by ≥min_replicates but NOT via exact match
                      (newly rescued by this analysis)
  6. Report all individual sites within qualifying clusters, tagged with:
       cluster_id, cluster_type, n_sites_in_cluster, n_replicates_in_cluster,
       cluster_start, cluster_end, representative_pos (highest edit_freq site)

Usage:
    python proximity_filter.py \\
        --raw-sites   results/raw_editing_sites.bed \\
        --distances   10 100 500 1000 \\
        --outdir      proximity_results/ \\
        [--min-replicates 2] \\
        [--max-ctrl-freq 0.01] \\
        [--exact-match-bed results/filtered_editing_sites.bed]

Note on strand: the current TRIBE pipeline sets strand='.' for all sites. When
strand is '.', sites on the same chromosome are clustered together regardless of
the underlying transcript strand. To enable true strand-aware clustering, add
strand inference to call_editing_sites_parallel.py or annotate sites from a GTF.

Outputs (per distance d):
    {outdir}/proximity_{d}nt_all.bed      — all sites in qualifying clusters
    {outdir}/proximity_{d}nt_rescued.bed  — proximal-only (not exact-match)
    {outdir}/proximity_{d}nt_combined.bed — exact-match ∪ proximity (representative sites)

Summary (across all distances):
    {outdir}/proximity_summary.tsv
    {outdir}/proximity_sites_barplot.pdf
    {outdir}/proximity_cluster_dist.pdf
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Column definitions ────────────────────────────────────────────────────────

RAW_COLS = [
    "chr", "start", "end", "gene", "edit_freq", "strand",
    "ctrl_cov", "ctrl_A", "ctrl_G",
    "treat_cov", "treat_A", "treat_G",
    "fold_change", "p_value", "rep_idx",
]

# Older raw BEDs (before rep_idx was added) have only 14 columns
RAW_COLS_LEGACY = RAW_COLS[:-1]

OUT_COLS = RAW_COLS + [
    "cluster_id", "cluster_type", "cluster_start", "cluster_end",
    "n_sites_cluster", "n_reps_cluster", "rep_site",
]


# ── Loading ───────────────────────────────────────────────────────────────────

def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", comment="#",
                     header=None, low_memory=False)
    if df.shape[1] == len(RAW_COLS):
        df.columns = RAW_COLS
    elif df.shape[1] == len(RAW_COLS_LEGACY):
        df.columns = RAW_COLS_LEGACY
        # rep_idx not present. For exact-match positions (≥2 rows), cumcount gives
        # correct 0/1 assignment. For single-rep positions, cumcount assigns 0 to
        # all, making two nearby single-rep sites look like the same replicate —
        # proximity rescue will be zero for those. Re-running the calling script
        # with the updated version (which emits rep_idx) is required for correct
        # proximity results.
        df = df.sort_values(["chr", "start"]).reset_index(drop=True)
        df["rep_idx"] = df.groupby(["chr", "start"]).cumcount()
        print("WARNING: raw BED has no rep_idx column. Proximity rescue of "
              "single-replicate sites requires re-running call_editing_sites_parallel.py "
              "(updated version emits rep_idx as column 15). Exact-match site counts "
              "will be correct; proximal-only counts will be zero.", file=sys.stderr)
    else:
        raise ValueError(f"Unexpected number of columns: {df.shape[1]}")
    df["ctrl_edit_freq"] = np.where(
        df["ctrl_A"] > 0, df["ctrl_G"] / df["ctrl_A"], 0.0
    )
    return df


def load_exact_positions(path: str) -> set:
    """Return set of (chr, start) positions from an existing filtered BED."""
    if not path or not Path(path).exists():
        return set()
    df = pd.read_csv(path, sep="\t", comment="#", header=None, usecols=[0, 1])
    return set(zip(df[0], df[1]))


# ── Clustering ────────────────────────────────────────────────────────────────

def chain_cluster(positions: np.ndarray, distance: int) -> np.ndarray:
    """
    Assign cluster IDs to sorted positions using chain merging.
    Consecutive positions within `distance` nt are merged into the same cluster.
    Returns an integer array of cluster IDs (0-based, per-chromosome).
    """
    ids = np.zeros(len(positions), dtype=int)
    cid = 0
    for i in range(1, len(positions)):
        if positions[i] - positions[i - 1] > distance:
            cid += 1
        ids[i] = cid
    return ids


def cluster_sites(df: pd.DataFrame, distance: int,
                  max_ctrl_freq: float, min_replicates: int) -> pd.DataFrame:
    """
    Cluster sites by proximity, then filter clusters by replicate support.
    Returns a DataFrame with cluster metadata columns added.
    All sites from qualifying clusters are returned (one row per site).
    """
    records = []

    group_keys = ["chr", "strand"] if (df["strand"] != ".").any() else ["chr"]

    for keys, group in df.groupby(group_keys):
        chrom = keys[0] if isinstance(keys, tuple) else keys
        strand = keys[1] if isinstance(keys, tuple) and len(keys) > 1 else "."

        group = group.sort_values("start").reset_index(drop=True)
        positions = group["start"].values
        cluster_ids = chain_cluster(positions, distance)
        group = group.copy()
        group["_cid"] = cluster_ids

        for cid, clust in group.groupby("_cid"):
            # Control frequency filter — all sites in cluster must pass
            if (clust["ctrl_edit_freq"] > max_ctrl_freq).any():
                continue

            # Replicate support filter
            n_reps = clust["rep_idx"].nunique()
            if n_reps < min_replicates:
                continue

            # Representative site: highest edit_freq
            rep_row = clust.loc[clust["edit_freq"].idxmax()]
            rep_pos = int(rep_row["start"])

            cluster_label = f"{chrom}:{strand}:{cid}" if strand != "." else f"{chrom}:{cid}"

            for _, row in clust.iterrows():
                r = row.to_dict()
                r["cluster_id"]     = cluster_label
                r["cluster_start"]  = int(clust["start"].min())
                r["cluster_end"]    = int(clust["start"].max())
                r["n_sites_cluster"] = len(clust)
                r["n_reps_cluster"] = n_reps
                r["rep_site"]       = rep_pos
                records.append(r)

    if not records:
        return pd.DataFrame(columns=OUT_COLS)

    result = pd.DataFrame(records)
    result.drop(columns=["_cid", "ctrl_edit_freq"], inplace=True, errors="ignore")
    return result


def classify_clusters(clustered: pd.DataFrame,
                      exact_positions: set) -> pd.DataFrame:
    """
    Tag each cluster as 'exact' or 'proximal'.
    A cluster is 'exact' if ANY site in it is at a position also found in the
    standard exact-match filtered BED (or if any position has ≥2 rep rows,
    meaning it was called in multiple replicates at the same site).
    """
    if clustered.empty:
        return clustered

    # Positions called in multiple replicates (from the raw BED itself)
    rep_counts = clustered.groupby(["chr", "start"])["rep_idx"].nunique()
    exact_from_raw = set(rep_counts[rep_counts >= 2].index)

    def _cluster_type(sub):
        positions = set(zip(sub["chr"], sub["start"]))
        if positions & exact_positions or positions & exact_from_raw:
            return "exact"
        return "proximal"

    types = (
        clustered.groupby("cluster_id")
        .apply(_cluster_type, include_groups=False)
        .rename("cluster_type")
    )
    clustered = clustered.merge(types, on="cluster_id")
    return clustered


# ── Output helpers ────────────────────────────────────────────────────────────

def write_bed(df: pd.DataFrame, path: Path, label: str):
    if df.empty:
        print(f"  {label}: 0 sites")
        path.write_text("")
        return
    df.to_csv(path, sep="\t", index=False, float_format="%.4f")
    n_sites    = len(df)
    n_clusters = df["cluster_id"].nunique()
    n_genes    = df[df["gene"] != "."]["gene"].nunique()
    print(f"  {label}: {n_sites} sites, {n_clusters} clusters, {n_genes} genes")


def representative_sites(df: pd.DataFrame) -> pd.DataFrame:
    """One row per cluster — the site with the highest edit frequency."""
    if df.empty:
        return df
    return (
        df.sort_values("edit_freq", ascending=False)
          .groupby("cluster_id", sort=False)
          .first()
          .reset_index()
    )


# ── Figures ───────────────────────────────────────────────────────────────────

def plot_summary_bars(summary: pd.DataFrame, outdir: Path):
    distances = summary["distance"].unique()
    x = np.arange(len(distances))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for ax, metric, ylabel in zip(
        axes,
        ["n_exact_sites",    "n_proximal_sites"],
        ["Editing sites",     "Editing sites"],
    ):
        exact     = summary.groupby("distance")["n_exact_sites"].first().reindex(distances).fillna(0)
        proximal  = summary.groupby("distance")["n_proximal_sites"].first().reindex(distances).fillna(0)

        ax.bar(x, exact,    width, label="Exact match (≥2 rep, same position)", color="#1a476f")
        ax.bar(x, proximal, width, bottom=exact,
               label="Proximity rescued",  color="#90caf9", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{d} nt" for d in distances])
        ax.set_ylabel("Editing sites")
        ax.set_title("Sites" if "site" in metric else "Genes")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8)

    # Right panel: genes
    ax = axes[1]
    ax.cla()
    exact_g    = summary.groupby("distance")["n_exact_genes"].first().reindex(distances).fillna(0)
    proximal_g = summary.groupby("distance")["n_proximal_genes"].first().reindex(distances).fillna(0)
    ax.bar(x, exact_g,    width, color="#1a476f",  label="Exact match")
    ax.bar(x, proximal_g, width, bottom=exact_g,
           color="#90caf9", alpha=0.85, label="Proximity rescued")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d} nt" for d in distances])
    ax.set_ylabel("Target genes")
    ax.set_title("Genes")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Proximity-based site rescue vs. distance threshold\n"
                 "(Abruzzi et al. 2023 approach, extended from STAMP to TRIBE)",
                 fontsize=10)
    fig.tight_layout()
    out = outdir / "proximity_sites_barplot.pdf"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_cluster_distributions(all_clustered: dict, outdir: Path):
    """Violin/box of intra-cluster span and cluster size for each distance."""
    distances = sorted(all_clustered.keys())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, metric, ylabel, title in zip(
        axes,
        ["span", "n_sites_cluster"],
        ["Cluster span (nt)", "Sites per cluster"],
        ["Intra-cluster span of proximal events",
         "Cluster size (sites) of proximal events"],
    ):
        data = []
        labels = []
        for d in distances:
            df = all_clustered[d]
            prox = df[df["cluster_type"] == "proximal"]
            if prox.empty:
                continue
            if metric == "span":
                vals = (prox["cluster_end"] - prox["cluster_start"]).clip(lower=0)
            else:
                vals = prox["n_sites_cluster"]
            # One value per cluster (not per site)
            vals = prox.groupby("cluster_id")[
                "cluster_end" if metric == "span" else "n_sites_cluster"
            ].first()
            if metric == "span":
                vals = vals  # already per-cluster
            data.append(vals.values)
            labels.append(f"{d} nt")

        if data:
            ax.violinplot(data, positions=range(len(data)), showmedians=True)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out = outdir / "proximity_cluster_dist.pdf"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-sites",    required=True,
                    help="raw_editing_sites.bed from call_editing_sites_parallel.py")
    ap.add_argument("--distances",    nargs="+", type=int,
                    default=[10, 100, 500, 1000],
                    help="Proximity windows in nt (default: 10 100 500 1000)")
    ap.add_argument("--min-replicates", type=int, default=2)
    ap.add_argument("--max-ctrl-freq",  type=float, default=0.01,
                    help="Max control A→G frequency (default 0.01 = 1%%)")
    ap.add_argument("--exact-match-bed", default=None,
                    help="Optional filtered_editing_sites.bed for exact-match tagging")
    ap.add_argument("--outdir", default="proximity_results")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {args.raw_sites}")
    raw = load_raw(args.raw_sites)
    print(f"  {len(raw):,} site-replicate rows, "
          f"{raw['chr'].nunique()} chromosomes, "
          f"{raw[['chr','start']].drop_duplicates().shape[0]:,} unique positions")

    if (raw["strand"] == ".").all():
        print("NOTE: all strand values are '.'. Clustering is chromosome-only "
              "(not strand-aware). Add strand inference to the calling script "
              "for true strand separation.")

    exact_positions = load_exact_positions(args.exact_match_bed)
    print(f"  {len(exact_positions):,} exact-match positions loaded for comparison")

    summary_rows = []
    all_clustered = {}

    for d in sorted(args.distances):
        print(f"\n── Distance: {d} nt ──────────────────────────────────────")
        clustered = cluster_sites(raw, d, args.max_ctrl_freq, args.min_replicates)

        if clustered.empty:
            print("  No clusters passing filters.")
            summary_rows.append({
                "distance": d,
                "n_exact_sites": 0, "n_proximal_sites": 0,
                "n_exact_clusters": 0, "n_proximal_clusters": 0,
                "n_exact_genes": 0, "n_proximal_genes": 0,
            })
            continue

        clustered = classify_clusters(clustered, exact_positions)
        all_clustered[d] = clustered

        exact   = clustered[clustered["cluster_type"] == "exact"]
        proximal = clustered[clustered["cluster_type"] == "proximal"]

        def _genes(df):
            return df[df["gene"] != "."]["gene"].nunique() if not df.empty else 0

        summary_rows.append({
            "distance": d,
            "n_exact_sites":       len(exact),
            "n_proximal_sites":    len(proximal),
            "n_exact_clusters":    exact["cluster_id"].nunique() if not exact.empty else 0,
            "n_proximal_clusters": proximal["cluster_id"].nunique() if not proximal.empty else 0,
            "n_exact_genes":       _genes(exact),
            "n_proximal_genes":    _genes(proximal),
        })

        # Write per-distance BEDs
        write_bed(clustered, outdir / f"proximity_{d}nt_all.bed",
                  f"All clusters at {d}nt")
        write_bed(proximal,  outdir / f"proximity_{d}nt_rescued.bed",
                  f"Proximal-only rescued at {d}nt")

        # Combined callset: exact-match representative sites + proximal representative sites
        rep_exact    = representative_sites(exact)
        rep_proximal = representative_sites(proximal)
        combined     = pd.concat([rep_exact, rep_proximal], ignore_index=True)
        write_bed(combined, outdir / f"proximity_{d}nt_combined.bed",
                  f"Combined callset (representative sites) at {d}nt")

    # Summary table
    summary = pd.DataFrame(summary_rows)
    out_summary = outdir / "proximity_summary.tsv"
    summary.to_csv(out_summary, sep="\t", index=False)
    print(f"\nSummary → {out_summary}")
    print(summary.to_string(index=False))

    # Figures
    if not summary.empty:
        plot_summary_bars(summary, outdir)
    if all_clustered:
        plot_cluster_distributions(all_clustered, outdir)

    print(f"\nAll outputs → {outdir}/")


if __name__ == "__main__":
    main()
