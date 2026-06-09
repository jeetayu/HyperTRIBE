#!/usr/bin/env python3
"""
Focused metabolic figure for PUF60 mutants — clonal haematopoiesis narrative.
Reads the output of analyze_metabolism_ch.py (metabolic_genes.tsv + go_enrichment.tsv).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from matplotlib_venn import venn3, venn3_circles
from gprofiler import GProfiler
import time

sns.set_style('whitegrid')
plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'pdf.fonttype': 42,
    'svg.fonttype': 'none',
    'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11,
})

_CPAL = {'WT': '#2980b9', 'S161': '#e74c3c', 'S206': '#27ae60', 'Both': '#8e44ad'}

# Genes to highlight in narrative plots — hand-curated from output inspection
_SPOTLIGHT = {
    # TCA / Pyruvate
    'PDHA1':  ('TCA entry',      '#8e44ad'),
    'PDPR':   ('TCA entry',      '#8e44ad'),
    'PDP2':   ('TCA entry',      '#8e44ad'),
    'GLS':    ('Glutamine',      '#e67e22'),
    'ACSS3':  ('FAO',            '#16a085'),
    'AIG1':   ('FAO',            '#16a085'),
    'HADHB':  ('FAO',            '#16a085'),
    'ADIPOR2':('FAO',            '#16a085'),
    # OxPhos / Mitochondria
    'NDUFV1': ('Complex I',      '#9b59b6'),
    'TIMMDC1':('Complex I',      '#9b59b6'),
    'MRPS17': ('Mito ribosome',  '#7f8c8d'),
    'MRPL49': ('Mito ribosome',  '#7f8c8d'),
    'MRPL35': ('Mito ribosome',  '#7f8c8d'),
    'MTHFD2': ('One-carbon',     '#27ae60'),
    'MTRR':   ('One-carbon',     '#27ae60'),
    # Cytoprotection / ROS
    'HMOX1':  ('ROS/cytoprotect','#e74c3c'),
    'FOXO3':  ('mTOR/HSC',       '#f39c12'),
    'DDIT4':  ('mTOR/HSC',       '#f39c12'),
    'FOXK2':  ('Autophagy/gluc', '#f39c12'),
    'VHL':    ('HIF/hypoxia',    '#1abc9c'),
    'PRKAA2': ('AMPK',           '#1abc9c'),
    # Drug resistance / nucleotide
    'DCK':    ('Drug target',    '#c0392b'),
    'PRPS1':  ('Purine synth',   '#2980b9'),
    'ADSL':   ('Purine synth',   '#2980b9'),
    # Canonical CH / cancer
    'CBL':    ('CH driver',      '#c0392b'),
    'ABL1':   ('Oncoprotein',    '#c0392b'),
    'BCR':    ('Oncoprotein',    '#c0392b'),
    'CTNNB1': ('Wnt/β-catenin',  '#e74c3c'),
}

_CAT_ORDER = [
    'TCA entry', 'Glutamine', 'FAO', 'Complex I', 'Mito ribosome',
    'One-carbon', 'ROS/cytoprotect', 'mTOR/HSC', 'Autophagy/gluc',
    'HIF/hypoxia', 'AMPK', 'Drug target', 'Purine synth',
    'CH driver', 'Oncoprotein', 'Wnt/β-catenin',
]


def load_genes_editcount(path):
    counts = {}
    with open(path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            fields = line.split('\t')
            if len(fields) > 3 and fields[3] not in ('.', ''):
                counts[fields[3]] = counts.get(fields[3], 0) + 1
    return counts


def run_go_s161_s206_shared(shared_genes):
    """GO enrichment on S161∩S206 metabolic targets not in WT."""
    gp = GProfiler(return_dataframe=True)
    df = gp.profile(
        organism='hsapiens', query=sorted(shared_genes),
        sources=['GO:BP', 'GO:MF', 'KEGG', 'REAC'],
        significance_threshold_method='fdr', user_threshold=0.1,
        no_iea=False,
    )
    if df.empty:
        return df
    mask = df['name'].str.lower().str.contains(
        r'metabol|biosynthes|catabol|oxidat|fatty acid|glycoly|gluconeo|'
        r'tricarbox|citric|pyruvat|phosphoryl|electron trans|mitochondri|'
        r'amino acid|purine|pyrimidine|nucleotide|lipid|mtor|autophagy|'
        r'redox|reactive oxygen|serine|glutamin|folate|methyl|heme',
        regex=True, case=False
    )
    return df[mask | df['source'].isin(['KEGG', 'REAC'])].copy()


# ── Panels ──────────────────────────────────────────────────────────────────

def panel_dotplot(ax, gene_df, wt_c, s161_c, s206_c):
    """
    Dot plot: rows = spotlight genes, cols = WT / S161 / S206.
    Dot size ∝ edit-site count; colour = metabolic category.
    """
    spotlight_genes = [g for g in _SPOTLIGHT if g in gene_df['gene'].values or
                       g in wt_c or g in s161_c or g in s206_c]
    spotlight_genes = sorted(spotlight_genes,
                             key=lambda g: _CAT_ORDER.index(_SPOTLIGHT.get(g, ('z', '#888'))[0])
                             if _SPOTLIGHT.get(g, ('z', '#888'))[0] in _CAT_ORDER else 99)

    conditions = ['WT', 'S161', 'S206']
    count_maps = {'WT': wt_c, 'S161': s161_c, 'S206': s206_c}

    for yi, gene in enumerate(spotlight_genes):
        cat, col = _SPOTLIGHT.get(gene, ('Other', '#888'))
        for xi, cond in enumerate(conditions):
            n = count_maps[cond].get(gene, 0)
            if n > 0:
                ax.scatter(xi, yi, s=n * 80 + 60, c=col,
                           alpha=0.82, zorder=3, edgecolors='white', linewidths=0.5)
            else:
                ax.scatter(xi, yi, s=25, c='#f0f0f0',
                           alpha=0.5, zorder=2, edgecolors='#ccc', linewidths=0.5)

    ax.set_xticks(range(3))
    ax.set_xticklabels(conditions, fontsize=11)
    ax.set_yticks(range(len(spotlight_genes)))
    ax.set_yticklabels(spotlight_genes, fontsize=8.5)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, len(spotlight_genes) - 0.5)
    ax.set_title('Key Metabolic Genes — Editing Site Counts\n(dot size = # editing sites)',
                 fontsize=11, fontweight='bold')

    # Category colour strip on left
    for yi, gene in enumerate(spotlight_genes):
        cat, col = _SPOTLIGHT.get(gene, ('Other', '#888'))
        ax.barh(yi, -0.35, left=-0.5, height=0.85, color=col, alpha=0.55, zorder=1)

    # Size legend
    for n, lab in [(1, '1 site'), (3, '3 sites'), (5, '5 sites')]:
        ax.scatter([], [], s=n * 80 + 60, c='#888', label=lab, alpha=0.75)
    ax.legend(fontsize=8, loc='lower right', frameon=True, title='Edit sites')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', linestyle=':', alpha=0.4)


def panel_go_dotplot(ax, go_df, title='S161 ∩ S206 (not WT) — Metabolic GO Terms'):
    """GO enrichment dot plot: x = gene ratio, y = term, size = intersection_size, colour = source."""
    if go_df is None or go_df.empty:
        ax.text(0.5, 0.5, 'No significant terms', ha='center', va='center',
                transform=ax.transAxes)
        ax.axis('off')
        return

    top = go_df.nsmallest(20, 'p_value').copy()
    top['-log10p'] = -np.log10(top['p_value'].clip(lower=1e-30))
    top['gene_ratio'] = top['intersection_size'] / top['query_size']
    top = top.sort_values('gene_ratio')

    source_colors = {
        'GO:BP': '#3498db', 'GO:MF': '#e74c3c',
        'GO:CC': '#2ecc71', 'KEGG': '#e67e22', 'REAC': '#9b59b6'
    }
    colors = [source_colors.get(s, '#888') for s in top['source']]
    sc = ax.scatter(
        top['gene_ratio'], range(len(top)),
        s=top['intersection_size'] * 5 + 20,
        c=colors, alpha=0.8, edgecolors='white', linewidths=0.4, zorder=3
    )
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(
        [f"{n[:52]}…" if len(n) > 52 else n for n in top['name']],
        fontsize=8
    )
    ax.set_xlabel('Gene ratio (intersection / query set)')
    ax.set_title(title, fontsize=11, fontweight='bold')

    legend_elements = [
        mpatches.Patch(facecolor=c, label=s, alpha=0.8)
        for s, c in source_colors.items() if s in top['source'].values
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc='lower right')
    for n_int, lab in [(5, '5 genes'), (15, '15 genes'), (30, '30 genes')]:
        ax.scatter([], [], s=n_int * 5 + 20, c='#888', label=lab, alpha=0.7)
    ax2_legend = ax.legend(
        handles=[Line2D([0], [0], marker='o', color='w',
                        markerfacecolor='#888', markersize=np.sqrt(n * 5 + 20), label=lab)
                 for n, lab in [(5, '5'), (15, '15'), (30, '30 genes')]],
        fontsize=8, loc='upper left', title='Intersect size', frameon=True
    )
    ax.add_artist(ax.get_legend())
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def panel_subcatbar(ax, df):
    """Stacked bar of metabolic sub-categories for mutant-only genes."""
    df = df.copy()
    df['subcats'] = df['subcats'].fillna('Other metabolic')
    mut_only = df[df['in_mutants_only'] == True]

    cats_raw = []
    for sc_str in mut_only['subcats']:
        cats_raw.extend([c.strip() for c in sc_str.split(';') if c.strip()])
    from collections import Counter
    cat_counts = Counter(cats_raw)
    cats = sorted(cat_counts, key=lambda x: -cat_counts[x])[:12]

    # Split by S161-only vs S206-only vs shared
    s161_only = df[(df['s161_count']>0) & (df['wt_count']==0) & (df['s206_count']==0)]
    s206_only = df[(df['s206_count']>0) & (df['wt_count']==0) & (df['s161_count']==0)]
    shared    = df[(df['s161_count']>0) & (df['s206_count']>0) & (df['wt_count']==0)]

    def count_subcat(sub_df, cat):
        if sub_df.empty:
            return 0
        return sub_df['subcats'].fillna('').str.contains(cat, regex=False).sum()

    x = np.arange(len(cats))
    s161_vals = [count_subcat(s161_only, c) for c in cats]
    s206_vals = [count_subcat(s206_only, c) for c in cats]
    shared_vals = [count_subcat(shared, c) for c in cats]

    w = 0.25
    b1 = ax.bar(x - w, s161_vals, w, color=_CPAL['S161'], alpha=0.8, label='S161 only', edgecolor='white')
    b2 = ax.bar(x,     s206_vals, w, color=_CPAL['S206'], alpha=0.8, label='S206 only', edgecolor='white')
    b3 = ax.bar(x + w, shared_vals, w, color=_CPAL['Both'], alpha=0.8, label='S161∩S206', edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace(' / ', '/\n') for c in cats],
                       rotation=35, ha='right', fontsize=9)
    ax.set_ylabel('Mutant-only target gene count')
    ax.set_title('Metabolic Sub-Categories in PUF60 Mutant–Only Targets',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, frameon=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def panel_ch_narrative(ax, df):
    """
    Heatmap-style: top CH/cancer-relevant metabolic genes
    rows = genes, cols = [WT, S161, S206, CH_relevance_score]
    """
    df = df.copy()
    df['subcats'] = df['subcats'].fillna('')

    # Extended CH-relevant gene list including new findings
    ch_extended = {
        # Known CH drivers (metabolic annotation)
        'CBL':     ('CH driver',           4),
        'VHL':     ('HIF/hypoxia',         4),
        'PRKAA2':  ('AMPK/energy sensor',  4),
        'FOXO3':   ('HSC quiescence/ROS',  4),
        'ABL1':    ('Oncogenic TK',        4),
        'BCR':     ('Oncogenic TK',        4),
        'CTNNB1':  ('Wnt/β-catenin',       3),
        # Drug resistance
        'DCK':     ('Ara-C activation',    4),
        # Glutamine/TCA
        'GLS':     ('Glutamine addiction', 4),
        'PDHA1':   ('TCA entry (PDH)',     3),
        'PDPR':    ('TCA entry (PDH)',     3),
        'PDP2':    ('TCA entry (PDH)',     3),
        'ACADM':   ('FAO (MCAD)',          3),
        'HADHB':   ('FAO (TFP-β)',         3),
        'ACSS3':   ('FAO/mito',            2),
        'ADIPOR2': ('AMPK/FAO',            3),
        # OxPhos
        'NDUFV1':  ('Complex I',           3),
        'TIMMDC1': ('Complex I assembly',  2),
        # ROS/protection
        'HMOX1':   ('LSC cytoprotection',  4),
        # mTOR axis
        'DDIT4':   ('mTOR/HIF1 (REDD1)',   3),
        'FOXK2':   ('Autophagy/glucose',   3),
        # One-carbon
        'MTHFD2':  ('One-carbon (AML↑)',   3),
        'MTRR':    ('One-carbon/cobalamin',2),
        # Purine
        'ADSL':    ('Purine synthesis',    3),
        'PRPS1':   ('Purine synthesis',    3),
    }

    rows = []
    for gene, (annotation, score) in ch_extended.items():
        wt_n  = int(df.loc[df['gene']==gene, 'wt_count'].values[0])  if gene in df['gene'].values else 0
        s1_n  = int(df.loc[df['gene']==gene, 's161_count'].values[0]) if gene in df['gene'].values else 0
        s2_n  = int(df.loc[df['gene']==gene, 's206_count'].values[0]) if gene in df['gene'].values else 0
        mut_only = (wt_n == 0) and (s1_n + s2_n > 0)
        rows.append({
            'gene': gene,
            'annotation': annotation,
            'score': score,
            'WT': wt_n,
            'S161': s1_n,
            'S206': s2_n,
            'mutant_only': mut_only,
        })

    rdf = pd.DataFrame(rows)
    # Only show genes that are actually targeted
    rdf = rdf[(rdf['WT'] + rdf['S161'] + rdf['S206']) > 0]
    rdf = rdf.sort_values(['score', 'S161', 'S206'], ascending=False)

    if rdf.empty:
        ax.text(0.5, 0.5, 'No CH-relevant metabolic targets',
                ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
        return

    mat = rdf[['WT', 'S161', 'S206']].values.astype(float)
    cmap = LinearSegmentedColormap.from_list('wh_red', ['#ffffff', '#e74c3c'])

    im = ax.imshow(mat, aspect='auto', cmap=cmap, vmin=0, vmax=max(mat.max(), 1))
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['WT', 'S161', 'S206'], fontsize=11)
    ax.set_yticks(range(len(rdf)))
    ax.set_yticklabels(rdf['gene'], fontsize=9)

    # Annotate values
    for i in range(len(rdf)):
        for j in range(3):
            val = int(mat[i, j])
            if val > 0:
                ax.text(j, i, str(val), ha='center', va='center',
                        fontsize=8, fontweight='bold',
                        color='white' if val >= 2 else '#333')

    # Right-side annotation strip
    for i, (_, row) in enumerate(rdf.iterrows()):
        star = ' ★' if row['mutant_only'] else ''
        ax.text(3.15, i, f"{row['annotation']}{star}",
                va='center', fontsize=7.5, color='#333')

    ax.set_xlim(-0.5, 2.5)
    ax.set_title('CH-Relevant Metabolic Genes — Editing Site Counts\n'
                 '(★ = mutant-only target; score→ CH relevance)',
                 fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Editing sites', shrink=0.6, pad=0.25)
    ax.spines[:].set_visible(False)


def panel_notes(ax):
    ax.axis('off')
    text = (
        "Biological Narrative — PUF60 S161/S206 Gain-of-Function and Clonal Haematopoiesis\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "PUF60 (U2AF65-like, RRM+UHM domains) facilitates U2AF65 binding at weak 3′SS polypyrimidine tracts. S161 and S206\n"
        "mutations alter RRM domain contacts, likely shifting 3′SS usage across metabolic gene introns.\n\n"
        "KEY FINDINGS:\n"
        "1. TCA ENTRY (PDH complex): PDHA1, PDPR, PDP2 — all S161-enriched, regulate pyruvate → acetyl-CoA flux.\n"
        "   Altered splicing of PDH-complex regulators could lock cells in aerobic glycolysis (Warburg) or TCA-dependence.\n\n"
        "2. GLUTAMINE (GLS, S161): Glutaminase is rate-limiting for TCA anaplerosis in AML. GLS inhibitors (CB-839)\n"
        "   are in clinical trials for AML; mutant PUF60 targeting GLS mRNA may affect drug sensitivity.\n\n"
        "3. FAO: HADHB (TFP-β), ADIPOR2 (AMPK/FAO activator), AIG1 — S206-enriched. FAO supports HSC quiescence;\n"
        "   elevated FAO dependency is a hallmark of LSCs and CHIP clones (Ito et al. 2012, Cell Stem Cell).\n\n"
        "4. HMOX1 (both mutants): Heme oxygenase-1 is overexpressed in AML LSCs and confers resistance to ROS-inducing\n"
        "   therapy. PUF60-mutant binding to HMOX1 pre-mRNA may stabilise a cytoprotective splice isoform.\n\n"
        "5. DCK (S161, ★ mutant-only): Deoxycytidine kinase activates cytarabine (Ara-C). Splice variants of DCK\n"
        "   causing exon skipping are a known Ara-C resistance mechanism in AML (Eliopoulos et al. 1998).\n\n"
        "6. COMPLEX I (NDUFV1, TIMMDC1, both mutants): Complex I deficiency in haematopoietic cells drives\n"
        "   metabolic rewiring toward glycolysis and is linked to clonal expansion of TET2-mutant cells.\n\n"
        "7. mTOR AXIS (FOXO3-S206, DDIT4/REDD1-S206): Both suppress mTOR. Loss of mTOR suppression → HSC\n"
        "   exhaustion (Yilmaz et al. 2006, FOXO3 HSC quiescence). If mutant PUF60 alters their splicing,\n"
        "   a proliferative advantage in the CH clone could result."
    )
    ax.text(0.01, 0.99, text, transform=ax.transAxes, fontsize=8.5,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#fdfefe',
                      edgecolor='#aaa', linewidth=1))


# ─────────────────────────────────────────────────────────────────────────────

def plot(gene_tsv, wt_bed, s161_bed, s206_bed, output):
    df = pd.read_csv(gene_tsv, sep='\t')
    df['subcats']    = df['subcats'].fillna('')
    df['present_in'] = df['present_in'].fillna('')

    wt_c   = load_genes_editcount(wt_bed)
    s161_c = load_genes_editcount(s161_bed)
    s206_c = load_genes_editcount(s206_bed)

    # S161∩S206 metabolic not in WT for focused GO enrichment
    shared_not_wt = set(
        df[(df['s161_count']>0) & (df['s206_count']>0) & (df['wt_count']==0)]['gene']
    )
    print(f"Running GO on {len(shared_not_wt)} S161∩S206 metabolic genes...", flush=True)
    go_shared = run_go_s161_s206_shared(shared_not_wt)
    time.sleep(0.5)

    fig = plt.figure(figsize=(22, 30))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.4)

    ax_dot    = fig.add_subplot(gs[0, 0])
    ax_ch     = fig.add_subplot(gs[0, 1])
    ax_subcat = fig.add_subplot(gs[1, 0])
    ax_go     = fig.add_subplot(gs[1, 1])
    ax_notes  = fig.add_subplot(gs[2, :])

    panel_dotplot(ax_dot, df, wt_c, s161_c, s206_c)
    panel_ch_narrative(ax_ch, df)
    panel_subcatbar(ax_subcat, df)
    panel_go_dotplot(ax_go, go_shared)
    panel_notes(ax_notes)

    fig.suptitle(
        "PUF60 S161 / S206 — Metabolic RNA Targets in Clonal Haematopoiesis Context",
        fontsize=15, fontweight='bold', y=1.002
    )
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gene-tsv', required=True)
    p.add_argument('--wt',   required=True)
    p.add_argument('--s161', required=True)
    p.add_argument('--s206', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    plot(args.gene_tsv, args.wt, args.s161, args.s206, args.output)


if __name__ == '__main__':
    main()
