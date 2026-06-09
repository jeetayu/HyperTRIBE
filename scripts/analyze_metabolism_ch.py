#!/usr/bin/env python3
"""
Metabolic Target Analysis for PUF60 Mutants — Clonal Haematopoiesis Context
=============================================================================

Interrogates the metabolic GO terms enriched in PUF60 mutant (S161, S206)
HyperTRIBE targets relative to WT PUF60 to identify gene-level drivers.

Steps
-----
1.  Define gene sets: WT-unique, S161-unique, S206-unique, S161+S206-only
2.  Retrieve GO annotations for all target genes via mygene.info
3.  Identify genes driving metabolic term enrichment (GO:0008152 hierarchy,
    KEGG/REAC metabolism modules)
4.  Cross-reference against a curated CH-relevant gene/pathway list
5.  Output:
      • metabolic_targets.pdf  — multi-panel figure
      • metabolic_genes.tsv    — gene × condition × GO term table

Figures
-------
  Row 1: Heatmap of metabolic GO terms × conditions (−log10 p)
  Row 2: Gene bubble plot — metabolic genes shared/unique to mutants,
          coloured by CH relevance category
  Row 3: Venn of metabolic gene overlap WT/S161/S206
  Row 4: Bar chart of metabolic sub-categories (energy, lipid, AA, nucleotide …)
  Row 5: Dot plot of top 20 metabolic genes ranked by S161+S206 edit count
          vs WT, annotated with known CH association
"""

import argparse
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from matplotlib_venn import venn3, venn3_circles

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

sns.set_style('whitegrid')
plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300,
    'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11,
})

_PALETTE = {'WT': '#2980b9', 'S161': '#e74c3c', 'S206': '#27ae60',
            'S161+S206': '#8e44ad', 'Core': '#2c3e50'}

# ─────────────────────────────────────────────────────────────────────────────
# Curated CH-relevant gene list with category labels
# Derived from: Jaiswal & Ebert 2019 NEJM, Steensma 2018, Bowman et al. 2018,
#               and published PUF60/U2AF2 splicing datasets
# ─────────────────────────────────────────────────────────────────────────────
_CH_GENES = {
    # Canonical CH driver mutations (direct)
    'TET2':    'CH driver',  'DNMT3A':   'CH driver',  'ASXL1': 'CH driver',
    'JAK2':    'CH driver',  'SF3B1':    'CH driver',  'U2AF1': 'CH driver',
    'SRSF2':   'CH driver',  'ZRSR2':    'CH driver',  'IDH1':  'CH driver',
    'IDH2':    'CH driver',  'PPM1D':    'CH driver',  'TP53':  'CH driver',
    'CBL':     'CH driver',  'KRAS':     'CH driver',  'NRAS':  'CH driver',
    'GNB1':    'CH driver',
    # Metabolic regulators critical for HSC/leukemia cell survival
    'MYC':     'HSC metabolic',  'MYCN':   'HSC metabolic',
    'HIF1A':   'HSC metabolic',  'VHL':    'HSC metabolic',
    'MTOR':    'HSC metabolic',  'RPTOR':  'HSC metabolic',
    'TSC1':    'HSC metabolic',  'TSC2':   'HSC metabolic',
    'PTEN':    'HSC metabolic',  'PIK3CA': 'HSC metabolic',
    'PIK3R1':  'HSC metabolic',  'AKT1':   'HSC metabolic',
    'AKT2':    'HSC metabolic',  'AKT3':   'HSC metabolic',
    'AMPK':    'HSC metabolic',  'PRKAA1': 'HSC metabolic',
    'PRKAA2':  'HSC metabolic',  'PRKAB1': 'HSC metabolic',
    'FOXO1':   'HSC metabolic',  'FOXO3':  'HSC metabolic',
    'SIRT1':   'HSC metabolic',  'SIRT3':  'HSC metabolic',
    # Mitochondrial / oxidative phosphorylation — known to drive AML/MDS
    'IDH1':    'TCA/OxPhos',  'IDH2':  'TCA/OxPhos',
    'CS':      'TCA/OxPhos',  'MDH2':  'TCA/OxPhos',
    'SDHA':    'TCA/OxPhos',  'SDHB':  'TCA/OxPhos',
    'FH':      'TCA/OxPhos',  'SUCLG1':'TCA/OxPhos',
    'PDHA1':   'TCA/OxPhos',  'PDHB':  'TCA/OxPhos',
    'PC':      'TCA/OxPhos',  'ME1':   'TCA/OxPhos',
    # Fatty acid oxidation — HSC quiescence / CHIP survival
    'ACADM':   'FAO',  'ACADL':  'FAO',  'ACADVL': 'FAO',
    'ACADS':   'FAO',  'ACAA2':  'FAO',  'HADHA':  'FAO',
    'HADHB':   'FAO',  'ECHS1':  'FAO',  'EHHADH': 'FAO',
    'CPT1A':   'FAO',  'CPT1B':  'FAO',  'CPT2':   'FAO',
    # Purine/pyrimidine metabolism — RNA/DNA synthesis in expanding clones
    'ADSL':    'Nucleotide',  'ADSS1': 'Nucleotide',  'ADSS2':  'Nucleotide',
    'PFAS':    'Nucleotide',  'GART':  'Nucleotide',  'PAICS':  'Nucleotide',
    'ATIC':    'Nucleotide',  'TYMS':  'Nucleotide',  'DHFR':   'Nucleotide',
    'UMPS':    'Nucleotide',  'CAD':   'Nucleotide',  'DHODH':  'Nucleotide',
    # One-carbon / folate — linked to DNMT3A mutation effects
    'MTHFR':   'One-carbon',  'MTR':    'One-carbon',  'MTRR':  'One-carbon',
    'SHMT1':   'One-carbon',  'SHMT2':  'One-carbon',  'ALDH1L1':'One-carbon',
    # Amino acid catabolism / anabolism — mTOR sensing
    'GLS':     'Amino acid',  'GLS2':   'Amino acid',  'ASNS':  'Amino acid',
    'PHGDH':   'Amino acid',  'PSAT1':  'Amino acid',  'PSPH':  'Amino acid',
    'SLC1A5':  'Amino acid',  'SLC7A5': 'Amino acid',
    # Glycolysis / gluconeogenesis
    'PFKM':    'Glycolysis',  'PFKL':   'Glycolysis',  'PKM':   'Glycolysis',
    'LDHA':    'Glycolysis',  'HK1':    'Glycolysis',  'HK2':   'Glycolysis',
    'ALDOA':   'Glycolysis',  'ENO1':   'Glycolysis',  'GAPDH': 'Glycolysis',
    'PCK1':    'Glycolysis',  'G6PC':   'Glycolysis',
    # Epigenetic metabolism (relevant to TET2/DNMT3A context)
    'ACLY':    'Epi-metabolite',  'ACACA': 'Epi-metabolite',
    'HMGCR':   'Epi-metabolite',
}

_CH_COLOR = {
    'CH driver':      '#c0392b',
    'HSC metabolic':  '#e67e22',
    'TCA/OxPhos':     '#8e44ad',
    'FAO':            '#16a085',
    'Nucleotide':     '#2980b9',
    'One-carbon':     '#27ae60',
    'Amino acid':     '#f39c12',
    'Glycolysis':     '#d35400',
    'Epi-metabolite': '#7f8c8d',
    'none':           '#bdc3c7',
}

# Metabolic GO root terms to search under
_METAB_GO_ROOTS = {
    'GO:0008152': 'metabolic process',
    'GO:0019752': 'carboxylic acid metabolic process',
    'GO:0006629': 'lipid metabolic process',
    'GO:0006520': 'amino acid metabolic process',
    'GO:0072521': 'purine-containing compound metabolic process',
    'GO:0072527': 'pyrimidine-containing compound metabolic process',
    'GO:0006090': 'pyruvate metabolic process',
    'GO:0006091': 'generation of precursor metabolites and energy',
    'GO:0007005': 'mitochondrion organization',
    'GO:0009056': 'catabolic process',
    'GO:0044281': 'small molecule metabolic process',
}

# Broader metabolic sub-category labels for GO terms
_METAB_KEYWORDS = {
    'fatty acid':      'Fatty acid / lipid',
    'lipid':           'Fatty acid / lipid',
    'oxidation':       'Fatty acid / lipid',
    'carboxylic acid': 'TCA / organic acid',
    'tricarboxylic':   'TCA / organic acid',
    'pyruvate':        'TCA / organic acid',
    'acetyl':          'TCA / organic acid',
    'glucose':         'Glycolysis / glucose',
    'gluconeogenesis': 'Glycolysis / glucose',
    'glycolysis':      'Glycolysis / glucose',
    'hexose':          'Glycolysis / glucose',
    'purine':          'Nucleotide',
    'pyrimidine':      'Nucleotide',
    'nucleotide':      'Nucleotide',
    'nucleobase':      'Nucleotide',
    'amino acid':      'Amino acid',
    'glutamine':       'Amino acid',
    'serine':          'Amino acid',
    'one-carbon':      'One-carbon / folate',
    'folate':          'One-carbon / folate',
    'methionine':      'One-carbon / folate',
    'mitochondri':     'Mitochondria / OxPhos',
    'oxidative phosph':'Mitochondria / OxPhos',
    'electron transport':'Mitochondria / OxPhos',
    'ATP synthesis':   'Mitochondria / OxPhos',
    'reactive oxygen': 'ROS / redox',
    'oxidoreduct':     'ROS / redox',
    'redox':           'ROS / redox',
    'mtor':            'mTOR signaling',
    'autophagy':       'Autophagy',
    'ubiquitin':       'Protein quality',
    'proteasome':      'Protein quality',
    'histone':         'Epigenetic regulation',
    'methylation':     'Epigenetic regulation',
    'chromatin':       'Epigenetic regulation',
}


def _go_subcategory(term_name: str) -> str:
    term_lower = term_name.lower()
    for kw, cat in _METAB_KEYWORDS.items():
        if kw in term_lower:
            return cat
    return 'Other metabolic'


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_genes_with_editcount(path: str):
    """Returns dict gene→edit_site_count."""
    counts = defaultdict(int)
    with open(path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            fields = line.split('\t')
            if len(fields) > 3 and fields[3] not in ('.', ''):
                counts[fields[3]] += 1
    return dict(counts)


def get_gene_go_annotations(gene_list: List[str]) -> Dict[str, List[dict]]:
    """Fetch GO annotations for genes via mygene.info."""
    import mygene
    logger.info(f"  Querying mygene.info for {len(gene_list)} genes...")
    mg = mygene.MyGeneInfo()
    results = mg.querymany(
        gene_list, scopes='symbol', fields='go,name,summary',
        species='human', returnall=False, verbose=False
    )
    gene_go = {}
    for r in results:
        if 'notfound' in r or 'go' not in r:
            continue
        gene = r['query']
        go_terms = []
        for cat in ('BP', 'MF', 'CC'):
            entries = r['go'].get(cat, [])
            if isinstance(entries, dict):
                entries = [entries]
            for e in entries:
                go_terms.append({
                    'id': e.get('id', ''),
                    'term': e.get('term', ''),
                    'category': cat,
                    'evidence': e.get('evidence', ''),
                })
        gene_go[gene] = go_terms
    return gene_go


def is_metabolic(go_terms: List[dict]) -> bool:
    """True if gene has any metabolic GO annotation."""
    for t in go_terms:
        name_lower = t['term'].lower()
        if any(kw in name_lower for kw in [
            'metabol', 'biosynthes', 'catabol', 'oxidat', 'fatty acid',
            'glycoly', 'gluconeo', 'tricarbox', 'citric acid', 'pyruvat',
            'oxidative phosph', 'electron transport', 'mitochondri',
            'amino acid', 'purine', 'pyrimidine', 'nucleotide', 'lipid',
        ]):
            return True
    return False


def get_metabolic_subcats(go_terms: List[dict]) -> List[str]:
    """Return metabolic sub-categories for a gene."""
    cats = set()
    for t in go_terms:
        cat = _go_subcategory(t['term'])
        if cat != 'Other metabolic' or any(
            kw in t['term'].lower() for kw in ['metabol', 'biosynthes', 'catabol']
        ):
            cats.add(cat)
    return sorted(cats)


# ─────────────────────────────────────────────────────────────────────────────
# GO enrichment for metabolic genes
# ─────────────────────────────────────────────────────────────────────────────

def run_metabolic_go(gene_set: Set[str], label: str) -> pd.DataFrame:
    from gprofiler import GProfiler
    if len(gene_set) < 5:
        return pd.DataFrame()
    gp = GProfiler(return_dataframe=True)
    df = gp.profile(
        organism='hsapiens', query=sorted(gene_set),
        sources=['GO:BP', 'GO:MF', 'KEGG', 'REAC'],
        significance_threshold_method='fdr', user_threshold=0.1,
        no_iea=False,
    )
    if df.empty:
        return df
    # Keep metabolic terms only
    mask = df['name'].str.lower().str.contains(
        'metabol|biosynthes|catabol|oxidat|fatty acid|glycoly|gluconeo|'
        'tricarbox|citric|pyruvat|phosphoryl|electron trans|mitochondri|'
        'amino acid|purine|pyrimidine|nucleotide|lipid|mtor|autophagy|'
        'redox|reactive oxygen|serine|glutamin|folate|methyl',
        regex=True
    )
    df = df[mask | df['source'].isin(['KEGG', 'REAC'])].copy()
    df['query_set'] = label
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Panels
# ─────────────────────────────────────────────────────────────────────────────

def _panel_go_heatmap(ax, go_frames: dict, n_top: int = 25):
    """Heatmap: metabolic GO terms (rows) × gene-sets (cols), −log10 p."""
    all_terms = set()
    for df in go_frames.values():
        if not df.empty:
            all_terms.update(df.nsmallest(n_top, 'p_value')['name'].tolist())

    if not all_terms:
        ax.text(0.5, 0.5, 'No metabolic GO terms', ha='center', va='center',
                transform=ax.transAxes); ax.axis('off'); return

    cols = list(go_frames.keys())
    mat = pd.DataFrame(index=sorted(all_terms), columns=cols, dtype=float).fillna(0)
    for col, df in go_frames.items():
        if df.empty:
            continue
        for _, row in df.iterrows():
            if row['name'] in mat.index:
                mat.loc[row['name'], col] = max(
                    mat.loc[row['name'], col],
                    -np.log10(max(row['p_value'], 1e-30))
                )

    # Cluster rows by max value
    mat = mat.loc[mat.max(axis=1).sort_values(ascending=False).index]
    mat = mat.head(n_top)

    cmap = LinearSegmentedColormap.from_list('wh_pur', ['#ffffff', '#8e44ad'])
    sns.heatmap(mat, ax=ax, cmap=cmap, linewidths=0.3, linecolor='#eee',
                cbar_kws={'label': '−log₁₀(FDR)', 'shrink': 0.7},
                annot=True, fmt='.1f', annot_kws={'size': 7})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=10)
    ax.set_yticklabels(
        [f"{n[:55]}…" if len(n) > 55 else n for n in mat.index],
        fontsize=8
    )
    ax.set_title('Metabolic GO / Pathway Enrichment (−log₁₀ FDR)',
                 fontsize=12, fontweight='bold')


def _panel_gene_bubble(ax, gene_df: pd.DataFrame):
    """
    Bubble plot: x = S161 edit count, y = S206 edit count.
    Size ∝ WT edit count (0 if not a WT target).
    Colour = CH relevance category.
    """
    if gene_df.empty:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes); ax.axis('off'); return

    for cat, color in _CH_COLOR.items():
        sub = gene_df[gene_df['ch_category'] == cat]
        if sub.empty:
            continue
        sizes = 40 + sub['wt_count'] * 15
        ax.scatter(
            sub['s161_count'], sub['s206_count'],
            s=sizes.clip(upper=800), c=color, alpha=0.75,
            edgecolors='white', linewidths=0.5, label=cat, zorder=3
        )
    # Label notable genes
    notable = gene_df[gene_df['ch_category'].isin(
        ['CH driver', 'HSC metabolic', 'FAO', 'TCA/OxPhos', 'Nucleotide']
    )].nlargest(30, 'mut_total')
    for _, row in notable.iterrows():
        ax.annotate(row['gene'], (row['s161_count'], row['s206_count']),
                    fontsize=7, xytext=(4, 2), textcoords='offset points',
                    color='#333', fontweight='bold' if row['ch_category'] == 'CH driver' else 'normal')

    ax.set_xlabel('S161-PUF60 editing site count')
    ax.set_ylabel('S206-PUF60 editing site count')
    ax.set_title('Metabolic Target Genes: S161 vs S206 Editing Frequency\n'
                 '(size ∝ WT count; labelled = CH/metabolic interest)',
                 fontsize=11, fontweight='bold')
    ax.axline((0, 0), slope=1, color='#aaa', linestyle='--', linewidth=0.8, zorder=0)
    legend_handles = [
        mpatches.Patch(facecolor=c, label=cat, alpha=0.8)
        for cat, c in _CH_COLOR.items() if cat != 'none'
    ]
    ax.legend(handles=legend_handles, fontsize=7, ncol=2,
              loc='upper left', frameon=True, framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _panel_metab_venn(ax, sets_metab: dict):
    labels = list(sets_metab.keys())
    if len(labels) < 3:
        ax.axis('off'); return
    vals = [sets_metab[l] for l in labels]
    v = venn3(vals, set_labels=labels, ax=ax,
              set_colors=[_PALETTE.get(l, '#888') for l in labels],
              alpha=0.55)
    venn3_circles(vals, ax=ax, linewidth=1.2)
    if v:
        for t in (v.set_labels or []):
            if t: t.set_fontsize(11)
        for t in (v.subset_labels or []):
            if t: t.set_fontsize(9)
    ax.set_title('Metabolic Target Gene Overlap', fontsize=12, fontweight='bold')


def _panel_subcategory_bar(ax, gene_df: pd.DataFrame):
    """Stacked bar of metabolic sub-categories per condition."""
    conditions = ['WT', 'S161', 'S206']
    cat_cols = sorted(_METAB_KEYWORDS.values())
    # count genes per (condition, subcat)
    data = {c: defaultdict(int) for c in conditions}
    for _, row in gene_df.iterrows():
        for subcat in row['subcats']:
            if 'WT' in row['present_in']:
                data['WT'][subcat] += 1
            if 'S161' in row['present_in']:
                data['S161'][subcat] += 1
            if 'S206' in row['present_in']:
                data['S206'][subcat] += 1

    unique_cats = sorted(set(
        cat for counts in data.values() for cat in counts
    ))
    cat_colors = sns.color_palette('tab20', len(unique_cats))

    x = np.arange(len(conditions))
    bottoms = np.zeros(len(conditions))
    for i, cat in enumerate(unique_cats):
        vals = [data[c].get(cat, 0) for c in conditions]
        bars = ax.bar(x, vals, bottom=bottoms, label=cat,
                      color=cat_colors[i], edgecolor='white', linewidth=0.4)
        bottoms += np.array(vals, dtype=float)

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=11)
    ax.set_ylabel('Gene count (metabolic)')
    ax.set_title('Metabolic Sub-Category Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc='upper left', frameon=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _panel_ch_table(ax, gene_df: pd.DataFrame):
    """Table of CH-relevant metabolic genes with edit counts and annotation."""
    ch_relevant = gene_df[
        gene_df['ch_category'].isin(
            ['CH driver', 'HSC metabolic', 'FAO', 'TCA/OxPhos', 'Nucleotide',
             'One-carbon', 'Amino acid', 'Glycolysis', 'Epi-metabolite']
        )
    ].sort_values(['ch_category', 'mut_total'], ascending=[True, False])

    if ch_relevant.empty:
        ax.text(0.5, 0.5, 'No CH-relevant metabolic targets found',
                ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
        return

    # Limit rows
    ch_relevant = ch_relevant.head(30)
    ax.axis('off')
    cols = ['gene', 'ch_category', 'wt_count', 's161_count', 's206_count',
            'in_mutants_only', 'top_go']
    col_labels = ['Gene', 'CH Category', 'WT sites', 'S161 sites', 'S206 sites',
                  'Mutant only', 'Top GO term']

    disp = ch_relevant[cols].copy()
    disp['in_mutants_only'] = disp['in_mutants_only'].map({True: 'Yes', False: 'No'})
    disp['top_go'] = disp['top_go'].apply(lambda x: x[:40] + '…' if len(x) > 40 else x)

    table = ax.table(
        cellText=disp.values,
        colLabels=col_labels,
        cellLoc='left',
        loc='center',
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)

    # Colour header
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor('#2c3e50')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    # Colour rows by CH category
    for i, (_, row_data) in enumerate(disp.iterrows(), start=1):
        cat = row_data['ch_category']
        fc = _CH_COLOR.get(cat, '#ffffff')
        for j in range(len(col_labels)):
            table[(i, j)].set_facecolor(fc + '40')  # 25% alpha hex

    ax.set_title('CH-Relevant Metabolic Targets in PUF60 Mutants',
                 fontsize=11, fontweight='bold', pad=10)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def analyze(wt_bed: str, s161_bed: str, s206_bed: str,
            output: str, table_output: str):

    # 1. Load gene sets with edit counts
    wt_counts   = load_genes_with_editcount(wt_bed)
    s161_counts = load_genes_with_editcount(s161_bed)
    s206_counts = load_genes_with_editcount(s206_bed)

    wt_genes   = set(wt_counts)
    s161_genes = set(s161_counts)
    s206_genes = set(s206_counts)

    all_genes = wt_genes | s161_genes | s206_genes
    logger.info(f"Total unique target genes across all conditions: {len(all_genes):,}")

    # 2. Fetch GO annotations
    logger.info("Fetching GO annotations from mygene.info...")
    gene_go = get_gene_go_annotations(sorted(all_genes))
    logger.info(f"  Got annotations for {len(gene_go):,} genes")

    # 3. Build gene-level table
    records = []
    for gene in sorted(all_genes):
        go_terms = gene_go.get(gene, [])
        metab = is_metabolic(go_terms)
        if not metab:
            continue
        subcats = get_metabolic_subcats(go_terms)
        ch_cat = _CH_GENES.get(gene, 'none')
        wt_c   = wt_counts.get(gene, 0)
        s161_c = s161_counts.get(gene, 0)
        s206_c = s206_counts.get(gene, 0)
        present = []
        if gene in wt_genes:   present.append('WT')
        if gene in s161_genes: present.append('S161')
        if gene in s206_genes: present.append('S206')
        top_go = ''
        for t in go_terms:
            if any(kw in t['term'].lower() for kw in ['metabol', 'biosynthes', 'oxidat', 'fatty']):
                top_go = t['term']
                break
        records.append({
            'gene':            gene,
            'ch_category':     ch_cat,
            'wt_count':        wt_c,
            's161_count':      s161_c,
            's206_count':      s206_c,
            'mut_total':       s161_c + s206_c,
            'present_in':      present,
            'in_mutants_only': (gene not in wt_genes),
            'subcats':         subcats,
            'top_go':          top_go,
            'n_go_terms':      len(go_terms),
        })

    gene_df = pd.DataFrame(records)
    logger.info(f"Metabolic target genes: {len(gene_df):,} "
                f"({(gene_df['in_mutants_only']).sum()} mutant-only)")

    # Save table
    if table_output:
        save_df = gene_df.copy()
        save_df['subcats']    = save_df['subcats'].apply(lambda x: '; '.join(x))
        save_df['present_in'] = save_df['present_in'].apply(lambda x: '; '.join(x))
        Path(table_output).parent.mkdir(parents=True, exist_ok=True)
        save_df.drop(columns=['n_go_terms']).to_csv(table_output, sep='\t', index=False)
        logger.info(f"Gene table: {table_output}")

    # 4. Metabolic gene sets per condition
    metab_wt   = set(gene_df[gene_df['wt_count']   > 0]['gene'])
    metab_s161 = set(gene_df[gene_df['s161_count'] > 0]['gene'])
    metab_s206 = set(gene_df[gene_df['s206_count'] > 0]['gene'])

    # 5. GO enrichment on mutant-unique metabolic genes
    logger.info("Running GO enrichment on metabolic gene subsets...")
    s161_metab_unique  = metab_s161 - metab_wt - metab_s206
    s206_metab_unique  = metab_s206 - metab_wt - metab_s161
    shared_mut_metab   = (metab_s161 & metab_s206) - metab_wt
    core_metab         = metab_wt & metab_s161 & metab_s206

    go_frames = {}
    for label, gs in [
        ('WT metabolic',       metab_wt),
        ('S161 metabolic',     metab_s161),
        ('S206 metabolic',     metab_s206),
        ('S161∩S206 (not WT)', shared_mut_metab),
    ]:
        logger.info(f"  {label}: {len(gs)} genes")
        go_frames[label] = run_metabolic_go(gs, label)
        time.sleep(0.4)

    # Log CH-relevant hits
    ch_hits = gene_df[gene_df['ch_category'] != 'none']
    logger.info(f"\nCH-relevant metabolic targets ({len(ch_hits)}):")
    for _, row in ch_hits.sort_values('mut_total', ascending=False).head(30).iterrows():
        logger.info(f"  {row['gene']:12s} [{row['ch_category']:16s}] "
                    f"WT:{row['wt_count']:3d} S161:{row['s161_count']:3d} "
                    f"S206:{row['s206_count']:3d}  "
                    f"mutant-only:{row['in_mutants_only']}")

    # 6. Build figure
    fig = plt.figure(figsize=(22, 32))
    gs_fig = gridspec.GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.38)

    ax_heatmap  = fig.add_subplot(gs_fig[0, :])
    ax_bubble   = fig.add_subplot(gs_fig[1, 0])
    ax_venn     = fig.add_subplot(gs_fig[1, 1])
    ax_subcat   = fig.add_subplot(gs_fig[2, 0])
    ax_chtable  = fig.add_subplot(gs_fig[2, 1])
    ax_notes    = fig.add_subplot(gs_fig[3, :])

    _panel_go_heatmap(ax_heatmap, go_frames)
    _panel_gene_bubble(ax_bubble, gene_df)
    _panel_metab_venn(ax_venn, {'WT': metab_wt, 'S161': metab_s161, 'S206': metab_s206})
    _panel_subcategory_bar(ax_subcat, gene_df)
    _panel_ch_table(ax_chtable, gene_df)

    # Notes panel
    ax_notes.axis('off')
    notes = (
        "Interpretation Notes\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "PUF60 S161 and S206 are point mutations in the RRM/UHM RNA-binding domain found in clonal haematopoiesis (CH) datasets.\n"
        "As a U2AF65-like factor, PUF60 stabilises U2AF65 binding at weak 3′ splice sites, particularly polypyrimidine-tract-\n"
        "deficient introns. Gain-of-function mutations may alter 3′SS selection and/or RBP–RNA affinity.\n\n"
        "Metabolic relevance to CH cell survival:\n"
        "  • FATTY ACID OXIDATION (FAO): HSCs and CHIP clones rely on FAO for quiescence maintenance (CPT1A, ACADM, HADHB).\n"
        "    Increased editing of FAO genes may alter splicing of rate-limiting steps.\n"
        "  • TCA CYCLE / OxPhos: IDH1/IDH2 mutations define AML subtypes; TCA dysregulation drives epigenetic reprogramming\n"
        "    through 2-HG production and altered α-KG levels, linking to TET2/DNMT3A effects.\n"
        "  • PURINE/PYRIMIDINE SYNTHESIS: Expanding clones have elevated nucleotide demand; ADSL, GART, DHODH are rate-limiting.\n"
        "  • mTOR/PI3K AXIS: mTOR senses nutrient status to control HSC quiescence vs. proliferation (MTOR, RPTOR, AKT).\n"
        "  • ONE-CARBON METABOLISM: SHMT2, MTHFR connect to SAM/SAH ratios — directly impacting DNMT3A-mediated methylation.\n\n"
        "Note: HyperTRIBE editing sites mark protein–RNA contact regions, not necessarily disrupted sites. Increased editing of\n"
        "a metabolic gene indicates the mutant PUF60 binds its pre-mRNA more avidly or at a new location, potentially altering\n"
        "splicing efficiency, exon inclusion, or intron retention for that gene."
    )
    ax_notes.text(0.01, 0.99, notes, transform=ax_notes.transAxes,
                  fontsize=9.5, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle='round,pad=0.6', facecolor='#f8f9fa',
                            edgecolor='#aaa', linewidth=1))

    fig.suptitle(
        "PUF60 Mutant (S161 / S206) — Metabolic RNA Target Analysis\n"
        "Context: Clonal Haematopoiesis & HSC Metabolic Reprogramming",
        fontsize=15, fontweight='bold', y=1.002
    )
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output}")


def main():
    parser = argparse.ArgumentParser(
        description='Metabolic GO analysis for PUF60 mutants — CH context',
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__
    )
    parser.add_argument('--wt',    required=True, help='WT annotated editing BED')
    parser.add_argument('--s161',  required=True, help='S161 annotated editing BED')
    parser.add_argument('--s206',  required=True, help='S206 annotated editing BED')
    parser.add_argument('--output', required=True, help='Output PDF')
    parser.add_argument('--table', default='', help='Output gene TSV (optional)')
    args = parser.parse_args()
    analyze(args.wt, args.s161, args.s206, args.output, args.table)


if __name__ == '__main__':
    main()
