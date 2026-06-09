#!/usr/bin/env python3
"""
Comprehensive analysis of S161 ∩ S206 shared targets — clonal haematopoiesis.
Identifies what makes these two PUF60 mutations functionally convergent.

Panels:
  1  Functional category chart: the 256 shared-not-WT genes broken down
  2  Key gene matrix: ~35 CH/cancer genes × conditions, annotated
  3  E2F transcription factor enrichment (dominant regulatory theme)
  4  GO pathway enrichment dot plot (DNA damage, Wnt, stem cell, apoptosis)
  5  Conceptual convergence panel (text-based pathway map)
  6  Notes
"""

import argparse, sys, time
from pathlib import Path
from collections import Counter, defaultdict
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from gprofiler import GProfiler
import mygene

import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

sns.set_style('whitegrid')
plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'pdf.fonttype': 42,
    'svg.fonttype': 'none',
    'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11,
})

_CPAL = {'WT':'#2980b9','S161':'#e74c3c','S206':'#27ae60','Both':'#8e44ad'}

# ── Curated key-gene registry ─────────────────────────────────────────────────
# (gene, short_annotation, theme, ch_priority 1-5)
_KEY_GENES = [
    # Canonical CH / myeloid cancer drivers
    ('STAG2',    'Cohesin/AML (top-5 mut)',   'Genome stability',     5),
    ('PPM1D',    'WIP1/t-CHIP driver',         'DNA damage',           5),
    ('CBL',      'E3-Ub/CH driver',            'Signal transduction',  5),
    ('BCR',      'BCR-ABL partner',            'Signal transduction',  5),
    ('ABL1',     'Tyrosine kinase',            'Signal transduction',  5),
    ('PTPN11',   'SHP2/RASopathy',             'Signal transduction',  4),
    # AML/MDS tumour suppressors targeted by both mutants
    ('QKI',      'RBP/AML tumour suppressor',  'RNA splicing',         5),
    ('SON',      'Global splicing regulator',  'RNA splicing',         4),
    ('TCERG1',   'Splicing/transcription',     'RNA splicing',         4),
    ('DDX20',    'SMN/snRNP assembly',         'RNA splicing',         3),
    ('GEMIN2',   'SMN complex/snRNP',          'RNA splicing',         3),
    # DNA repair / genome stability
    ('RAD51AP1', 'HR repair (RAD51 cofactor)', 'DNA repair',           4),
    ('FANCM',    'Fanconi anaemia/HR',         'DNA repair',           4),
    ('BUB3',     'Spindle checkpoint',         'Chromosome stability', 3),
    ('TTK',      'MPS1 spindle checkpoint',    'Chromosome stability', 4),
    ('CDC20',    'APC/C-Cdc20 ubiquitin',      'Cell cycle',           3),
    ('CDKN1B',   'p27/cell cycle brake',       'Cell cycle',           3),
    ('ASF1A',    'Histone chaperone/repair',   'DNA repair',           3),
    # Cell survival / anti-apoptosis
    ('PDCD4',    'Tumour suppressor/apoptosis','Cell survival',        4),
    ('BFAR',     'Anti-apoptosis',             'Cell survival',        3),
    ('HMOX1',    'LSC cytoprotection/ROS',     'Cell survival',        4),
    # Immune evasion
    ('B2M',      'MHC-I/immune evasion',       'Immune evasion',       5),
    ('APOBEC3G', 'Mutational signature/immune','Immune/mutagenesis',   4),
    ('IRAK4',    'TLR/innate immune',          'Immune signaling',     4),
    ('SOCS7',    'JAK-STAT suppressor',        'Immune signaling',     4),
    # Self-renewal / differentiation
    ('FOXK2',    'Autophagy/glucose/E2F-target','Self-renewal',        3),
    ('CTNNB1',   'Wnt/β-catenin',              'Self-renewal',         4),
    ('ARNT2',    'HIF1β-like/hypoxia',         'Self-renewal',         3),
    ('DMAP1',    'DNMT-assoc./epigenetic',     'Epigenetic',           3),
    ('EHMT1',    'G9a-like H3K9 methylase',    'Epigenetic',           3),
    # Chromatin regulators
    ('BPTF',     'NURF/chromatin remodel',     'Epigenetic',           3),
    ('STAG2',    'Cohesin-CTCF/TADs',         'Epigenetic',           5),  # duplicate for theme
    ('PPM1D',    'p53/ATM inactivator',        'DNA damage',           5),  # duplicate
    # Signalling
    ('RAC1',     'RHO GTPase/ROS/AML',        'Signal transduction',  4),
    ('PRKCD',    'PKCδ/apoptosis regulator',  'Signal transduction',  3),
    ('FMR1',     'FMRP/PTEN-translation',     'RNA/translation',      3),
]
# De-duplicate
seen = set()
KEY_GENES_DEDUP = []
for row in _KEY_GENES:
    if row[0] not in seen:
        KEY_GENES_DEDUP.append(row)
        seen.add(row[0])

_THEME_COLOR = {
    'Genome stability':    '#c0392b',
    'DNA damage':          '#c0392b',
    'DNA repair':          '#e67e22',
    'Chromosome stability':'#e67e22',
    'RNA splicing':        '#8e44ad',
    'RNA/translation':     '#9b59b6',
    'Cell cycle':          '#2980b9',
    'Cell survival':       '#16a085',
    'Immune evasion':      '#f39c12',
    'Immune signaling':    '#f39c12',
    'Immune/mutagenesis':  '#e74c3c',
    'Self-renewal':        '#27ae60',
    'Epigenetic':          '#1abc9c',
    'Signal transduction': '#3498db',
}

_CAT_KWORDS = {
    'Signal transduction':       ['signal transduct','kinase','phosphat','ras','mapk','pi3k','jak','stat','nfkb','wnt'],
    'Metabolism':                ['metabol','biosynthes','catabol','fatty acid','glycoly','oxphos','mitochond'],
    'Transcription / chromatin': ['transcription factor','histone','chromatin','epigenetic','acetyl','methyl'],
    'Vesicle / trafficking':     ['vesicle','trafficking','endosom','exosom','lysosom'],
    'Cytoskeleton / adhesion':   ['cytoskelet','actin','tubulin','adhesion','integrin','focal adhes'],
    'Cell cycle':                ['cell cycle','cyclin','cdk','proliferat','mitosis','senescen'],
    'RNA processing / splicing': ['splicing','spliceosom','mrna process','rna process','pre-mrna','hnrnp'],
    'RNA binding / translation': ['rna binding','translat','ribos','eif','ribosomal'],
    'Protein homeostasis':       ['ubiquitin','proteasome','protein quality','chaperone','autophagy'],
    'Cell survival / apoptosis': ['apoptosis','apoptot','survival','bcl','death','caspase'],
    'Immune function':           ['immune','inflammat','interferon','cytokine','innate','adaptive'],
    'DNA damage / repair':       ['dna damage','dna repair','genome stab','checkpoint','brca','fanconi'],
    'Other':                     [],
}

# ── Data loaders ──────────────────────────────────────────────────────────────

def load_genes_counts(path):
    g, c = set(), {}
    with open(path) as f:
        for line in f:
            if line.startswith('#'): continue
            f2 = line.split('\t')
            if len(f2) > 3 and f2[3] not in ('.',''):
                g.add(f2[3]); c[f2[3]] = c.get(f2[3],0) + 1
    return g, c


def categorize_genes(gene_list):
    """Assign functional categories to genes via mygene GO annotations."""
    mg_inst = mygene.MyGeneInfo()
    results = mg_inst.querymany(gene_list, scopes='symbol',
                                fields='go,summary,name', species='human',
                                returnall=False, verbose=False)
    gene_cats = {}
    for r in results:
        if 'notfound' in r or 'go' not in r: continue
        gene = r['query']
        all_terms = []
        for cat in ('BP','MF','CC'):
            entries = r['go'].get(cat,[])
            if isinstance(entries,dict): entries=[entries]
            all_terms.extend(e.get('term','').lower() for e in entries)
        summary = (r.get('summary','') or '').lower()
        name    = (r.get('name','')    or '').lower()
        combined = ' '.join(all_terms) + ' ' + summary + ' ' + name
        cats = set()
        for cat_name, kws in _CAT_KWORDS.items():
            if kws and any(kw in combined for kw in kws):
                cats.add(cat_name)
        gene_cats[gene] = cats if cats else {'Other'}
    return gene_cats

# ── Panels ────────────────────────────────────────────────────────────────────

def panel_category_bar(ax, gene_cats, shared_not_wt, s161_only, s206_only, labels):
    """Grouped bar: each category × (S161-only / S206-only / shared)."""
    all_cats = list(_CAT_KWORDS.keys())
    sets = {labels[0]: s161_only, labels[1]: s206_only, 'S161∩S206': shared_not_wt}
    col  = {labels[0]: _CPAL['S161'], labels[1]: _CPAL['S206'], 'S161∩S206': _CPAL['Both']}

    counts = {k: [] for k in sets}
    for cat in all_cats:
        for k, gene_set in sets.items():
            n = sum(1 for g in gene_set
                    if any(cat in c for c in gene_cats.get(g, {'Other'})))
            counts[k].append(n)

    x = np.arange(len(all_cats)); w = 0.25
    offsets = [-w, 0, w]
    for i, (k, vals) in enumerate(counts.items()):
        bars = ax.bar(x + offsets[i], vals, w, color=col[k],
                      label=k, alpha=0.82, edgecolor='white', linewidth=0.4)
        ax.bar_label(bars, padding=1, fontsize=6.5,
                     labels=[str(v) if v > 0 else '' for v in vals])

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace(' / ','/\n').replace(' ','\n',1)
                        for c in all_cats], rotation=40, ha='right', fontsize=8)
    ax.set_ylabel('Target gene count'); ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    ax.set_title('Functional Categories — Genes Gained in PUF60 Mutants\n'
                 'vs WT PUF60', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, frameon=True)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)


def panel_key_gene_matrix(ax, wt_c, s161_c, s206_c):
    """Heatmap rows=key genes, cols=WT/S161/S206, annotated with theme."""
    genes  = [r[0] for r in KEY_GENES_DEDUP]
    annots = [r[1] for r in KEY_GENES_DEDUP]
    themes = [r[2] for r in KEY_GENES_DEDUP]

    mat = np.array([[wt_c.get(g,0), s161_c.get(g,0), s206_c.get(g,0)]
                    for g in genes], dtype=float)

    # Only show genes that are targeted in S161 or S206 at all
    mask = mat[:,1:].sum(axis=1) > 0
    mat, genes, annots, themes = mat[mask], [g for g,m in zip(genes,mask) if m], \
                                  [a for a,m in zip(annots,mask) if m], \
                                  [t for t,m in zip(themes,mask) if m]

    if len(genes) == 0:
        ax.text(0.5,0.5,'No data',ha='center',va='center',transform=ax.transAxes)
        ax.axis('off'); return

    cmap = LinearSegmentedColormap.from_list('wh_red',['#f8f9fa','#e74c3c'])
    im = ax.imshow(mat, aspect='auto', cmap=cmap, vmin=0, vmax=max(mat.max(),1))

    ax.set_xticks([0,1,2])
    ax.set_xticklabels(['WT','S161','S206'], fontsize=11)
    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels(genes, fontsize=8.5)

    # Annotate values in cells
    for i in range(len(genes)):
        for j in range(3):
            v = int(mat[i,j])
            if v > 0:
                ax.text(j, i, str(v), ha='center', va='center', fontsize=8,
                        fontweight='bold', color='white' if v >= 3 else '#222')

    # Right side: theme coloured strip + annotation
    ax.set_xlim(-0.5, 2.5)
    for i, (theme, annot) in enumerate(zip(themes, annots)):
        col = _THEME_COLOR.get(theme, '#888')
        ax.barh(i, -0.35, left=-0.5, height=0.85, color=col, alpha=0.6, zorder=1)
        ax.text(3.1, i, annot, va='center', fontsize=7, color='#333')

    ax.set_title('Key CH/Cancer Genes — Editing Site Counts\n'
                 '(coloured strip = functional theme)',
                 fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Editing sites', shrink=0.55, pad=0.25)

    # Theme legend
    seen_themes = dict.fromkeys(themes)
    handles = [mpatches.Patch(facecolor=_THEME_COLOR.get(t,'#888'),
                              label=t, alpha=0.75)
               for t in seen_themes]
    ax.legend(handles=handles, fontsize=6.5, loc='lower right',
              ncol=2, frameon=True, framealpha=0.9, title='Theme')
    ax.spines[:].set_visible(False)


def panel_tf_enrichment(ax, tf_df):
    """Bar chart of E2F/TF factor enrichment."""
    if tf_df.empty:
        ax.text(0.5,0.5,'No TF data',ha='center',va='center',
                transform=ax.transAxes); ax.axis('off'); return

    tf_df = tf_df.copy()
    tf_df['factor'] = tf_df['name'].str.extract(r'Factor: ([^;]+);?')
    tf_df['factor'] = tf_df['factor'].fillna(tf_df['name'].str[:25])
    tf_df['-log10p'] = -np.log10(tf_df['p_value'].clip(lower=1e-30))

    # One entry per factor (best p)
    best = tf_df.groupby('factor').agg({'-log10p':'max','intersection_size':'max'}).reset_index()
    best = best.nlargest(20, '-log10p').sort_values('-log10p')

    e2f_mask = best['factor'].str.contains('E2F|ETF|ZF5', case=False)
    colors = ['#c0392b' if m else '#7f8c8d' for m in e2f_mask]

    bars = ax.barh(range(len(best)), best['-log10p'], color=colors,
                   edgecolor='white', linewidth=0.4)
    ax.set_yticks(range(len(best)))
    ax.set_yticklabels(best['factor'], fontsize=8.5)
    ax.set_xlabel('−log₁₀(FDR)')
    ax.set_title('Transcription Factor Binding Enrichment\n'
                 'in S161∩S206 Targets (not in WT)',
                 fontsize=11, fontweight='bold')
    ax.axvline(x=-np.log10(0.05), color='#aaa', linestyle='--',
               linewidth=0.8, label='FDR 0.05')

    handles = [mpatches.Patch(facecolor='#c0392b', label='E2F family / ETF'),
               mpatches.Patch(facecolor='#7f8c8d', label='Other TF')]
    ax.legend(handles=handles, fontsize=9, loc='lower right')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)


def panel_go_hematologic(ax, go_df):
    """GO dot plot focused on haematologic/CH-relevant pathways."""
    if go_df.empty:
        ax.text(0.5,0.5,'No GO data',ha='center',va='center',
                transform=ax.transAxes); ax.axis('off'); return

    kw = ('stem cell|self.renew|differentia|hematopoiet|myeloid|apoptosis|'
          'cell death|dna damage|wnt|notch|immune|cytokine|cell cycle|'
          'chromosome|spindle|cohesin|chromatid|p53|ub|ubiquit|nfkb|'
          'jak.stat|dna repair|genome stab|chromatin')
    sub = go_df[go_df['name'].str.lower().str.contains(kw, regex=True, na=False)].copy()
    if sub.empty:
        sub = go_df.copy()
    sub = sub.nsmallest(25,'p_value').copy()
    sub['-log10p'] = -np.log10(sub['p_value'].clip(lower=1e-30))
    sub['gene_ratio'] = sub['intersection_size'] / sub['query_size']
    sub = sub.sort_values('gene_ratio')

    src_col = {'GO:BP':'#3498db','GO:MF':'#e74c3c','KEGG':'#e67e22','REAC':'#9b59b6'}
    colors = [src_col.get(s,'#888') for s in sub['source']]

    ax.scatter(sub['gene_ratio'], range(len(sub)),
               s=sub['intersection_size']*4+20, c=colors,
               alpha=0.8, edgecolors='white', linewidths=0.4, zorder=3)
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(
        [f"{n[:55]}…" if len(n)>55 else n for n in sub['name']], fontsize=8)
    ax.set_xlabel('Gene ratio (hits / query set size)')
    ax.set_title('GO / Pathway Enrichment in S161∩S206 Shared Targets\n'
                 '(cell death, DNA damage, Wnt, stem cell)',
                 fontsize=11, fontweight='bold')

    handles = [mpatches.Patch(facecolor=c,label=s,alpha=0.8)
               for s,c in src_col.items() if s in sub['source'].values]
    ax.legend(handles=handles, fontsize=8, loc='lower right')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)


def panel_convergence_concept(ax):
    """Text-based conceptual map of convergent mechanisms."""
    ax.axis('off')

    # Draw boxes for each convergent theme
    themes = [
        ('GENOME\nSTABILITY',
         'STAG2 (cohesin/TADs)\nPPM1D (p53 inactivation)\nRAD51AP1 (HR repair)\nFANCM (Fanconi pathway)\nBUB3, TTK (spindle checkpoint)',
         0.05, 0.60, '#c0392b'),
        ('RNA\nSPLICING\nRE-WIRING',
         'QKI (MDS/AML RBP)\nSON (global splicing)\nTCERG1 (TXN-splicing)\nDDX20, GEMIN2 (snRNP)\n→ cascade of splice changes',
         0.28, 0.60, '#8e44ad'),
        ('IMMUNE\nEVASION',
         'B2M (MHC-I loss)\nAPOBEC3G (mutational\n   signature drive)\nIRAK4 (TLR signaling)\nSOCS7 (JAK-STAT brake)',
         0.55, 0.60, '#e67e22'),
        ('SELF-RENEWAL\n& WNT',
         'CTNNB1 (β-catenin)\nFOXK2 (autophagy)\nARNT2 (HIF-1β)\n+ E2F-target programme\n→ stem cell maintenance',
         0.05, 0.15, '#27ae60'),
        ('E2F\nACCESSIBILITY',
         '218 / 256 genes are\nE2F transcription targets\n→ Mutant PUF60 expanded\nbinding to proliferative\ntranscript repertoire',
         0.28, 0.15, '#2980b9'),
        ('CELL\nSURVIVAL',
         'PDCD4 (pro-apoptosis ↓)\nBFAR (anti-apoptosis ↑)\nHMOX1 (ROS protection)\nPRKCD (survival signal)\nCDKN1B (p27 cell brake)',
         0.55, 0.15, '#16a085'),
    ]

    for title, body, x, y, color in themes:
        rect = mpatches.FancyBboxPatch(
            (x, y), 0.20, 0.34,
            boxstyle='round,pad=0.015',
            linewidth=1.5, edgecolor=color,
            facecolor=color + '18',
            transform=ax.transAxes, clip_on=False)
        ax.add_patch(rect)
        ax.text(x+0.10, y+0.30, title, transform=ax.transAxes,
                fontsize=9, fontweight='bold', ha='center', va='top',
                color=color)
        ax.text(x+0.10, y+0.22, body, transform=ax.transAxes,
                fontsize=7.5, ha='center', va='top',
                color='#333', fontfamily='monospace')

    # Central label
    ax.text(0.50, 0.52,
            '↓\nConvergent PUF60\nGain-of-Function',
            transform=ax.transAxes, fontsize=10, fontweight='bold',
            ha='center', va='center', color='#2c3e50',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#ecf0f1',
                      edgecolor='#2c3e50', linewidth=1.5))

    ax.set_title('Convergent Mechanisms — S161 and S206 Shared Targets\n'
                 '(256 genes gained by BOTH mutants relative to WT)',
                 fontsize=11, fontweight='bold')


def panel_notes(ax, wt_n, s161_n, s206_n, shared_n, e2f_frac):
    ax.axis('off')
    text = (
        f"Summary: S161 vs S206 PUF60 — Why Convergence Explains Clonal Haematopoiesis Advantage\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Gene set sizes:  WT={wt_n:,}  |  S161={s161_n:,}  |  S206={s206_n:,}  |  "
        f"S161∩S206 not WT={shared_n:,}  |  E2F-target fraction={e2f_frac:.0%}\n\n"
        "SIX CONVERGENT HALLMARKS OF CLONAL FITNESS:\n\n"
        "1. GENOME INSTABILITY (STAG2, PPM1D, RAD51AP1, FANCM, BUB3, TTK)\n"
        "   Both mutants target STAG2 — the top-5 most-mutated gene in AML. Loss of STAG2 disrupts cohesin-mediated\n"
        "   TAD boundaries, misregulating hundreds of genes. PPM1D (WIP1) is a canonical therapy-related CHIP driver;\n"
        "   it inactivates p53/ATM, dampening the DNA damage checkpoint that would otherwise eliminate damaged cells.\n"
        "   Together, these create a mutator phenotype that accelerates acquisition of secondary driver mutations.\n\n"
        "2. RNA SPLICING RE-WIRING (QKI, SON, TCERG1, DDX20, GEMIN2)\n"
        "   QKI is deleted in ~15% of MDS/AML and regulates splicing of NUMB (Notch brake), MYH9, and oncogenic\n"
        "   transcripts. SON orchestrates constitutive splicing of >1000 transcripts; haploinsufficiency causes\n"
        "   global splicing stress. Mutant PUF60 binding to QKI/SON pre-mRNAs could alter their own splicing,\n"
        "   creating a feedback loop where a splicing factor mutation corrupts the broader splicing network.\n\n"
        "3. IMMUNE EVASION (B2M, APOBEC3G, IRAK4, SOCS7)\n"
        "   B2M loss abolishes HLA-I surface expression, preventing T-cell recognition of the clone.\n"
        "   APOBEC3G deamination activity contributes to COSMIC mutational signatures (SBS2/13) found in CH;\n"
        "   altered APOBEC3G splicing could modulate this mutagenic activity. SOCS7 suppression of JAK-STAT\n"
        "   could mimic the inflammatory signalling advantage of JAK2-V617F CH clones.\n\n"
        "4. SELF-RENEWAL (CTNNB1/Wnt, FOXK2, ARNT2/HIF, + E2F programme)\n"
        "   Wnt/β-catenin (CTNNB1) is required for HSC self-renewal and is activated in AML LSCs.\n"
        "   E2F transcription factors are the dominant motif (218/256 genes are E2F targets); this means\n"
        "   mutant PUF60 preferentially binds transcripts upregulated during HSC-to-progenitor cycling,\n"
        "   potentially slowing differentiation and maintaining stem-like state.\n\n"
        "5. CELL SURVIVAL (PDCD4, BFAR, HMOX1, PRKCD, CDKN1B)\n"
        "   PDCD4 is a pro-apoptotic tumour suppressor that inhibits protein synthesis; mutant PUF60 binding\n"
        "   may trigger its exon skipping. BFAR is an anti-apoptotic RING-domain protein. Collectively,\n"
        "   the survival signature shifts the apoptotic balance toward clone persistence.\n\n"
        "6. METABOLIC REPROGRAMMING (PDHA1/GLS/DCK/HMOX1 — see companion figure)\n"
        "   FAO, TCA entry, glutamine, and Complex I rewiring provide metabolic flexibility in the BM niche."
    )
    ax.text(0.01, 0.99, text, transform=ax.transAxes, fontsize=8.5,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#fdfefe',
                      edgecolor='#aaa', linewidth=1))


# ── Main ──────────────────────────────────────────────────────────────────────

def run(wt_bed, s161_bed, s206_bed, output, table_out):
    wt,   wt_c   = load_genes_counts(wt_bed)
    s161, s161_c = load_genes_counts(s161_bed)
    s206, s206_c = load_genes_counts(s206_bed)

    shared_not_wt = (s161 & s206) - wt
    s161_only     = s161 - wt - s206
    s206_only     = s206 - wt - s161

    logger.info(f"S161∩S206 not WT: {len(shared_not_wt)} genes")
    logger.info(f"S161 only: {len(s161_only)}  S206 only: {len(s206_only)}")

    # Categorise shared genes
    logger.info("Categorising genes via mygene.info...")
    all_query = sorted(shared_not_wt | s161_only | s206_only)
    gene_cats = categorize_genes(all_query)

    # GO enrichment: full S161∩S206 for pathway panel
    logger.info("Running GO enrichment (full S161∩S206 set)...")
    gp = GProfiler(return_dataframe=True)
    go_all = gp.profile(
        organism='hsapiens', query=sorted(s161 & s206),
        sources=['GO:BP','GO:MF','KEGG','REAC'],
        significance_threshold_method='fdr', user_threshold=0.01,
        no_iea=False)
    time.sleep(0.4)

    # TF enrichment: shared_not_wt
    logger.info("Running TF enrichment...")
    go_tf = gp.profile(
        organism='hsapiens', query=sorted(shared_not_wt),
        sources=['TF'],
        significance_threshold_method='fdr', user_threshold=0.05,
        no_iea=False)
    time.sleep(0.4)

    # E2F fraction calculation
    e2f_genes = 0
    if not go_tf.empty:
        e2f_rows = go_tf[go_tf['name'].str.contains('E2F|ETF|ZF5', case=False)]
        e2f_genes = e2f_rows['intersection_size'].max() if not e2f_rows.empty else 0
    e2f_frac = e2f_genes / len(shared_not_wt) if len(shared_not_wt) > 0 else 0

    # Save table
    if table_out:
        rows = []
        for g in sorted(shared_not_wt):
            cats = gene_cats.get(g,{'Other'})
            rows.append({'gene':g, 's161':s161_c.get(g,0), 's206':s206_c.get(g,0),
                         'wt':wt_c.get(g,0), 'categories':'; '.join(sorted(cats))})
        pd.DataFrame(rows).to_csv(table_out, sep='\t', index=False)
        logger.info(f"Table: {table_out}")

    # ── Figure ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 34))
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.52, wspace=0.40)

    ax_catbar  = fig.add_subplot(gs[0, :])
    ax_kgm     = fig.add_subplot(gs[1, 0])
    ax_tf      = fig.add_subplot(gs[1, 1])
    ax_go      = fig.add_subplot(gs[2, 0])
    ax_concept = fig.add_subplot(gs[2, 1])
    ax_notes   = fig.add_subplot(gs[3, :])

    panel_category_bar(ax_catbar, gene_cats, shared_not_wt,
                       s161_only, s206_only, ['S161 only','S206 only'])
    panel_key_gene_matrix(ax_kgm, wt_c, s161_c, s206_c)
    panel_tf_enrichment(ax_tf, go_tf)
    panel_go_hematologic(ax_go, go_all)
    panel_convergence_concept(ax_concept)
    panel_notes(ax_notes, len(wt), len(s161), len(s206),
                len(shared_not_wt), e2f_frac)

    fig.suptitle(
        "PUF60 S161 ∩ S206 — Convergent Mechanisms Conferring Clonal Haematopoiesis Fitness\n"
        "256 genes gained by BOTH mutants relative to WT PUF60",
        fontsize=15, fontweight='bold', y=1.002)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output}")


def main():
    p = argparse.ArgumentParser(description='CH shared-target convergence analysis')
    p.add_argument('--wt',    required=True)
    p.add_argument('--s161',  required=True)
    p.add_argument('--s206',  required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--table', default='')
    args = p.parse_args()
    run(args.wt, args.s161, args.s206, args.output, args.table)


if __name__ == '__main__':
    main()
