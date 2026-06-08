#!/usr/bin/env python3
"""
Generate HTML Summary Report for HyperTRIBE Analysis
=====================================================

Reads all pipeline outputs and produces a self-contained HTML report
with alignment QC, editing site statistics, gene target tables, and
links to plot files.

Usage:
    python generate_report.py \\
        --editing-sites   results/annotated_editing_sites.bed \\
        --alignment-stats results/alignment_stats.txt \\
        --fastp-jsons     qc/sample1_fastp.json qc/sample2_fastp.json \\
        --gene-list       results/target_genes.txt \\
        --gene-ranks      results/target_genes_by_editcount.txt \\
        --output          results/analysis_report.html
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

_BASE_COLS = [
    'chr', 'start', 'end', 'gene', 'edit_freq', 'strand',
    'control_cov', 'control_A', 'control_G',
    'treatment_cov', 'treatment_A', 'treatment_G',
    'fold_change', 'p_value', 'n_replicates',
    'gene_id', 'gene_type',
]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_bed(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep='\t', comment='#', header=None)
    n = df.shape[1]
    df.columns = (_BASE_COLS + [f'extra_{i}' for i in range(n - len(_BASE_COLS))])[:n]
    df['edit_freq'] = pd.to_numeric(df['edit_freq'], errors='coerce')
    df['fold_change'] = pd.to_numeric(df['fold_change'], errors='coerce')
    df['p_value'] = pd.to_numeric(df['p_value'], errors='coerce')
    return df


def _load_alignment_stats(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep='\t')
    except Exception as e:
        logger.warning(f"Could not parse alignment stats: {e}")
        return pd.DataFrame()


def _load_fastp_json(path: str) -> dict:
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return {}


def _load_gene_ranks(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep='\t', header=None, names=['gene', 'n_sites'])
        return df.sort_values('n_sites', ascending=False).head(50)
    except Exception:
        return pd.DataFrame(columns=['gene', 'n_sites'])


# ─────────────────────────────────────────────────────────────────────────────
# HTML building blocks
# ─────────────────────────────────────────────────────────────────────────────

_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       margin: 0; padding: 0; background: #f5f6fa; color: #2d3436; }
.container { max-width: 1100px; margin: 0 auto; padding: 20px; }
h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
h2 { color: #2c3e50; margin-top: 40px; padding-bottom: 6px;
     border-bottom: 1px solid #dfe6e9; }
h3 { color: #636e72; margin-top: 24px; }
.card { background: white; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,.07);
        padding: 20px 24px; margin-bottom: 24px; }
.stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
             gap: 16px; margin: 16px 0; }
.stat-box { background: #f0f4f8; border-radius: 6px; padding: 14px 18px;
            text-align: center; }
.stat-box .value { font-size: 2em; font-weight: 700; color: #2980b9; }
.stat-box .label { font-size: 0.85em; color: #636e72; margin-top: 4px; }
table { width: 100%; border-collapse: collapse; font-size: 0.9em; }
th { background: #2c3e50; color: white; padding: 10px 12px; text-align: left; }
tr:nth-child(even) { background: #f8f9fa; }
td { padding: 8px 12px; border-bottom: 1px solid #eee; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 10px;
         font-size: 0.8em; font-weight: 600; }
.badge-blue  { background: #d6eaf8; color: #2980b9; }
.badge-green { background: #d5f5e3; color: #27ae60; }
.badge-red   { background: #fde8e8; color: #c0392b; }
.plot-link { display: inline-block; margin: 6px 10px 6px 0;
             padding: 8px 16px; background: #3498db; color: white;
             border-radius: 5px; text-decoration: none; font-size: 0.9em; }
.plot-link:hover { background: #2980b9; }
.warn { background: #fff3cd; border-left: 4px solid #ffc107;
        padding: 12px 16px; border-radius: 0 6px 6px 0; margin: 10px 0; }
.info { background: #d6eaf8; border-left: 4px solid #3498db;
        padding: 12px 16px; border-radius: 0 6px 6px 0; margin: 10px 0; }
footer { text-align: center; color: #aaa; font-size: 0.8em; margin-top: 40px;
         padding: 20px; }
"""

_JS = """
function filterTable(inputId, tableId) {
    var q = document.getElementById(inputId).value.toLowerCase();
    var rows = document.querySelectorAll('#' + tableId + ' tbody tr');
    rows.forEach(function(r) {
        r.style.display = r.textContent.toLowerCase().includes(q) ? '' : 'none';
    });
}
"""


def _stat_box(value: str, label: str) -> str:
    return (
        f'<div class="stat-box">'
        f'<div class="value">{value}</div>'
        f'<div class="label">{label}</div>'
        f'</div>'
    )


def _df_to_html_table(df: pd.DataFrame, table_id: str = '') -> str:
    id_attr = f' id="{table_id}"' if table_id else ''
    ths = ''.join(f'<th>{c}</th>' for c in df.columns)
    rows = ''
    for _, row in df.iterrows():
        cells = ''.join(f'<td>{v}</td>' for v in row)
        rows += f'<tr>{cells}</tr>'
    return (
        f'<div style="overflow-x:auto"><table{id_attr}>'
        f'<thead><tr>{ths}</tr></thead>'
        f'<tbody>{rows}</tbody>'
        f'</table></div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Report sections
# ─────────────────────────────────────────────────────────────────────────────

def _section_summary(sites: pd.DataFrame) -> str:
    n_sites = len(sites)
    n_genes = sites['gene'].replace('.', np.nan).dropna().nunique()
    med_freq = sites['edit_freq'].median()
    mean_freq = sites['edit_freq'].mean()
    pct_sig = (sites['p_value'] < 0.05).mean() * 100 if 'p_value' in sites.columns else float('nan')

    stats_html = (
        f'<div class="stat-grid">'
        f'{_stat_box(f"{n_sites:,}", "Total Editing Sites")}'
        f'{_stat_box(f"{n_genes:,}", "Target Genes")}'
        f'{_stat_box(f"{med_freq:.1f}%", "Median Edit Freq")}'
        f'{_stat_box(f"{mean_freq:.1f}%", "Mean Edit Freq")}'
        f'{_stat_box(f"{pct_sig:.1f}%", "Sites p < 0.05")}'
        f'</div>'
    )

    # Chromosome distribution (top 5)
    chr_counts = sites['chr'].value_counts().head(5)
    chr_html = ', '.join(
        f'<span class="badge badge-blue">{c}: {n:,}</span>'
        for c, n in chr_counts.items()
    )

    return (
        f'<div class="card">'
        f'<h2>Analysis Summary</h2>'
        f'{stats_html}'
        f'<p><strong>Top chromosomes:</strong> {chr_html}</p>'
        f'</div>'
    )


def _section_alignment(aln_df: pd.DataFrame) -> str:
    if aln_df.empty:
        return (
            '<div class="card"><h2>Alignment Statistics</h2>'
            '<p class="warn">alignment_stats.txt not found or could not be parsed.</p>'
            '</div>'
        )

    # Highlight mapping rate column if present
    display = aln_df.copy()
    for col in display.columns:
        if 'pct' in col.lower() or '%' in col.lower():
            display[col] = display[col].apply(
                lambda v: f'<span class="badge {"badge-green" if str(v).strip("%") and float(str(v).strip("%")) >= 70 else "badge-red"}">{v}</span>'
                if pd.notna(v) else v
            )

    return (
        f'<div class="card"><h2>Alignment Statistics</h2>'
        f'{_df_to_html_table(display)}'
        f'<p class="info">Mapping rate ≥ 70% is expected for well-prepared RNA-seq libraries.</p>'
        f'</div>'
    )


def _section_fastp(fastp_data: list[tuple[str, dict]]) -> str:
    if not fastp_data:
        return ''

    rows = []
    for sample_name, data in fastp_data:
        try:
            summary = data.get('summary', {})
            before = summary.get('before_filtering', {})
            after = summary.get('after_filtering', {})
            q30_before = before.get('q30_rate', 0) * 100
            q30_after = after.get('q30_rate', 0) * 100
            dup_rate = data.get('duplication', {}).get('rate', 0) * 100
            total_reads = before.get('total_reads', 'N/A')
            passed = after.get('total_reads', 'N/A')
            rows.append({
                'Sample': sample_name,
                'Total Reads': f'{total_reads:,}' if isinstance(total_reads, int) else total_reads,
                'Passed QC': f'{passed:,}' if isinstance(passed, int) else passed,
                'Q30 Before': f'{q30_before:.1f}%',
                'Q30 After': f'{q30_after:.1f}%',
                'Duplication': f'{dup_rate:.1f}%',
            })
        except Exception:
            rows.append({'Sample': sample_name, 'Note': 'Could not parse fastp JSON'})

    df = pd.DataFrame(rows)
    return (
        f'<div class="card"><h2>Read Quality Control (fastp)</h2>'
        f'{_df_to_html_table(df)}'
        f'</div>'
    )


def _section_top_genes(gene_ranks: pd.DataFrame) -> str:
    if gene_ranks.empty:
        return ''

    display = gene_ranks.copy()
    display.columns = ['Gene', 'Editing Sites']
    display['Editing Sites'] = display['Editing Sites'].apply(lambda v: f'{int(v):,}')

    search_html = (
        '<input type="text" placeholder="Search genes..." '
        'oninput="filterTable(\'gene-search\', \'gene-table\')" '
        'style="margin-bottom:10px; padding:6px 10px; border:1px solid #ccc; '
        'border-radius:4px; width:250px" id="gene-search">'
    )

    return (
        f'<div class="card"><h2>Top Target Genes</h2>'
        f'{search_html}'
        f'{_df_to_html_table(display, table_id="gene-table")}'
        f'</div>'
    )


def _section_biotype(sites: pd.DataFrame) -> str:
    if 'gene_type' not in sites.columns:
        return ''

    counts = sites['gene_type'].value_counts().head(15)
    total = counts.sum()
    rows = []
    for biotype, count in counts.items():
        rows.append({
            'Biotype': biotype,
            'Sites': f'{int(count):,}',
            'Percentage': f'{count / total * 100:.1f}%',
        })

    return (
        f'<div class="card"><h2>Editing Sites by Gene Biotype</h2>'
        f'{_df_to_html_table(pd.DataFrame(rows))}'
        f'</div>'
    )


def _section_plots(plot_dir: str) -> str:
    plot_files = {
        'Editing Frequency Distribution': 'plots/editing_frequency_distribution.pdf',
        'Chromosome Distribution': 'plots/chromosome_distribution.pdf',
        'Gene Biotype Distribution': 'plots/gene_biotype_distribution.pdf',
    }

    links = ''
    for label, rel_path in plot_files.items():
        links += (
            f'<a class="plot-link" href="{rel_path}" target="_blank">'
            f'&#128202; {label}</a>'
        )

    return (
        f'<div class="card"><h2>Plots</h2>'
        f'<p>Click to open publication-quality PDF figures:</p>'
        f'{links}'
        f'<p style="margin-top:16px; color:#888; font-size:0.85em">'
        f'Plots are in the <code>results/plots/</code> directory relative to this report.</p>'
        f'</div>'
    )


def _section_parameters(sites: pd.DataFrame) -> str:
    n = len(sites)
    if n == 0:
        return ''

    edit_freqs = sites['edit_freq'].dropna()
    p_values = sites['p_value'].dropna()

    rows = [
        {'Parameter': 'Total annotated editing sites', 'Value': f'{n:,}'},
        {'Parameter': 'Editing frequency range',
         'Value': f'{edit_freqs.min():.2f}% – {edit_freqs.max():.2f}%'},
        {'Parameter': 'Median editing frequency',
         'Value': f'{edit_freqs.median():.2f}%'},
        {'Parameter': 'Sites with p-value < 0.05',
         'Value': f'{(p_values < 0.05).sum():,} ({(p_values < 0.05).mean() * 100:.1f}%)'},
        {'Parameter': 'Sites with p-value < 0.01',
         'Value': f'{(p_values < 0.01).sum():,} ({(p_values < 0.01).mean() * 100:.1f}%)'},
        {'Parameter': 'Chromosomes with sites',
         'Value': str(sites['chr'].nunique())},
        {'Parameter': 'Intergenic sites',
         'Value': f"{(sites['gene'] == '.').sum():,}"},
    ]

    return (
        f'<div class="card"><h2>Dataset Statistics</h2>'
        f'{_df_to_html_table(pd.DataFrame(rows))}'
        f'</div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main report assembly
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(
    editing_sites: str,
    alignment_stats: str,
    fastp_jsons: list,
    gene_list: str,
    gene_ranks: str,
    output: str,
) -> None:
    logger.info("Loading pipeline outputs...")

    sites = _load_bed(editing_sites)
    logger.info(f"  Editing sites: {len(sites):,}")

    aln_df = _load_alignment_stats(alignment_stats)

    fastp_data = []
    for json_path in (fastp_jsons or []):
        name = Path(json_path).stem.replace('_fastp', '')
        fastp_data.append((name, _load_fastp_json(json_path)))

    gene_ranks_df = _load_gene_ranks(gene_ranks)

    n_genes_list = 0
    try:
        with open(gene_list) as fh:
            n_genes_list = sum(1 for line in fh if line.strip())
    except Exception:
        pass

    logger.info("Generating HTML report...")

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    body = (
        f'<div class="container">'
        f'<h1>HyperTRIBE Analysis Report</h1>'
        f'<p style="color:#888">Generated: {timestamp} &nbsp;|&nbsp; '
        f'Target gene list: {n_genes_list:,} genes</p>'
        + _section_summary(sites)
        + _section_alignment(aln_df)
        + _section_fastp(fastp_data)
        + _section_parameters(sites)
        + _section_top_genes(gene_ranks_df)
        + _section_biotype(sites)
        + _section_plots(str(Path(output).parent))
        + f'<footer>HyperTRIBE Pipeline v2.0 &mdash; '
          f'<a href="https://doi.org/10.1261/rna.064691.117">Xu et al. 2018</a></footer>'
        f'</div>'
    )

    html = (
        '<!DOCTYPE html>\n<html lang="en">\n<head>\n'
        '<meta charset="UTF-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        '<title>HyperTRIBE Analysis Report</title>\n'
        f'<style>{_CSS}</style>\n'
        f'<script>{_JS}</script>\n'
        '</head>\n<body>\n'
        + body
        + '\n</body>\n</html>'
    )

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w', encoding='utf-8') as fh:
        fh.write(html)

    logger.info(f"Report written: {output}")
    logger.info(f"  File size: {Path(output).stat().st_size / 1024:.1f} KB")


def main():
    parser = argparse.ArgumentParser(
        description='Generate HTML summary report for HyperTRIBE analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--editing-sites', required=True,
                        help='Annotated editing sites BED file')
    parser.add_argument('--alignment-stats', required=True,
                        help='Alignment statistics TSV (from compile_alignment_stats)')
    parser.add_argument('--fastp-jsons', nargs='*', default=[],
                        help='fastp JSON QC files (one per sample)')
    parser.add_argument('--gene-list', required=True,
                        help='target_genes.txt (one gene per line)')
    parser.add_argument('--gene-ranks', required=True,
                        help='target_genes_by_editcount.txt (gene<TAB>count)')
    parser.add_argument('--output', required=True,
                        help='Output HTML report file')
    args = parser.parse_args()

    logger.info('=' * 60)
    logger.info('HyperTRIBE Report Generator')
    logger.info('=' * 60)

    generate_report(
        editing_sites=args.editing_sites,
        alignment_stats=args.alignment_stats,
        fastp_jsons=args.fastp_jsons,
        gene_list=args.gene_list,
        gene_ranks=args.gene_ranks,
        output=args.output,
    )

    logger.info('=' * 60)
    logger.info('Report generation complete!')
    logger.info('=' * 60)


if __name__ == '__main__':
    main()
