"""
Entry point 5 of 5 — Pathway enrichment and manuscript / supplementary figures.

Run every stage, or pick with --stage:

  pathways               Over-representation analysis (gseapy.enrich) of each
                         protein list against the 87-protein panel background;
                         gene sets restricted to the panel; BH across all terms
                         of the three libraries per list. Libraries are
                         downloaded from Enrichr at run time (network needed).
                         -> pathway/enrichment_<list>.csv, enrichment_all_lists.csv
  fig_candidates         Four primary candidates, uncorrected NPX (LC arms)
  fig_heatmap            Sample-level heatmap, primary comparison
  fig_pathways           Top LC-Neuro vs Healthy pathways (bar plot)
  fig_venn               Venn of nominal hits across regression comparisons
  fig_combat_diagnostics ComBat PCA before/after, total abundance by group

Inputs are read from --results_dir (default results/pediatric): the regression
and ComBat tables written by 02/03, pathway/enrichment_healthy_vs_neuro.csv for
fig_pathways, and the protein lists used for the pathway analysis
(panel_87_proteins.txt, core4.txt, expanded11.txt, expanded11_no_cactus.txt,
healthy23.txt). Those lists are inputs: no script in this repository creates them.
Each equals the nominal (p < 0.05, evaluable) hits of one regression table:
core4 = primary raw LC strict model; expanded11 = ComBat expanded model;
expanded11_no_cactus = expanded11 ∩ core4 (identical to core4); healthy23 =
ComBat LC_Neuro vs Healthy model; panel_87_proteins = all 87 assays (background).
See README.md ("Pathway inputs") for the library names and acquisition record.

Existing outputs are never overwritten unless --overwrite is given.

Usage
-----
  python scripts/pediatric/05_pathways_figures.py --outdir results/pediatric/rerun
  python scripts/pediatric/05_pathways_figures.py --stage fig_heatmap fig_candidates --outdir /tmp/x
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "shared"))

import cohort  # noqa: E402
from npx_io import map_to_gene_symbols  # noqa: E402
from output_safety import check_outputs_writable  # noqa: E402
from figures import (  # noqa: E402
    make_candidate_boxplots, make_combat_diagnostic_plots, make_pathway_barplot,
    make_regression_venn, make_sample_heatmap,
)

DEFAULT_METADATA = cohort.ROOT / "data" / "interim" / "harmonized_metadata.tsv"

# (label, list file in results_dir) — the four lists behind the existing
# results/pediatric/pathway/enrichment_<label>.csv files.
PATHWAY_BACKGROUND = "panel_87_proteins.txt"
PATHWAY_LISTS = [
    ("core", "core4.txt"),
    ("expanded", "expanded11.txt"),
    ("expanded_no_cactus", "expanded11_no_cactus.txt"),
    ("healthy_vs_neuro", "healthy23.txt"),
]

STAGE_OUTPUTS = {
    "pathways": [f"pathway/enrichment_{label}.csv" for label, _ in PATHWAY_LISTS]
                + ["pathway/enrichment_all_lists.csv"],
    "fig_candidates": ["primary_top4_uncorrected_npx_boxplots.png",
                       "primary_top4_uncorrected_npx_boxplots.pdf",
                       "primary_top4_uncorrected_npx_boxplots_source_data.tsv"],
    "fig_heatmap": ["expression_heatmaps"],
    "fig_pathways": ["pathway/healthy_vs_neuro_pathways.png",
                     "pathway/healthy_vs_neuro_pathways.pdf"],
    "fig_venn": ["venn_primary_regression_nominal.png", "venn_primary_regression_nominal.pdf"],
    "fig_combat_diagnostics": ["combat_pca_before_after_pretty.png",
                               "combat_pca_before_after_pretty.pdf",
                               "combat_total_abundance_by_group_pretty.png",
                               "combat_total_abundance_by_group_pretty.pdf",
                               "combat_pca_missingness_qc.tsv"],
}


# ═════════════════════════════════════════════════════════════════════════════
# Stage: pathways — panel-restricted over-representation analysis
# ═════════════════════════════════════════════════════════════════════════════

DEFAULT_LIBRARIES = [
    "GO_Biological_Process_2023",
    "Reactome_2022",
    "KEGG_2021_Human",
]

MIN_QUERY_SIZE = 5  # lists below this size are run deliberately here (core, expanded_no_cactus)


def read_list(path: Path) -> list[str]:
    with open(path) as fh:
        return [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]


def load_libraries(names: list[str]) -> dict[str, dict[str, list[str]]]:
    """Fetch each Enrichr library as a term -> gene-list dict.

    Fetched as dicts (rather than passing library names straight to enrich())
    so that the same gene sets can be cached, inspected and re-used, and so the
    run is reproducible if the remote library is later updated.
    """
    import gseapy

    libs = {}
    for name in names:
        try:
            libs[name] = gseapy.get_library(name=name)
            print(f"[pathway] loaded {name}: {len(libs[name])} terms", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            print(f"[pathway] FAILED to load {name}: {exc}", file=sys.stderr)
    if not libs:
        raise SystemExit("No gene-set libraries could be loaded.")
    return libs


def run_enrichment(
    query: list[str],
    background: list[str],
    libraries: dict[str, dict[str, list[str]]],
    label: str,
) -> pd.DataFrame:
    import gseapy

    frames = []
    for lib_name, gene_sets in libraries.items():
        # Restrict every gene set to the assayed panel before testing. Without
        # this, term sizes are counted over the whole genome while the query is
        # counted over 87 proteins, and the hypergeometric test is malformed.
        restricted = {
            term: sorted(set(genes) & set(background))
            for term, genes in gene_sets.items()
        }
        restricted = {t: g for t, g in restricted.items() if len(g) >= 3}
        if not restricted:
            print(f"[pathway] {lib_name}: no terms with >=3 panel members; skipped",
                  file=sys.stderr)
            continue

        res = gseapy.enrich(
            gene_list=query,
            gene_sets=restricted,
            background=background,
            outdir=None,
            verbose=False,
        )
        df = res.results.copy()
        df["library"] = lib_name
        df["list"] = label
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    # Benjamini-Hochberg across all terms from all libraries for this list.
    # gseapy adjusts within a library; correcting once across the whole family
    # is the honest accounting when three libraries are reported together.
    pcol = "P-value" if "P-value" in out.columns else "pvalue"
    out = out.sort_values(pcol).reset_index(drop=True)
    m = len(out)
    ranks = pd.Series(range(1, m + 1))
    raw = out[pcol].to_numpy()
    bh = (raw * m / ranks.to_numpy())
    out["fdr_across_libraries"] = pd.Series(bh)[::-1].cummin()[::-1].clip(upper=1.0).values
    return out


def run_pathways(outdir: Path, list_dir: Path) -> None:
    """Each list is tested independently (its own BH family), so results per
    list do not depend on which other lists are run in the same call."""
    pathway_dir = outdir / "pathway"
    pathway_dir.mkdir(parents=True, exist_ok=True)

    background_raw = read_list(list_dir / PATHWAY_BACKGROUND)
    background, _ = map_to_gene_symbols(background_raw, strict=True)
    background = sorted(set(background))
    print(f"[pathway] background: {len(background)} symbols (from {len(background_raw)} assays)")
    libraries = load_libraries(DEFAULT_LIBRARIES)

    all_results = []
    for label, fname in PATHWAY_LISTS:
        query, _ = map_to_gene_symbols(read_list(list_dir / fname), strict=True)
        query = sorted(set(query))
        not_in_bg = [g for g in query if g not in background]
        if not_in_bg:
            raise SystemExit(f"[{label}] query proteins absent from the background: {not_in_bg}")
        if len(query) < MIN_QUERY_SIZE:
            print(f"[pathway] NOTE '{label}' has {len(query)} proteins (< {MIN_QUERY_SIZE}); "
                  "interpret descriptively.")
        print(f"[pathway] {label}: {len(query)} proteins")
        res = run_enrichment(query, background, libraries, label)
        if res.empty:
            print(f"[pathway] {label}: no testable terms.")
            continue
        res.to_csv(pathway_dir / f"enrichment_{label}.csv", index=False)
        all_results.append(res)
        n_sig = int((res["fdr_across_libraries"] < 0.05).sum())
        print(f"[pathway] {label}: {n_sig} terms at FDR < 0.05 (of {len(res)} tested)")
    if all_results:
        pd.concat(all_results, ignore_index=True).to_csv(
            pathway_dir / "enrichment_all_lists.csv", index=False)


# ═════════════════════════════════════════════════════════════════════════════
# Figures: functions live in figures.py (moved from the former plot_*.py scripts)
# ═════════════════════════════════════════════════════════════════════════════

# ═════════════════════════════════════════════════════════════════════════════
# Command line
# ═════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", nargs="+", choices=list(STAGE_OUTPUTS),
                        default=list(STAGE_OUTPUTS), help="Stages to run (default: all)")
    parser.add_argument("--outdir", type=Path, default=cohort.OUTDIR)
    parser.add_argument("--results_dir", type=Path, default=cohort.OUTDIR,
                        help="Where regression/ComBat/enrichment tables and protein lists are read")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir, results_dir = args.outdir.resolve(), args.results_dir.resolve()
    check_outputs_writable([outdir / n for s in args.stage for n in STAGE_OUTPUTS[s]],
                           overwrite=args.overwrite)
    outdir.mkdir(parents=True, exist_ok=True)

    stages = {
        "pathways": lambda: run_pathways(outdir, results_dir),
        "fig_candidates": lambda: make_candidate_boxplots(
            outdir / "primary_top4_uncorrected_npx_boxplots.png", args.dpi),
        "fig_heatmap": lambda: make_sample_heatmap(
            outdir / "expression_heatmaps", args.metadata.resolve(), args.dpi),
        "fig_pathways": lambda: make_pathway_barplot(
            results_dir / "pathway" / "enrichment_healthy_vs_neuro.csv",
            outdir / "pathway" / "healthy_vs_neuro_pathways.png"),
        "fig_venn": lambda: make_regression_venn(results_dir, outdir),
        "fig_combat_diagnostics": lambda: make_combat_diagnostic_plots(results_dir, outdir),
    }
    for stage in STAGE_OUTPUTS:
        if stage in args.stage:
            print(f"\n{'#' * 70}\n# STAGE: {stage}\n{'#' * 70}")
            # Each figure restores matplotlib settings afterwards (the heatmap
            # changes rcParams), so every figure renders as it did standalone.
            with plt.rc_context():
                stages[stage]()
    print(f"\nOutputs written to: {outdir}")


if __name__ == "__main__":
    main()
