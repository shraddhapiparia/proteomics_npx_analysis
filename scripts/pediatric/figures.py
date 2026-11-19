"""
Figure functions for the pediatric manuscript, used by 05_pathways_figures.py.

Each function is the body of one former plot_*.py script, moved unchanged
except that input/output paths are parameters:

  make_candidate_boxplots       four primary candidates, uncorrected NPX (LC arms)
                                (from plot_candidate_proteins.py)
  make_sample_heatmap           sample-level heatmap, primary comparison
                                (from plot_heatmaps.py)
  make_pathway_barplot          top LC-Neuro vs Healthy pathways
                                (from plot_pathway_enrichment.py)
  make_regression_venn          Venn of nominal hits across regression comparisons
                                (from plot_regression_venn.py)
  make_combat_diagnostic_plots  ComBat PCA before/after, total abundance by group
                                (from plot_combat_diagnostics.py)

Calculations, appearance and output filenames are those of the original scripts.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib_venn import venn3, venn3_circles  # noqa: E402


def make_candidate_boxplots(figure_png: Path, dpi: int = 600) -> None:
    """Figure: four primary candidates, uncorrected NPX, LC arms only (moved from plot_candidate_proteins.py).
    Writes <figure_png>, .pdf and _source_data.tsv."""
    ROOT = Path(__file__).resolve().parent.parent.parent
    SCRIPT_DIR = Path(__file__).resolve().parent
    DEFAULT_OUTDIR = ROOT / "results" / "pediatric"

    PROTEINS = ["CCL2", "IL17C", "TNFRSF11B", "FGF21"]
    GROUPS = ["LC_NonNeuro_strict", "LC_Neuro"]

    GROUP_LABELS = {
        "LC_NonNeuro_strict": "LC-NonNeuro",
        "LC_Neuro": "LC-Neuro",
    }

    GROUP_COLORS = {
        "LC_NonNeuro_strict": "#e08214",
        "LC_Neuro": "#7b3294",
    }

    PVALS = {
        "CCL2": 0.027,
        "IL17C": 0.050,
        "TNFRSF11B": 0.035,
        "FGF21": 0.010,
    }

    def load_primary_raw() -> pd.DataFrame:
        """Load raw NPX and retain the primary same-center LC comparison."""
        sys.path.insert(0, str(SCRIPT_DIR))

        from cohort import (
            load_data,
            assign_groups,
            compute_total_abundance,
        )

        npx, pheno, protein_cols = load_data(verbose=False)

        missing = [p for p in PROTEINS if p not in protein_cols]
        if missing:
            sys.exit(
                "ERROR: candidate protein(s) not found in panel: "
                + ", ".join(missing)
            )

        df = assign_groups(npx, pheno, verbose=False)
        df = compute_total_abundance(df, protein_cols, verbose=False)

        # Same aggregate-NPX outlier exclusion used in the analysis pipeline
        ta_z = (
            (df["total_abundance"] - df["total_abundance"].mean())
            / df["total_abundance"].std(ddof=1)
        )
        df = df.loc[ta_z.abs() <= 3].copy()

        # Primary comparison only
        df = df[df["group"].isin(GROUPS)].copy()

        keep_cols = ["SampleID", "group"] + PROTEINS
        return df[keep_cols].copy()


    def main() -> None:
        ap = argparse.ArgumentParser()
        ap.add_argument(
            "--out",
            default=None,
            help="Output PNG path. A PDF with the same stem is also written.",
        )
        ap.add_argument("--dpi", type=int, default=600)
        args = argparse.Namespace(out=str(figure_png), dpi=dpi)

        df = load_primary_raw()

        print("\nPrimary-group sample counts:")
        print(df["group"].value_counts().reindex(GROUPS).to_string())

        fig, axes = plt.subplots(2, 2, figsize=(8.2, 7.0))
        axes = axes.flatten()

        rng = np.random.default_rng(42)

        for ax, protein in zip(axes, PROTEINS):
            data = []
            labels = []
            colors = []

            for group in GROUPS:
                vals = (
                    df.loc[df["group"] == group, protein]
                    .dropna()
                    .astype(float)
                    .values
                )

                data.append(vals)
                labels.append(f"{GROUP_LABELS[group]}\n(n = {len(vals)})")
                colors.append(GROUP_COLORS[group])

            bp = ax.boxplot(
                data,
                patch_artist=True,
                widths=0.55,
                showfliers=False,
                whis=1.5,
                medianprops={"color": "black", "linewidth": 1.5},
                boxprops={"linewidth": 1.1, "edgecolor": "black"},
                whiskerprops={"linewidth": 1.0, "color": "black"},
                capprops={"linewidth": 1.0, "color": "black"},
            )

            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.45)

            # Individual participant points
            for i, (vals, color) in enumerate(zip(data, colors), start=1):
                jitter = rng.uniform(-0.09, 0.09, size=len(vals))
                x = i + jitter

                ax.scatter(
                    x,
                    vals,
                    s=28,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=0.35,
                    alpha=0.85,
                    zorder=3,
                )

            ax.set_title(protein, fontsize=12, fontweight="bold", pad=8)
            ax.text(
                0.03, 0.97,
                f"p = {PVALS[protein]:.3f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8),
            )
            ax.set_xticks([1, 2], labels, fontsize=9)
            ax.set_ylabel("NPX", fontsize=10)

            ax.grid(axis="y", alpha=0.18, linewidth=0.7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="y", labelsize=9)
            ax.tick_params(axis="x", length=0)

            print(f"\n{protein}")
            for group, vals in zip(GROUPS, data):
                print(
                    f"  {GROUP_LABELS[group]:<14}"
                    f" n={len(vals):2d}"
                    f" mean={np.mean(vals):.3f}"
                    f" median={np.median(vals):.3f}"
                )

        fig.tight_layout(h_pad=2.0, w_pad=2.0)

        if args.out is None:
            out_png = DEFAULT_OUTDIR / "primary_top4_uncorrected_npx_boxplots.png"
        else:
            out_png = Path(args.out)

        out_png.parent.mkdir(parents=True, exist_ok=True)
        out_pdf = out_png.with_suffix(".pdf")

        fig.savefig(out_png, dpi=args.dpi, bbox_inches="tight")
        fig.savefig(out_pdf, bbox_inches="tight")
        plt.close(fig)

        source_out = out_png.with_name(out_png.stem + "_source_data.tsv")
        df.to_csv(source_out, sep="\t", index=False)

        print("\nSaved:")
        print(f"  {out_png}")
        print(f"  {out_pdf}")
        print(f"  {source_out}")


    main()


def make_sample_heatmap(outdir: Path, metadata_file: Path, dpi: int = 600) -> None:
    """Figure: sample-level heatmap, primary comparison, uncorrected NPX (moved from plot_heatmaps.py).
    Writes fig_sample_heatmap_primary_* into <outdir>."""
    # -----------------------------------------------------------------------------
    # arguments
    # -----------------------------------------------------------------------------

    ROOT = Path(__file__).resolve().parent.parent.parent

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--npx",
        default=ROOT / "data" / "raw" / "Inflammation_NPX_AgeM.txt",
        help="Olink NPX file, long or wide format.",
    )

    parser.add_argument(
        "--meta",
        default=ROOT / "data" / "interim" / "harmonized_metadata.tsv",
        help="Harmonized metadata file.",
    )

    parser.add_argument(
        "--outdir",
        default=ROOT / "results" / "pediatric" / "expression_heatmaps",
    )

    parser.add_argument(
        "--proteins",
        nargs="+",
        default=["CCL2", "IL17C", "TNFRSF11B", "FGF21"],
    )

    parser.add_argument(
        "--exclude",
        nargs="*",
        default=["CACTUS154_T0"],
    )

    parser.add_argument("--id-col", default="SampleID")
    parser.add_argument("--group-col", default="group")
    parser.add_argument("--neuro", default="LC_Neuro")
    parser.add_argument("--nonneuro", default="LC_NonNeuro_strict")
    parser.add_argument("--dpi", type=int, default=600)

    args = parser.parse_args(["--outdir", str(outdir), "--meta", str(metadata_file),
                              "--dpi", str(dpi)])

    OUTDIR = Path(args.outdir)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    PROTEINS = args.proteins


    # -----------------------------------------------------------------------------
    # style
    # -----------------------------------------------------------------------------

    for candidate_font in ("Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"):
        if any(candidate_font == f.name for f in font_manager.fontManager.ttflist):
            FONT = candidate_font
            break
    else:
        FONT = "DejaVu Sans"

    plt.rcParams.update(
        {
            "font.family": FONT,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": args.dpi,
            "figure.dpi": 150,
        }
    )

    INK = "#1f1f1f"
    GRID = "#ffffff"

    DIVERGING = LinearSegmentedColormap.from_list(
        "bluered",
        [
            "#2166ac",
            "#67a9cf",
            "#d1e5f0",
            "#f0efec",
            "#fddbc7",
            "#ef8a62",
            "#b2182b",
        ],
    )
    DIVERGING.set_bad("#d9d9d9")


    # -----------------------------------------------------------------------------
    # input helpers
    # -----------------------------------------------------------------------------

    def read_table(path):
        return pd.read_csv(path, sep=None, engine="python")


    def load_npx(path):
        """
    Return a samples x proteins NPX matrix.

    Supports:
    1. Olink long format with SampleID / Assay / NPX
    2. already-wide matrix with sample IDs in the first column
    """
        df = read_table(path)
        cols = {c.lower(): c for c in df.columns}

        if "assay" in cols and "npx" in cols:
            sid = cols.get("sampleid", df.columns[0])

            # If the supplied Olink file contains explicit technical QC warnings,
            # omit those individual assay values.
            if "qc_warning" in cols:
                bad = (
                    df[cols["qc_warning"]]
                    .astype(str)
                    .str.upper()
                    .isin(["WARNING", "WARN", "FAIL"])
                )
                if bad.any():
                    print(f"Dropping {int(bad.sum())} assay values with QC warning.")
                df = df.loc[~bad]

            wide = df.pivot_table(
                index=sid,
                columns=cols["assay"],
                values=cols["npx"],
                aggfunc="mean",
            )

        else:
            wide = df.set_index(df.columns[0])

        wide.index = wide.index.astype(str).str.strip()
        wide.columns.name = None

        return wide.apply(pd.to_numeric, errors="coerce")


    # -----------------------------------------------------------------------------
    # load and subset
    # -----------------------------------------------------------------------------

    npx = load_npx(args.npx)

    meta = read_table(args.meta)
    meta[args.id_col] = meta[args.id_col].astype(str).str.strip()
    meta = meta.set_index(args.id_col)

    # Keep only the two same-center Long COVID groups.
    keep_groups = [args.nonneuro, args.neuro]
    meta = meta[meta[args.group_col].isin(keep_groups)].copy()

    # Remove pre-specified QC exclusions, if present.
    to_exclude = [s for s in args.exclude if s in meta.index]
    if to_exclude:
        print("Excluding:", ", ".join(to_exclude))
        meta = meta.drop(index=to_exclude)

    # Keep only samples present in both files.
    common = meta.index.intersection(npx.index)

    missing_npx = meta.index.difference(npx.index)
    if len(missing_npx):
        print(
            f"WARNING: {len(missing_npx)} metadata samples have no NPX data: "
            f"{list(missing_npx)[:5]}"
        )

    meta = meta.loc[common].copy()
    npx = npx.loc[common].copy()

    # Confirm requested proteins exist.
    missing_proteins = [p for p in PROTEINS if p not in npx.columns]
    if missing_proteins:
        raise SystemExit(
            "Requested protein(s) not found in NPX file: "
            + ", ".join(missing_proteins)
        )

    print("\nPrimary-group sample counts:")
    print(meta[args.group_col].value_counts().reindex(keep_groups).to_string())


    # -----------------------------------------------------------------------------
    # fixed participant order
    # -----------------------------------------------------------------------------
    #
    # No clustering.
    # Participants are sorted alphabetically/numerically by SampleID within group.
    # This avoids creating visually exaggerated blocks of similar expression.
    #

    col_order = []

    for group in keep_groups:
        ids = meta.index[meta[args.group_col] == group]
        col_order.extend(sorted(ids.astype(str)))

    meta = meta.loc[col_order]
    vals = npx.loc[col_order, PROTEINS].copy()


    # -----------------------------------------------------------------------------
    # z-score within protein
    # -----------------------------------------------------------------------------
    #
    # For each protein:
    #     z = (participant NPX - protein mean) / protein SD
    #
    # The calculation uses only the participants shown in this primary figure.
    #

    means = vals.mean(axis=0)
    sds = vals.std(axis=0, ddof=1)

    if (sds == 0).any():
        bad = list(sds.index[sds == 0])
        raise SystemExit(
            "Cannot z-score protein(s) with zero variance: " + ", ".join(bad)
        )

    Z = (vals - means) / sds
    Zp = Z.T  # proteins x participants


    # -----------------------------------------------------------------------------
    # source data
    # -----------------------------------------------------------------------------

    source = vals.copy()
    source.insert(0, "group", meta[args.group_col])
    source.insert(0, "SampleID", source.index)

    for p in PROTEINS:
        source[f"{p}_z"] = Z[p]

    source.to_csv(
        OUTDIR / "fig_sample_heatmap_primary_source_data.tsv",
        sep="\t",
        index=False,
    )


    # -----------------------------------------------------------------------------
    # plot
    # -----------------------------------------------------------------------------

    nrow = len(PROTEINS)
    ncol = len(col_order)

    # Width scales gently with participant count.
    fig_w = max(7.0, 1.8 + ncol * 0.19)
    fig_h = 2.8 + 0.30 * nrow

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Keep a symmetric and interpretable scale.
    # Values outside ±2.5 SD saturate at the end colors.
    lim = 2.5

    im = ax.imshow(
        Zp.values.astype(float),
        cmap=DIVERGING,
        norm=TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim),
        aspect="auto",
        interpolation="nearest",
    )

    # White boundaries between individual cells.
    ax.set_xticks(np.arange(ncol + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(nrow + 1) - 0.5, minor=True)
    ax.grid(which="minor", color=GRID, linewidth=0.4)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Protein labels.
    ax.set_yticks(np.arange(nrow))
    ax.set_yticklabels(PROTEINS, fontsize=11)

    # Individual participant IDs are intentionally omitted.
    ax.set_xticks([])
    ax.tick_params(axis="both", length=0)

    for spine in ax.spines.values():
        spine.set_visible(False)


    # -----------------------------------------------------------------------------
    # group divider + group labels
    # -----------------------------------------------------------------------------

    n_nonneuro = int((meta[args.group_col] == args.nonneuro).sum())
    n_neuro = int((meta[args.group_col] == args.neuro).sum())

    divider = n_nonneuro - 0.5
    ax.axvline(divider, color=INK, linewidth=1.2)

    # Labels below heatmap.
    y_text = nrow - 0.5 + 0.30

    ax.text(
        (n_nonneuro - 1) / 2,
        y_text,
        f"LC-NonNeuro (n = {n_nonneuro})",
        ha="center",
        va="top",
        fontsize=10,
    )

    ax.text(
        n_nonneuro + (n_neuro - 1) / 2,
        y_text,
        f"LC-Neuro (n = {n_neuro})",
        ha="center",
        va="top",
        fontsize=10,
    )

    ax.set_title(
        "Candidate protein expression in the primary LC-Neuro comparison",
        loc="left",
        fontsize=11,
        pad=10,
    )


    # -----------------------------------------------------------------------------
    # colorbar
    # -----------------------------------------------------------------------------

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.025)
    cbar.set_label("NPX z-score within protein", fontsize=9)
    cbar.ax.tick_params(labelsize=8, length=2)


    # Add bottom room for group labels.
    plt.subplots_adjust(bottom=0.20)

    png = OUTDIR / "fig_sample_heatmap_primary_uncorrected.png"
    pdf = OUTDIR / "fig_sample_heatmap_primary_uncorrected.pdf"

    fig.savefig(png, bbox_inches="tight", dpi=args.dpi)
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"\nWrote:\n  {png}\n  {pdf}")
    print(
        f"  {OUTDIR / 'fig_sample_heatmap_primary_source_data.tsv'}"
    )


def make_pathway_barplot(enrichment_csv: Path, out_png: Path) -> None:
    """Figure: top LC-Neuro vs Healthy pathways (moved from plot_pathway_enrichment.py)."""
    #!/usr/bin/env python3




    INPUT = Path(enrichment_csv)

    OUTPUT = Path(out_png)

    TOP_N = 12


    def overlap_count(x):
        """
    Converts strings such as '4/8' into 4.
    """
        if pd.isna(x):
            return np.nan

        m = re.match(r"(\d+)\s*/\s*(\d+)", str(x))
        if m:
            return int(m.group(1))

        return np.nan


    def clean_term(term):
        """
    Remove GO IDs / Reactome IDs where possible to make labels cleaner.
    """
        term = str(term)

        # Remove trailing GO IDs such as "(GO:1904427)"
        term = re.sub(r"\s*\(GO:\d+\)\s*$", "", term)

        # Replace underscores if present
        term = term.replace("_", " ")

        return term


    df = pd.read_csv(INPUT)

    # Identify expected columns
    p_col = "P-value"
    fdr_col = "fdr_across_libraries"
    or_col = "Odds Ratio"
    overlap_col = "Overlap"

    # Keep nominally significant pathways only
    plot_df = df[df[p_col] < 0.05].copy()

    if plot_df.empty:
        raise SystemExit("No pathways with nominal p < 0.05.")

    # Number of query proteins contributing to each pathway
    plot_df["overlap_n"] = plot_df[overlap_col].apply(overlap_count)

    # Plotting metric
    plot_df["minus_log10_p"] = -np.log10(plot_df[p_col])

    # Clean pathway names
    plot_df["term_clean"] = plot_df["Term"].apply(clean_term)

    # Keep top pathways by nominal p-value
    plot_df = (
        plot_df
        .sort_values(p_col)
        .head(TOP_N)
        .sort_values(or_col)
    )

    # Dynamic figure height
    fig_height = max(5.5, 0.48 * len(plot_df))

    fig, ax = plt.subplots(figsize=(8.5, fig_height))

    scatter = ax.scatter(
        plot_df[or_col],
        np.arange(len(plot_df)),
        s=plot_df["overlap_n"] * 55,
        c=plot_df["minus_log10_p"],
        cmap="viridis",
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_yticks(np.arange(len(plot_df)))
    ax.set_yticklabels(plot_df["term_clean"], fontsize=10)

    ax.set_xlabel("Enrichment odds ratio", fontsize=12)
    ax.set_ylabel("")
    ax.set_title(
        "Healthy controls vs LC-Neuro: nominal pathway enrichment",
        fontsize=13,
    )

    ax.tick_params(axis="x", labelsize=10)

    # Color bar
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label(r"$-\log_{10}$(nominal p-value)", fontsize=10)

    # Size legend
    sizes = sorted(
        plot_df["overlap_n"].dropna().astype(int).unique()
    )

    # Use a few representative overlap counts
    if len(sizes) > 3:
        sizes = [
            min(sizes),
            int(np.median(sizes)),
            max(sizes),
        ]

    handles = [
        ax.scatter(
            [],
            [],
            s=n * 55,
            facecolor="white",
            edgecolor="black",
            linewidth=0.5,
            label=f"{n} proteins",
        )
        for n in sizes
    ]

    ax.legend(
        handles=handles,
        title="Overlap",
        frameon=False,
        loc="lower right",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        OUTPUT,
        dpi=600,
        bbox_inches="tight",
    )

    plt.savefig(
        OUTPUT.with_suffix(".pdf"),
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {OUTPUT}")
    print(f"Saved: {OUTPUT.with_suffix('.pdf')}")


def make_regression_venn(results_dir: Path, outdir: Path) -> None:
    """Figure: Venn of nominal (p < 0.05) hits: strict (raw), Healthy and expanded (ComBat).
    Moved from plot_regression_venn.py; regression tables read from <results_dir>."""
    #!/usr/bin/env python3




    # =============================================================================
    # Paths
    # =============================================================================

    ROOT = Path(__file__).resolve().parent.parent.parent
    RESULTS = Path(results_dir)

    # Same-source comparison: uncorrected NPX
    STRICT_FILE = (
        RESULTS
        / "regression_LC_NonNeuro_strict_vs_LC_Neuro_age_sex_adjusted.tsv"
    )

    # Cross-source comparison: ComBat-adjusted NPX
    HEALTHY_FILE = (
        RESULTS
        / "regression_combat_Healthy_vs_LC_Neuro_age_sex_adjusted.tsv"
    )

    # Expanded LC-NonNeuro = strict LC-NonNeuro + CACTUS
    # Use the ComBat-adjusted analysis
    EXPANDED_FILE = (
        RESULTS
        / "regression_combat_LC_NonNeuro_expanded_vs_LC_Neuro_age_sex_adjusted.tsv"
    )

    OUT_PNG = Path(outdir) / "venn_primary_regression_nominal.png"
    OUT_PDF = Path(outdir) / "venn_primary_regression_nominal.pdf"


    # =============================================================================
    # Settings
    # =============================================================================

    P_THRESHOLD = 0.05


    # =============================================================================
    # Load nominal hits
    # =============================================================================

    def load_nominal_hits(path: Path):
        """
    Return:
      hits       = set of nominally associated proteins
      directions = protein -> ↑ or ↓ according to regression beta

    Positive beta = higher in LC-Neuro.
    Negative beta = lower in LC-Neuro.
    """

        if not path.exists():
            raise FileNotFoundError(f"\nRegression file not found:\n{path}\n")

        df = pd.read_csv(path, sep="\t")

        required = {"protein", "p_value", "evaluable", "beta"}
        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"{path.name} missing required columns: {sorted(missing)}"
            )

        if df["evaluable"].dtype == object:
            evaluable = (
                df["evaluable"]
                .astype(str)
                .str.lower()
                .isin(["true", "1"])
            )
        else:
            evaluable = df["evaluable"].astype(bool)

        sig = df.loc[
            evaluable
            & df["p_value"].notna()
            & (df["p_value"] < P_THRESHOLD)
            & df["beta"].notna()
        ].copy()

        hits = set(sig["protein"].astype(str))

        directions = {
            row["protein"]: "↑" if row["beta"] > 0 else "↓"
            for _, row in sig.iterrows()
        }

        return hits, directions


    # =============================================================================
    # Main
    # =============================================================================

    def main():

        # -------------------------------------------------------------------------
        # Load the three sets
        # -------------------------------------------------------------------------

        strict, strict_dir = load_nominal_hits(STRICT_FILE)
        healthy, healthy_dir = load_nominal_hits(HEALTHY_FILE)
        expanded, expanded_dir = load_nominal_hits(EXPANDED_FILE)

        all_proteins = strict | healthy | expanded

        direction_map = {}

        for protein in all_proteins:

            observed = []

            if protein in strict_dir:
                observed.append(strict_dir[protein])

            if protein in healthy_dir:
                observed.append(healthy_dir[protein])

            if protein in expanded_dir:
                observed.append(expanded_dir[protein])

            unique_directions = set(observed)

            if len(unique_directions) > 1:
                raise ValueError(
                    f"Inconsistent direction across comparisons for {protein}: "
                    f"{observed}"
                )

            direction_map[protein] = observed[0]

        print("\n" + "=" * 65)
        print("NOMINALLY ASSOCIATED PROTEINS (p < 0.05)")
        print("=" * 65)

        print(f"\nLC-NonNeuro vs LC-Neuro: n={len(strict)}")
        print(sorted(strict))

        print(f"\nHealthy vs LC-Neuro [ComBat]: n={len(healthy)}")
        print(sorted(healthy))

        print(f"\nExpanded LC-NonNeuro vs LC-Neuro [ComBat]: n={len(expanded)}")
        print(sorted(expanded))

        # -------------------------------------------------------------------------
        # Exact Venn regions
        # -------------------------------------------------------------------------

        strict_only = strict - healthy - expanded
        healthy_only = healthy - strict - expanded
        expanded_only = expanded - strict - healthy

        strict_healthy = (strict & healthy) - expanded
        strict_expanded = (strict & expanded) - healthy
        healthy_expanded = (healthy & expanded) - strict

        all_three = strict & healthy & expanded

        print("\n" + "=" * 65)
        print("VENN REGIONS")
        print("=" * 65)

        print(f"\nStrict only ({len(strict_only)}):")
        print(sorted(strict_only))

        print(f"\nHealthy only ({len(healthy_only)}):")
        print(sorted(healthy_only))

        print(f"\nExpanded only ({len(expanded_only)}):")
        print(sorted(expanded_only))

        print(f"\nStrict + Healthy only ({len(strict_healthy)}):")
        print(sorted(strict_healthy))

        print(f"\nStrict + Expanded only ({len(strict_expanded)}):")
        print(sorted(strict_expanded))

        print(f"\nHealthy + Expanded only ({len(healthy_expanded)}):")
        print(sorted(healthy_expanded))

        print(f"\nAll three ({len(all_three)}):")
        print(sorted(all_three))

        # -------------------------------------------------------------------------
        # Create Venn
        # -------------------------------------------------------------------------

        fig, ax = plt.subplots(figsize=(9.2, 7.5))

        venn = venn3(
            subsets=(strict, healthy, expanded),
            set_labels=(
                f"LC-NonNeuro vs LC-Neuro\n(n={len(strict)})",
                f"Healthy vs LC-Neuro\n(n={len(healthy)})",
                f"Expanded LC-NonNeuro vs LC-Neuro\n(n={len(expanded)})",
            ),
            ax=ax,
        )

        # -------------------------------------------------------------------------
        # Muted manuscript colors
        # -------------------------------------------------------------------------
        #
        # matplotlib-venn represents each overlap as its own patch, so explicitly
        # setting each region gives a more controlled appearance than relying on
        # default transparent primary colors.
        # -------------------------------------------------------------------------

        region_colors = {
            "100": "#C9D5E8",   # strict only: muted blue
            "010": "#C6DFC5",   # healthy only: muted green
            "001": "#E1C9DE",   # expanded only: muted mauve

            "110": "#C6D8D3",   # strict + healthy
            "101": "#D4C9DF",   # strict + expanded
            "011": "#D5D7CC",   # healthy + expanded

            "111": "#C9C2CF",   # all three
        }

        set_label_colors = {
            0: "#5B6FA8",  # strict: darker blue
            1: "#5E8A5E",  # healthy: darker green
            2: "#9A6E91",  # expanded: darker mauve
        }

        for region_id, color in region_colors.items():

            patch = venn.get_patch_by_id(region_id)

            if patch is not None:
                patch.set_facecolor(color)
                patch.set_alpha(1.0)
                patch.set_edgecolor("none")

        # Circle boundaries
        circles = venn3_circles(
            subsets=(strict, healthy, expanded),
            ax=ax,
            linewidth=0.9,
            color="#444444",
        )

        # -------------------------------------------------------------------------
        # Region → protein mapping
        # -------------------------------------------------------------------------

        region_map = {
            "100": strict_only,
            "010": healthy_only,
            "001": expanded_only,
            "110": strict_healthy,
            "101": strict_expanded,
            "011": healthy_expanded,
            "111": all_three,
        }

        # -------------------------------------------------------------------------
        # Replace numbers with protein names
        #
        # Empty regions are blank — NO "0".
        # -------------------------------------------------------------------------

        for region_id, proteins in region_map.items():

            label = venn.get_label_by_id(region_id)

            if label is None:
                continue

            if len(proteins) == 0:
                label.set_text("")
                continue

            protein_text = [
                f"{protein} {direction_map[protein]}"
                for protein in sorted(proteins)
            ]

            label.set_text("\n".join(protein_text))
            label.set_fontsize(10)
            label.set_fontweight("normal")
            label.set_color("#111111")
            label.set_horizontalalignment("center")
            label.set_verticalalignment("center")

            # Three-way replicated overlap
            if region_id == "111":
                label.set_fontsize(10.5)
                label.set_fontweight("bold")

        # -------------------------------------------------------------------------
        # Manual protein-label positioning
        #
        # These positions are tuned for your current set sizes:
        # strict = 4
        # healthy = 23
        # expanded = 11
        #
        # If the number of nominal hits changes substantially after rerunning the
        # regressions, these may need minor adjustment.
        # -------------------------------------------------------------------------

        region_positions = {
            "010": (0.27, 0.10),     # Healthy-only proteins
            "001": (-0.52, -0.08),   # Expanded-only proteins
            "101": (-0.345, 0.00),    # FGF21
            "111": (-0.12, 0.02),     # CCL2, IL17C, TNFRSF11B
        }

        for region_id, position in region_positions.items():

            label = venn.get_label_by_id(region_id)

            if label is not None and label.get_text().strip():
                label.set_position(position)

        # -------------------------------------------------------------------------
        # Set labels
        # -------------------------------------------------------------------------

        set_positions = {
            0: (-0.41, 0.17),   # strict
            1: (0.41, 0.48),    # healthy
            2: (-0.33, -0.46),  # expanded
        }

        for idx, label in enumerate(venn.set_labels):

            if label is None:
                continue

            label.set_position(set_positions[idx])
            label.set_fontsize(10.5)
            label.set_fontweight("semibold")
            label.set_color(set_label_colors[idx])
            label.set_horizontalalignment("center")

        # -------------------------------------------------------------------------
        # Final manuscript formatting
        # -------------------------------------------------------------------------

        ax.set_axis_off()

        # Leave room for comparison names outside circles
        ax.set_xlim(-0.70, 0.70)
        ax.set_ylim(-0.57, 0.60)

        # No title inside figure; use manuscript figure legend instead.
        fig.tight_layout()

        # PNG for viewing / slides
        fig.savefig(
            OUT_PNG,
            dpi=600,
            bbox_inches="tight",
            facecolor="white",
        )

        # Vector version for manuscript submission
        fig.savefig(
            OUT_PDF,
            bbox_inches="tight",
            facecolor="white",
        )

        plt.close(fig)

        print("\n" + "=" * 65)
        print("OUTPUT")
        print("=" * 65)
        print(f"PNG: {OUT_PNG}")
        print(f"PDF: {OUT_PDF}")


    main()


def make_combat_diagnostic_plots(results_dir: Path, outdir: Path) -> None:
    """Figure: ComBat PCA before/after and total abundance by group (moved from plot_combat_diagnostics.py).
    ComBat matrix/metadata read from <results_dir>."""
    #!/usr/bin/env python3




    # ── Paths ──────────────────────────────────────────────────────────────────────
    ROOT    = Path(__file__).resolve().parent.parent.parent
    RAW_NPX = ROOT / "data" / "raw" / "Inflammation_NPX_AgeM.txt"
    ADJ_NPX = Path(results_dir) / "combat_adjusted_npx.tsv"
    META    = Path(results_dir) / "combat_metadata.tsv"
    OUT_DIR = Path(outdir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Display label mapping ──────────────────────────────────────────────────────
    GROUP_LABELS = {
        "Healthy":            "Healthy",
        "LC_Neuro":           "LC Neuro",
        "LC_NonNeuro_strict": "LC Non-Neuro",
        "CACTUS":             "CACTUS / MIS-C",
    }
    GROUP_ORDER = ["Healthy", "LC_Neuro", "LC_NonNeuro_strict", "CACTUS"]

    GROUP_COLORS = {
        "Healthy":            "#2CA02C",   # green
        "LC_Neuro":           "#D62728",   # red
        "LC_NonNeuro_strict": "#FF7F0E",   # orange
        "CACTUS":             "#1F77B4",   # blue
    }

    # Batch → marker shape
    BATCH_MARKERS = {
        "LC_source":        "o",
        "HC_CACTUS_source": "^",
    }
    BATCH_DISPLAY = {
        "LC_source":        "LC Neuro + LC Non-Neuro",
        "HC_CACTUS_source": "Healthy + CACTUS / MIS-C",
    }


    # ── PCA helper (numpy SVD; no sklearn required) ────────────────────────────────
    def run_pca(X: np.ndarray, scale: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Return (scores[:, :2], pct_variance[:2]) after centering, optionally scaling."""
        Xc = X - X.mean(axis=0)

        if scale:
            sd = Xc.std(axis=0, ddof=1)
            sd[sd == 0] = 1.0
            Xc = Xc / sd

        U, s, _ = np.linalg.svd(Xc, full_matrices=False)
        scores = U * s
        pct_var = 100 * s**2 / (s**2).sum()
        return scores[:, :2], pct_var[:2]


    # ── Load data ──────────────────────────────────────────────────────────────────
    print(f"[INFO] Raw NPX : {RAW_NPX}")
    print(f"[INFO] Adj NPX : {ADJ_NPX}")
    print(f"[INFO] Metadata: {META}")

    for f in [RAW_NPX, ADJ_NPX, META]:
        if not f.exists():
            raise FileNotFoundError(f"Required input not found: {f}")

    raw_npx = pd.read_csv(RAW_NPX, sep="\t")
    adj_npx = pd.read_csv(ADJ_NPX, sep="\t")
    meta    = pd.read_csv(META,    sep="\t")

    raw_npx["SampleID"] = raw_npx["SampleID"].str.strip()
    adj_npx["SampleID"] = adj_npx["SampleID"].str.strip()
    meta["SampleID"]    = meta["SampleID"].str.strip()

    protein_cols = [c for c in adj_npx.columns if c != "SampleID"]
    print(f"[INFO] Proteins : {len(protein_cols)}")
    print(f"[INFO] Samples  : raw={len(raw_npx)}  adj={len(adj_npx)}  meta={len(meta)}")

    raw_ids = set(raw_npx["SampleID"])
    meta_ids = set(meta["SampleID"])
    adj_ids = set(adj_npx["SampleID"])

    print("[CHECK] Raw samples not in metadata:")
    print(sorted(raw_ids - meta_ids))

    print("[CHECK] Metadata samples not in raw:")
    print(sorted(meta_ids - raw_ids))

    print("[CHECK] Metadata samples not in adjusted NPX:")
    print(sorted(meta_ids - adj_ids))

    # Restrict raw NPX to samples present in metadata (respects ComBat exclusions)
    raw_npx = raw_npx[raw_npx["SampleID"].isin(meta["SampleID"])].copy()

    # Align adj NPX to metadata order
    adj_npx = adj_npx.set_index("SampleID").loc[meta["SampleID"]].reset_index()
    raw_npx = raw_npx.set_index("SampleID").reindex(meta["SampleID"]).reset_index()

    # ── Identify complete samples (no NAs in raw data) ─────────────────────────────
    # raw_mat = raw_npx[protein_cols].values.astype(float)
    # complete_mask = ~np.any(np.isnan(raw_mat), axis=1)
    # n_complete = int(complete_mask.sum())
    # n_total    = len(meta)
    # print(f"[INFO] Complete samples for PCA: {n_complete} / {n_total}")

    # ── Identify missingness, then impute for PCA ──────────────────────────────────
    raw_mat = raw_npx[protein_cols].values.astype(float)
    adj_mat = adj_npx[protein_cols].values.astype(float)

    n_total = len(meta)
    nan_counts = np.sum(np.isnan(raw_mat), axis=1)
    n_with_missing = int((nan_counts > 0).sum())
    n_missing_values = int(np.isnan(raw_mat).sum())

    IMPUTE_METHOD = "median"

    def impute_by_protein(X: np.ndarray, method: str = "median") -> np.ndarray:
        """Impute missing values column-wise, protein by protein."""
        X_imp = X.copy()

        if method == "median":
            fill_vals = np.nanmedian(X_imp, axis=0)
        elif method == "mean":
            fill_vals = np.nanmean(X_imp, axis=0)
        else:
            raise ValueError("IMPUTE_METHOD must be 'median' or 'mean'")

        row_idx, col_idx = np.where(np.isnan(X_imp))
        X_imp[row_idx, col_idx] = fill_vals[col_idx]
        return X_imp

    raw_mat_imp = impute_by_protein(raw_mat, IMPUTE_METHOD)
    adj_mat_imp = impute_by_protein(adj_mat, IMPUTE_METHOD)

    print(
        f"[INFO] PCA samples after {IMPUTE_METHOD} imputation: "
        f"{n_total} / {n_total}"
    )
    print(
        f"[INFO] Imputed missing raw NPX values: "
        f"{n_missing_values} value(s) across {n_with_missing} sample(s)"
    )

    # ── Missingness QC table ───────────────────────────────────────────────────────
    # nan_counts = np.sum(np.isnan(raw_mat), axis=1)   # per sample

    qc_rows = []
    for i, sid in enumerate(meta["SampleID"]):
        n_miss = int(nan_counts[i])
        miss_proteins = [protein_cols[j] for j in range(len(protein_cols))
                         if np.isnan(raw_mat[i, j])]
        g = meta["group"].iloc[i]
        qc_rows.append({
            "SampleID":          sid,
            "group":             g,
            "display_group":     GROUP_LABELS.get(g, g),
            "site_source":       meta["site_source"].iloc[i],
            "n_missing_proteins": n_miss,
            "missing_proteins":  ";".join(miss_proteins) if miss_proteins else "",
        })

    qc_df = pd.DataFrame(qc_rows)
    qc_path = OUT_DIR / "combat_pca_missingness_qc.tsv"
    qc_df.to_csv(qc_path, sep="\t", index=False)
    print(f"[OUT] {qc_path.name}")

    # Concise missingness summary
    # n_excluded = n_total - n_complete
    # excluded_df = qc_df[~complete_mask]
    # print(f"\n[MISSINGNESS SUMMARY]")
    # print(f"  Samples excluded from PCA: {n_excluded} / {n_total}")
    # if n_excluded > 0:
    #     by_group = excluded_df.groupby("display_group").size()
    #     for dg, count in by_group.items():
    #         print(f"    {dg}: {count}")
    #     protein_freq = {}
    #     for proteins_str in excluded_df["missing_proteins"]:
    #         for p in proteins_str.split(";"):
    #             if p:
    #                 protein_freq[p] = protein_freq.get(p, 0) + 1
    #     print(f"  Missing protein frequencies (across excluded samples):")
    #     for p, freq in sorted(protein_freq.items(), key=lambda x: -x[1]):
    #         print(f"    {p}: {freq} sample(s)")
    # else:
    #     print("  No samples excluded — all have complete protein data.")

    # meta_pca = meta[complete_mask].reset_index(drop=True)

    # Concise missingness summary
    imputed_df = qc_df[qc_df["n_missing_proteins"] > 0]
    print(f"\n[MISSINGNESS SUMMARY]")
    print(f"  Samples excluded from PCA: 0 / {n_total}")
    print(
        f"  Samples imputed for PCA: {len(imputed_df)} / {n_total} "
        f"using protein-wise {IMPUTE_METHOD}"
    )

    if len(imputed_df) > 0:
        by_group = imputed_df.groupby("display_group").size()
        for dg, count in by_group.items():
            print(f"    {dg}: {count}")

        protein_freq = {}
        for proteins_str in imputed_df["missing_proteins"]:
            for p in proteins_str.split(";"):
                if p:
                    protein_freq[p] = protein_freq.get(p, 0) + 1

        print(f"  Imputed protein frequencies:")
        for p, freq in sorted(protein_freq.items(), key=lambda x: -x[1]):
            print(f"    {p}: {freq} sample(s)")
    else:
        print("  No imputation needed — all samples have complete protein data.")

    meta_pca = meta.reset_index(drop=True)


    # Per-group plotted counts for PCA legend
    pca_group_n = meta_pca["group"].value_counts()
    for g in GROUP_ORDER:
        print(f"  PCA {GROUP_LABELS[g]}: n={pca_group_n.get(g, 0)}")

    # raw_pca_mat = raw_mat[complete_mask]
    # adj_pca_mat = adj_npx[protein_cols].values.astype(float)[complete_mask]

    raw_pca_mat = raw_mat_imp
    adj_pca_mat = adj_mat_imp

    scores_bef, pct_bef = run_pca(raw_pca_mat, scale=True)
    scores_aft, pct_aft = run_pca(adj_pca_mat, scale=True)

    # ── Figure 1: PCA before / after ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(
        "PCA of Olink NPX — Before and After ComBat Batch Correction",
        fontsize=13, fontweight="bold", y=1.01,
    )

    panel_titles  = ["Before ComBat", "After ComBat"]
    scores_list   = [scores_bef, scores_aft]
    pct_list      = [pct_bef,    pct_aft]

    for ax, title, scores, pct in zip(axes, panel_titles, scores_list, pct_list):
        for g in GROUP_ORDER:
            mask = (meta_pca["group"] == g).values
            if not mask.any():
                continue
            batch = meta_pca.loc[mask, "site_source"].iloc[0]
            n_g   = mask.sum()
            label = f"{GROUP_LABELS[g]} (n={n_g})"
            ax.scatter(
                scores[mask, 0], scores[mask, 1],
                c      = GROUP_COLORS[g],
                marker = BATCH_MARKERS.get(batch, "o"),
                s      = 60,
                alpha  = 0.85,
                label  = label,
                edgecolors = "white",
                linewidths = 0.4,
                zorder = 3,
            )
        ax.set_xlabel(f"PC1 ({pct[0]:.1f}% variance)", fontsize=10)
        ax.set_ylabel(f"PC2 ({pct[1]:.1f}% variance)", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)

    # Shared legend on the first axis
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(
        handles, labels,
        title      = "Clinical group",
        title_fontsize = 8,
        fontsize   = 8,
        frameon    = False,
        loc        = "best",
    )

    # Batch shape legend on second axis
    shape_handles = [
        mpatches.Patch(color="none", label="Shape = sample source / batch"),
        plt.Line2D([0], [0], marker="o", color="grey", linestyle="none",
                   markersize=6, label=f"○  {BATCH_DISPLAY['LC_source']}"),
        plt.Line2D([0], [0], marker="^", color="grey", linestyle="none",
                   markersize=6, label=f"△  {BATCH_DISPLAY['HC_CACTUS_source']}"),
    ]
    axes[1].legend(
        handles    = shape_handles,
        fontsize   = 7.5,
        frameon    = False,
        loc        = "best",
        handlelength = 1.5,
    )

    fig.text(
        0.5, -0.03,
        "Batch/source: LC source = LC Neuro + LC Non-Neuro; "
        "HC/CACTUS source = Healthy + CACTUS / MIS-C.",
        ha="center", fontsize=8, color="grey",
    )

    fig.tight_layout()

    for suffix, fmt in [("pretty.pdf", "pdf"), ("pretty.png", "png")]:
        out_path = OUT_DIR / f"combat_pca_before_after_{suffix}"
        fig.savefig(out_path, format=fmt, bbox_inches="tight", dpi=150)
        print(f"[OUT] {out_path.name}")

    plt.close(fig)


    # ── Figure 2: Total abundance boxplot by group ─────────────────────────────────
    # total_abundance in metadata was computed pre-ComBat from original NPX rowSums
    total_group_n = meta["group"].value_counts()
    for g in GROUP_ORDER:
        print(f"  Abundance {GROUP_LABELS[g]}: n={total_group_n.get(g, 0)}")

    fig2, ax2 = plt.subplots(figsize=(7, 5))

    display_names = [GROUP_LABELS[g] for g in GROUP_ORDER]
    x_pos        = np.arange(len(GROUP_ORDER))

    box_data = []
    for g in GROUP_ORDER:
        vals = meta.loc[meta["group"] == g, "total_abundance"].dropna().values
        box_data.append(vals)

    bp = ax2.boxplot(
        box_data,
        positions  = x_pos,
        widths     = 0.45,
        patch_artist = True,
        medianprops  = dict(color="black", linewidth=1.5),
        whiskerprops = dict(linewidth=1),
        capprops     = dict(linewidth=1),
        flierprops   = dict(marker="", linestyle="none"),
        zorder       = 2,
    )

    for patch, g in zip(bp["boxes"], GROUP_ORDER):
        patch.set_facecolor(GROUP_COLORS[g])
        patch.set_alpha(0.55)

    rng = np.random.default_rng(42)
    for i, (g, vals) in enumerate(zip(GROUP_ORDER, box_data)):
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax2.scatter(
            x_pos[i] + jitter, vals,
            color      = GROUP_COLORS[g],
            s          = 28,
            alpha      = 0.8,
            edgecolors = "white",
            linewidths = 0.3,
            zorder     = 3,
        )

    tick_labels = [
        f"{GROUP_LABELS[g]}\n(n={total_group_n.get(g, 0)})"
        for g in GROUP_ORDER
    ]
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(tick_labels, fontsize=9)
    ax2.set_ylabel(f"Sum NPX across {len(protein_cols)} proteins", fontsize=10)
    ax2.set_title("Total protein abundance by group", fontsize=12, fontweight="bold")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.tick_params(axis="y", labelsize=8)

    fig2.tight_layout()

    for suffix, fmt in [("pretty.pdf", "pdf"), ("pretty.png", "png")]:
        out_path = OUT_DIR / f"combat_total_abundance_by_group_{suffix}"
        fig2.savefig(out_path, format=fmt, bbox_inches="tight", dpi=150)
        print(f"[OUT] {out_path.name}")

    plt.close(fig2)

    print("\n[DONE] Diagnostic plots written to", OUT_DIR)
