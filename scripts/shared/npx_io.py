"""
Shared I/O and configuration for pediatric long-COVID NPX analyses.

Place at: scripts/shared/npx_io.py

Group assignment and covariate definitions here are deliberately identical to
scripts/pediatric/cohort.py (used by the pediatric entry points
01_cohort_qc.py ... 05_pathways_figures.py). If the two ever diverge, the
AUC values will describe a different sample set than the regression tables in
the manuscript -- exactly the kind of inconsistency a reviewer finds.

Expected inputs
---------------
NPX matrix : TSV/CSV, first column = SampleID, remaining columns = protein
             assay names, values = NPX (log2 scale).
             e.g. data/raw/Inflammation_NPX_AgeM.txt

Metadata   : EITHER
             (a) data/interim/harmonized_metadata.tsv exported by
                 01_cohort_qc.py -- preferred, since it already
                 carries harmonized_age, group and sex_01; OR
             (b) data/raw/phenotype.txt, in which case group and sex_01 are
                 derived here. harmonized_age must already be present either
                 way: age derivation from DOB / Data Vial uses group-specific
                 date formats and is NOT duplicated in this module.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

SAMPLE_COL = "SampleID"

CASE_GROUP = "LC_Neuro"

REFERENCE_GROUPS = {
    "strict": ["LC_NonNeuro_strict"],
    "expanded": ["LC_NonNeuro_strict", "CACTUS"],
    "healthy": ["Healthy"],
    "cactus": ["CACTUS"],
}

# Comparators drawing on both recruitment centres; used only to warn.
CROSS_SITE_COMPARISONS = {"expanded", "healthy", "cactus"}

CORE_CANDIDATES = ["IL17C", "TNFRSF11B"]
PHENOTYPE_CANDIDATE = ["FGF21"]
CANDIDATE_PANEL = CORE_CANDIDATES + PHENOTYPE_CANDIDATE

# Must match PRIMARY_COV in scripts/pediatric/logistic_models.py.
COVARIATES = ["harmonized_age", "sex_01"]

# Total-abundance outlier threshold, matching the regression pipeline.
TOTAL_ABUNDANCE_Z = 3.0

# Olink assay names that are NOT valid HGNC gene symbols.
OLINK_TO_SYMBOL: dict[str, str] = {}

@dataclass
class Dataset:
    """NPX matrix and metadata, aligned on SampleID."""

    npx: pd.DataFrame          # samples x proteins
    meta: pd.DataFrame         # samples x [group, harmonized_age, sex_01, ...]
    npx_source: str            # "uncorrected" or "combat"
    proteins: list[str]

    def subset(self, groups: list[str]) -> "Dataset":
        keep = self.meta.index[self.meta["group"].isin(groups)]
        return Dataset(
            npx=self.npx.loc[keep],
            meta=self.meta.loc[keep],
            npx_source=self.npx_source,
            proteins=self.proteins,
        )


def _read_table(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=sep)


# --------------------------------------------------------------------------
# Group and covariate derivation
# --------------------------------------------------------------------------

def derive_group(meta: pd.DataFrame) -> pd.Series:
    """Assign analysis groups from SampleID prefix plus Group / NEURO.

    Logic mirrors assign_groups() in scripts/pediatric/cohort.py (verified
    identical on the current phenotype file; note that an LC subject with
    missing NEURO would be LC_NonNeuro_strict there but Unknown here):

        SampleID starts with HC      -> Healthy
        Group == "LC" and NEURO == 1 -> LC_Neuro
        Group == "LC" and NEURO == 0 -> LC_NonNeuro_strict
        SampleID starts with CACTUS  -> CACTUS
        otherwise                    -> Unknown

    NEURO alone cannot serve as the group variable: it is NaN for CACTUS and
    Healthy samples, so on its own it separates only the two long-COVID arms.
    """
    for col in ("Group", "NEURO"):
        if col not in meta.columns:
            raise ValueError(
                f"Metadata lacks '{col}', needed to derive analysis groups. "
                "Pass data/raw/phenotype.txt, or the exported "
                "harmonized_metadata.tsv which already contains 'group'."
            )

    neuro = pd.to_numeric(meta["NEURO"], errors="coerce")
    is_lc = meta["Group"].astype(str).str.strip() == "LC"
    sid = pd.Series(meta.index.astype(str), index=meta.index)

    group = pd.Series("Unknown", index=meta.index, dtype=object)
    group[sid.str.startswith("CACTUS")] = "CACTUS"
    group[sid.str.startswith("HC")] = "Healthy"
    group[is_lc & (neuro == 0)] = "LC_NonNeuro_strict"
    group[is_lc & (neuro == 1)] = "LC_Neuro"
    return group


def derive_sex01(meta: pd.DataFrame) -> pd.Series:
    """sex_01 = 1 for female, matching the regression pipeline."""
    if "Gender" not in meta.columns:
        raise ValueError("Metadata lacks 'Gender', needed to derive sex_01.")
    return (meta["Gender"].astype(str).str.strip().str.upper().str[0] == "F").astype(float)


def load_dataset(
    npx_path: str | Path,
    meta_path: str | Path,
    npx_source: str,
    sample_col: str = SAMPLE_COL,
    drop_abundance_outliers: bool = True,
) -> Dataset:
    """Load and align the NPX matrix and metadata.

    npx_source must be stated explicitly ("uncorrected" or "combat") because it
    determines whether the discrimination estimate is contaminated by the
    outcome (ComBat is fitted with the LC_Neuro indicator protected).

    drop_abundance_outliers reproduces the |total_abundance z| > 3 exclusion
    applied before regression in 02_protein_associations.py. Leave it on,
    or the AUC sample size will not match the manuscript tables.
    """
    if npx_source not in {"uncorrected", "combat"}:
        raise ValueError("npx_source must be 'uncorrected' or 'combat'")

    npx = _read_table(Path(npx_path))
    meta = _read_table(Path(meta_path))
    for frame in (npx, meta):
        frame[sample_col] = frame[sample_col].astype(str).str.strip()
    npx = npx.set_index(sample_col)
    meta = meta.set_index(sample_col)

    shared = npx.index.intersection(meta.index)
    dropped_npx = len(npx.index.difference(shared))
    dropped_meta = len(meta.index.difference(shared))
    if dropped_npx or dropped_meta:
        print(
            f"[npx_io] Aligned on {len(shared)} samples "
            f"({dropped_npx} NPX-only, {dropped_meta} metadata-only dropped).",
            file=sys.stderr,
        )
    npx, meta = npx.loc[shared].copy(), meta.loc[shared].copy()

    # -- derive what is not already present -------------------------------
    if "group" not in meta.columns:
        meta["group"] = derive_group(meta)
    if "sex_01" not in meta.columns:
        meta["sex_01"] = derive_sex01(meta)

    if "harmonized_age" not in meta.columns:
        raise ValueError(
            "Metadata lacks 'harmonized_age'. Age is derived from DOB and "
            "Data Vial using group-specific date formats in "
            "scripts/pediatric/cohort.py; that logic is deliberately not "
            "duplicated here. Export data/interim/harmonized_metadata.tsv "
            "with 01_cohort_qc.py and pass it as --meta. Do NOT substitute the "
            "raw 'Age' column: it is populated for LC subjects only, so every "
            "Healthy and CACTUS sample would be dropped by listwise deletion."
        )
    meta["harmonized_age"] = pd.to_numeric(meta["harmonized_age"], errors="coerce")

    n_unknown = int((meta["group"] == "Unknown").sum())
    if n_unknown:
        print(f"[npx_io] {n_unknown} sample(s) assigned group 'Unknown'.",
              file=sys.stderr)
    print("[npx_io] group counts: "
          + ", ".join(f"{g}={n}" for g, n in meta["group"].value_counts().items()),
          file=sys.stderr)

    numeric = npx.select_dtypes(include=[np.number])
    if numeric.shape[1] < npx.shape[1]:
        dropped = sorted(set(npx.columns) - set(numeric.columns))
        print(f"[npx_io] Dropped {len(dropped)} non-numeric columns: {dropped[:5]}",
              file=sys.stderr)

    # -- total-abundance outlier exclusion --------------------------------
    if drop_abundance_outliers:
        total = numeric.apply(np.nansum, axis=1)
        z = (total - total.mean()) / total.std(ddof=1)
        keep = z.abs() <= TOTAL_ABUNDANCE_Z
        if (~keep).any():
            excluded = list(numeric.index[~keep])
            print(f"[npx_io] Excluded {len(excluded)} total-abundance outlier(s) "
                  f"(|z| > {TOTAL_ABUNDANCE_Z}): {excluded}", file=sys.stderr)
            numeric, meta = numeric.loc[keep], meta.loc[keep]

    n_no_age = int(meta["harmonized_age"].isna().sum())
    if n_no_age:
        print(f"[npx_io] WARNING: {n_no_age} sample(s) have no harmonized_age; "
              "they will be dropped from any fitted model.", file=sys.stderr)

    return Dataset(
        npx=numeric,
        meta=meta,
        npx_source=npx_source,
        proteins=list(numeric.columns),
    )


def build_comparison(ds: Dataset, reference: str) -> tuple[Dataset, pd.Series]:
    """Return the case+reference subset and a 0/1 outcome (1 = LC_Neuro).

    Rows with a missing covariate are dropped here rather than inside the
    model, so the reported n is the n actually analysed.
    """
    if reference not in REFERENCE_GROUPS:
        raise ValueError(f"reference must be one of {sorted(REFERENCE_GROUPS)}")

    groups = [CASE_GROUP] + REFERENCE_GROUPS[reference]
    sub = ds.subset(groups)

    complete = sub.meta[COVARIATES].notna().all(axis=1)
    if (~complete).any():
        print(f"[npx_io] Dropped {int((~complete).sum())} sample(s) with missing "
              f"covariates from '{reference}'.", file=sys.stderr)
        sub = Dataset(npx=sub.npx.loc[complete], meta=sub.meta.loc[complete],
                      npx_source=sub.npx_source, proteins=sub.proteins)

    y = (sub.meta["group"] == CASE_GROUP).astype(int)
    if y.sum() == 0 or (1 - y).sum() == 0:
        raise ValueError(f"Comparison '{reference}' has an empty class.")

    print(f"[npx_io] {reference}: {int(y.sum())} {CASE_GROUP} vs "
          f"{int((1 - y).sum())} reference", file=sys.stderr)

    if reference in CROSS_SITE_COMPARISONS and ds.npx_source == "uncorrected":
        print(f"[npx_io] WARNING: '{reference}' spans both recruitment centres "
              "but uncorrected NPX was supplied; site effects are unadjusted.",
              file=sys.stderr)
    if reference not in CROSS_SITE_COMPARISONS and ds.npx_source == "combat":
        print(f"[npx_io] NOTE: '{reference}' is within-site and does not require "
              "ComBat; using corrected NPX adds leakage for no benefit.",
              file=sys.stderr)

    return sub, y


def map_to_gene_symbols(names: list[str], strict: bool = True) -> tuple[list[str], list[str]]:
    """Map Olink assay names to HGNC symbols.

    Returns (mapped, unmapped). Names already matching the symbol convention
    pass through unchanged.
    """
    mapped, unmapped = [], []
    for n in names:
        if n in OLINK_TO_SYMBOL:
            mapped.append(OLINK_TO_SYMBOL[n])
        elif n.replace("-", "").isalnum() and n == n.upper() and " " not in n:
            mapped.append(n)
        else:
            unmapped.append(n)
    if strict and unmapped:
        raise ValueError(
            "Unmapped assay names -- Enrichr would silently drop these, "
            f"biasing the background: {unmapped}"
        )
    return mapped, unmapped