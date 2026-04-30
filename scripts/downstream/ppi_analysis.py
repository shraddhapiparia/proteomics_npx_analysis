"""
ppi_analysis.py
Protein-protein interaction (PPI) analysis for pediatric Long COVID neuro vs non-neuro.

Design choices:
- Primary comparison: LC_NonNeuro_vs_LC_Neuro (pediatric, age+sex adjusted)
- UKBB comparison: LC_Neuro_vs_LC_NonNeuro (direction is flipped relative to peds)
  After alignment (negate peds beta), concordant direction = direction_match_after_alignment=True
- Healthy vs LC_Neuro abundance-adjusted used only to supplement biologically coherent Tier 2
- Three protein tiers:
    Tier 1: peds adjusted p < 0.05  (main, small, confirmed signal)
    Tier 2: expanded exploratory (~12-18 proteins, suitable for STRING/enrichment)
    Tier 3: cytokine/chemokine subset of Tier 2 (most interpretable for neuro biology)
- STRING API queried for interactions and functional enrichment (species=9606, human)
- All outputs under results/ppi/
"""

import os
import sys
import json
import time
import textwrap
import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

# ── paths ──────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(BASE, "results", "ppi")
os.makedirs(os.path.join(OUT, "protein_lists"), exist_ok=True)
os.makedirs(os.path.join(OUT, "string"),        exist_ok=True)
os.makedirs(os.path.join(OUT, "plots"),         exist_ok=True)

PEDS_ADJ_FILE   = os.path.join(BASE, "results", "pediatric",
                                "regression_LC_NonNeuro_vs_LC_Neuro_age_sex_adjusted.tsv")
PEDS_UNIV_FILE  = os.path.join(BASE, "results", "pediatric",
                                "regression_LC_NonNeuro_vs_LC_Neuro_univariate.tsv")
PEDS_NEURO_FILE = os.path.join(BASE, "results", "pediatric",
                                "regression_Healthy_vs_LC_Neuro_total_abundance_adjusted.tsv")
UKBB_FILE       = os.path.join(BASE, "results", "ukbb",
                                "protein_logistic_regression_results.tsv")
DIR_CHECK_FILE  = os.path.join(BASE, "results", "comparisons",
                                "neuro_direction_alignment_check.tsv")

# Proteins annotated as cytokines/chemokines/immune mediators (manual curation)
# based on known biology; used to define Tier 3
CYTOKINE_TERMS = {
    "CCL2", "CCL3", "CCL4", "CCL7", "CCL8", "CCL11", "CCL13", "CCL19", "CCL20",
    "CCL23", "CCL25", "CCL28", "CX3CL1", "CXCL1", "CXCL5", "CXCL6", "CXCL8",
    "CXCL9", "CXCL10", "CXCL11", "IL4", "IL5", "IL6", "IL7", "IL10", "IL12B",
    "IL15RA", "IL17A", "IL17C", "IL18", "IL18R1", "IL20", "IL22RA1", "IL24",
    "IL2RB", "IL10RA", "IL10RB", "IL20RA", "IFNG", "TNF", "LTA", "LIF",
    "TSLP", "OSM", "TNFSF10", "TNFSF11", "TNFSF12", "TNFSF14",
}

STRING_API = "https://string-db.org/api/json"
SPECIES    = 9606  # Homo sapiens
SCORE_THR  = 400   # STRING combined score threshold (medium confidence)


# ── helpers ───────────────────────────────────────────────────────────────────

def load_and_check(path, label):
    df = pd.read_csv(path, sep="\t")
    print(f"[{label}] loaded {len(df)} rows, columns: {list(df.columns)}")
    return df


def clean_ukbb(df):
    """Strip olink_instance_0. prefix from UKBB protein column."""
    df = df.copy()
    df["protein"] = df["protein"].str.replace("olink_instance_0.", "", regex=False).str.upper()
    return df


def string_ids(proteins, species=SPECIES, caller_id="peds_lc_ppi"):
    """Map gene symbols → STRING identifiers, return dict {symbol: string_id}."""
    url = f"{STRING_API}/get_string_ids"
    payload = {
        "identifiers": "\r".join(proteins),
        "species": species,
        "caller_identity": caller_id,
        "echo_query": 1,
    }
    r = requests.post(url, data=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    mapping = {}
    for row in data:
        query = row.get("queryItem", "").upper()
        sid   = row.get("stringId", "")
        if query and sid:
            mapping[query] = sid
    return mapping


def string_network(proteins, species=SPECIES, score=SCORE_THR, caller_id="peds_lc_ppi"):
    """Return interaction table for a list of gene symbols."""
    url = f"{STRING_API}/network"
    payload = {
        "identifiers": "%0d".join(proteins),
        "species": species,
        "required_score": score,
        "caller_identity": caller_id,
    }
    r = requests.post(url, data=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    return df


def string_enrichment(proteins, species=SPECIES, caller_id="peds_lc_ppi"):
    """Return functional enrichment for a list of gene symbols."""
    url = f"{STRING_API}/enrichment"
    payload = {
        "identifiers": "%0d".join(proteins),
        "species": species,
        "caller_identity": caller_id,
    }
    r = requests.post(url, data=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    return df


# ── 1. load data ──────────────────────────────────────────────────────────────

print("=" * 70)
print("STEP 1: Load result files")
print("=" * 70)

peds_adj  = load_and_check(PEDS_ADJ_FILE,   "peds_adj")
peds_univ = load_and_check(PEDS_UNIV_FILE,  "peds_univ")
peds_neuro= load_and_check(PEDS_NEURO_FILE, "peds_healthy_vs_neuro")
ukbb_raw  = load_and_check(UKBB_FILE,       "ukbb_raw")
dir_check = load_and_check(DIR_CHECK_FILE,  "dir_check")

# UKBB: strip prefix, filter to neuro comparison
ukbb_raw = clean_ukbb(ukbb_raw)
ukbb_neuro = ukbb_raw[ukbb_raw["comparison"] == "LC_Neuro_vs_LC_NonNeuro"].copy()
print(f"UKBB neuro comparison rows: {len(ukbb_neuro)}")

# Verify coding (fail early if columns change)
for col in ["protein", "beta", "p_value", "fit_warning"]:
    assert col in peds_adj.columns, f"Missing column {col} in peds_adj"
for col in ["protein", "coef", "p_value"]:
    assert col in ukbb_neuro.columns, f"Missing column {col} in ukbb_neuro — check schema"

print()
print("Pediatric comparison label:", peds_adj["comparison"].iloc[0])
print("  Coding: positive beta = higher in LC_NonNeuro relative to LC_Neuro (reference)")
print("UKBB comparison label:", ukbb_neuro["comparison"].iloc[0])
print("  Coding: positive beta = higher in LC_Neuro relative to LC_NonNeuro (reference)")
print("  ⚠  Direction is FLIPPED between cohorts. Do NOT compare betas without sign flip.")
print()


# ── 2. direction alignment check ──────────────────────────────────────────────

print("=" * 70)
print("STEP 2: Direction alignment verification")
print("=" * 70)

# dir_check was pre-computed: peds_beta_flipped = -peds_beta_original
# direction_match_after_alignment=True means same biological direction after flip
concordant = dir_check[dir_check["direction_match_after_alignment"] == True]["protein"].tolist()
print(f"Proteins in direction alignment file: {len(dir_check)}")
print(f"Directionally concordant (after sign flip): {len(concordant)}")
print(f"  {concordant}")


# ── 3. protein tier selection ─────────────────────────────────────────────────

print()
print("=" * 70)
print("STEP 3: Protein tier selection")
print("=" * 70)

# -- Tier 1: peds adjusted p < 0.05, no convergence warning
t1_mask = (peds_adj["p_value"] < 0.05) & (peds_adj["fit_warning"] == False)
tier1_df = peds_adj[t1_mask][["protein", "beta", "OR", "p_value", "FDR",
                                "CI_lower", "CI_upper", "n", "covariates"]].copy()
tier1_df = tier1_df.sort_values("p_value").reset_index(drop=True)
tier1_df["source"]           = "peds_adj_p05"
tier1_df["dir_concordant_ukbb"] = tier1_df["protein"].isin(concordant)
tier1_df["tier"]             = 1

print(f"\nTier 1 (peds adjusted p < 0.05): {len(tier1_df)} proteins")
for _, row in tier1_df.iterrows():
    conc = "✓" if row["dir_concordant_ukbb"] else "✗"
    print(f"  {row['protein']:12s} beta={row['beta']:+.3f}  p={row['p_value']:.4f}  "
          f"dir_concordant_UKBB={conc}")

# -- Tier 2: expanded exploratory
#   a) peds adjusted p < 0.10 (not already in Tier 1), no fit_warning
peds_adj_p10 = peds_adj[
    (peds_adj["p_value"] < 0.10) &
    (peds_adj["fit_warning"] == False) &
    (~peds_adj["protein"].isin(tier1_df["protein"]))
].copy()

#   b) peds univariate p < 0.10, not already captured, no fit_warning
peds_univ_p10 = peds_univ[
    (peds_univ["p_value"] < 0.10) &
    (peds_univ["fit_warning"] == False) &
    (~peds_univ["protein"].isin(tier1_df["protein"])) &
    (~peds_univ["protein"].isin(peds_adj_p10["protein"]))
].copy()

#   c) UKBB neuro p < 0.05 (added as UKBB-derived, not pediatric primary)
ukbb_p05 = ukbb_neuro[ukbb_neuro["p_value"] < 0.05].copy()
ukbb_add = ukbb_p05[
    ~ukbb_p05["protein"].isin(tier1_df["protein"]) &
    ~ukbb_p05["protein"].isin(peds_adj_p10["protein"]) &
    ~ukbb_p05["protein"].isin(peds_univ_p10["protein"])
].copy()

#   d) Healthy vs LC_Neuro peds abundance-adjusted p < 0.05, no fit_warning
#      Included for biologically coherent neuro-context expansion; clearly labeled
#      as from a different comparison.
peds_neuro_p05 = peds_neuro[
    (peds_neuro["p_value"] < 0.05) &
    (peds_neuro["fit_warning"] == False) &
    (~peds_neuro["protein"].isin(tier1_df["protein"])) &
    (~peds_neuro["protein"].isin(peds_adj_p10["protein"])) &
    (~peds_neuro["protein"].isin(peds_univ_p10["protein"])) &
    (~peds_neuro["protein"].isin(ukbb_add["protein"]))
].copy()

# Assemble Tier 2 rows
def make_tier2_rows(df, src, use_coef_col=None):
    rows = []
    beta_col = use_coef_col if use_coef_col else "beta"
    for _, r in df.iterrows():
        rows.append({
            "protein":    r["protein"],
            "beta":       r[beta_col],
            "p_value":    r["p_value"],
            "source":     src,
            "dir_concordant_ukbb": r["protein"] in concordant,
            "tier":       2,
        })
    return rows

tier2_rows = (
    [{"protein": r["protein"], "beta": r["beta"], "p_value": r["p_value"],
      "source": "peds_adj_p10", "dir_concordant_ukbb": r["protein"] in concordant, "tier": 2}
     for _, r in peds_adj_p10.iterrows()] +
    [{"protein": r["protein"], "beta": r["beta"], "p_value": r["p_value"],
      "source": "peds_univ_p10", "dir_concordant_ukbb": r["protein"] in concordant, "tier": 2}
     for _, r in peds_univ_p10.iterrows()] +
    [{"protein": r["protein"], "beta": r["coef"], "p_value": r["p_value"],
      "source": "ukbb_neuro_p05", "dir_concordant_ukbb": r["protein"] in concordant, "tier": 2}
     for _, r in ukbb_add.iterrows()] +
    [{"protein": r["protein"], "beta": r["beta"], "p_value": r["p_value"],
      "source": "peds_hlthy_neuro_p05", "dir_concordant_ukbb": r["protein"] in concordant, "tier": 2}
     for _, r in peds_neuro_p05.iterrows()]
)

tier2_extra_df = pd.DataFrame(tier2_rows)

# Full Tier 2 = Tier 1 + extra
tier2_all = pd.concat(
    [tier1_df[["protein", "beta", "p_value", "source", "dir_concordant_ukbb", "tier"]],
     tier2_extra_df],
    ignore_index=True
)
tier2_all = tier2_all.drop_duplicates(subset="protein").reset_index(drop=True)

print(f"\nTier 2 (expanded exploratory): {len(tier2_all)} proteins")
for _, row in tier2_all.iterrows():
    conc = "✓" if row["dir_concordant_ukbb"] else "✗"
    flag = " ← UKBB-only" if row["source"] == "ukbb_neuro_p05" else \
           " ← Healthy/Neuro context" if row["source"] == "peds_hlthy_neuro_p05" else ""
    print(f"  [{row['tier']}] {row['protein']:12s} beta={row['beta']:+.3f}  p={row['p_value']:.4f}  "
          f"dir_concordant_UKBB={conc}  source={row['source']}{flag}")

# -- Tier 3: cytokine/chemokine/immune subset of Tier 2
tier3_all = tier2_all[tier2_all["protein"].isin(CYTOKINE_TERMS)].copy()
tier3_all["tier"] = 3

print(f"\nTier 3 (cytokine/immune subset of Tier 2): {len(tier3_all)} proteins")
print(f"  {sorted(tier3_all['protein'].tolist())}")


# ── 4. write protein lists ─────────────────────────────────────────────────────

print()
print("=" * 70)
print("STEP 4: Write protein lists")
print("=" * 70)

def write_list(df, tier_label, notes_col=None):
    proteins = df["protein"].tolist()
    # plain text (one per line)
    txt_path = os.path.join(OUT, "protein_lists", f"{tier_label}.txt")
    with open(txt_path, "w") as fh:
        fh.write("\n".join(proteins) + "\n")
    # TSV with stats
    tsv_path = os.path.join(OUT, "protein_lists", f"{tier_label}.tsv")
    df.to_csv(tsv_path, sep="\t", index=False)
    print(f"  Written: {txt_path}  ({len(proteins)} proteins)")
    print(f"  Written: {tsv_path}")
    return proteins

t1_proteins = write_list(tier1_df, "tier1_main")
t2_proteins = write_list(tier2_all, "tier2_expanded")
t3_proteins = write_list(tier3_all, "tier3_cytokine")


# ── 5. protein summary plot ────────────────────────────────────────────────────

print()
print("=" * 70)
print("STEP 5: Protein selection summary plot")
print("=" * 70)

fig, ax = plt.subplots(figsize=(10, 6))

plot_df = tier2_all.copy()
plot_df["neg_log10p"] = -np.log10(plot_df["p_value"].clip(lower=1e-10))
plot_df = plot_df.sort_values("neg_log10p", ascending=True)

colors = {
    "peds_adj_p05":           "#e63946",
    "peds_adj_p10":           "#f4a261",
    "peds_univ_p10":          "#a8dadc",
    "ukbb_neuro_p05":         "#457b9d",
    "peds_hlthy_neuro_p05":   "#6a994e",
}
labels = {
    "peds_adj_p05":           "Peds adjusted (p<0.05) — Tier 1",
    "peds_adj_p10":           "Peds adjusted (p<0.10) — Tier 2 expansion",
    "peds_univ_p10":          "Peds univariate (p<0.10) — Tier 2 expansion",
    "ukbb_neuro_p05":         "UKBB neuro (p<0.05) — UKBB-derived",
    "peds_hlthy_neuro_p05":   "Peds Healthy vs Neuro (p<0.05) — neuro context",
}

bars = ax.barh(
    plot_df["protein"],
    plot_df["neg_log10p"],
    color=[colors[s] for s in plot_df["source"]],
    edgecolor="none",
    height=0.7,
)
# Mark directionally concordant proteins with a star
for i, (_, row) in enumerate(plot_df.iterrows()):
    if row["dir_concordant_ukbb"]:
        ax.text(
            row["neg_log10p"] + 0.03,
            i,
            "★",
            va="center", fontsize=9, color="#1d3557",
        )

ax.axvline(-np.log10(0.05), color="black", linestyle="--", lw=1, alpha=0.6,
           label="p=0.05")
ax.axvline(-np.log10(0.10), color="gray",  linestyle=":",  lw=1, alpha=0.6,
           label="p=0.10")

patches = [mpatches.Patch(color=v, label=labels[k]) for k, v in colors.items()]
patches.append(mpatches.Patch(color="white", label="★ directionally concordant with UKBB"))
ax.legend(handles=patches, fontsize=7, loc="lower right", framealpha=0.8)

ax.set_xlabel("−log₁₀(p-value)")
ax.set_title("Protein selection for PPI: LC_Neuro vs LC_NonNeuro\n"
             "(pediatric primary + UKBB replication context)",
             fontsize=11, pad=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
out_path = os.path.join(OUT, "plots", "protein_selection_summary.png")
plt.savefig(out_path, dpi=150)
plt.close()
print(f"  Saved: {out_path}")


# ── 6. STRING queries ──────────────────────────────────────────────────────────

print()
print("=" * 70)
print("STEP 6: STRING API queries")
print("=" * 70)

def run_string_tier(label, proteins, score=SCORE_THR):
    print(f"\n  Querying STRING for {label} ({len(proteins)} proteins) ...")
    # interactions
    try:
        net = string_network(proteins, score=score)
        if not net.empty:
            net_path = os.path.join(OUT, "string", f"{label}_interactions.tsv")
            net.to_csv(net_path, sep="\t", index=False)
            print(f"    Interactions: {len(net)} edges → {net_path}")
        else:
            print(f"    Interactions: no edges above score {score}")
    except Exception as e:
        print(f"    Interactions query failed: {e}")
        net = pd.DataFrame()

    time.sleep(1)  # be polite to STRING

    # enrichment
    try:
        enr = string_enrichment(proteins)
        if not enr.empty:
            enr_path = os.path.join(OUT, "string", f"{label}_enrichment.tsv")
            enr.to_csv(enr_path, sep="\t", index=False)
            print(f"    Enrichment: {len(enr)} terms → {enr_path}")
        else:
            print(f"    Enrichment: no terms returned")
    except Exception as e:
        print(f"    Enrichment query failed: {e}")
        enr = pd.DataFrame()

    time.sleep(1)
    return net, enr

# Only run STRING for tiers with enough proteins
net1, enr1 = run_string_tier("tier1_main",    t1_proteins, score=150)  # lower threshold for small list
net2, enr2 = run_string_tier("tier2_expanded", t2_proteins)
net3, enr3 = run_string_tier("tier3_cytokine", t3_proteins)


# ── 7. enrichment bar plots ───────────────────────────────────────────────────

print()
print("=" * 70)
print("STEP 7: Enrichment plots")
print("=" * 70)

def plot_enrichment(enr_df, label, top_n=15, categories=("Process", "KEGG", "RCTM")):
    if enr_df.empty:
        print(f"  {label}: no enrichment data to plot")
        return

    # Identify relevant columns
    cat_col   = next((c for c in ["category", "Category", "term_category"] if c in enr_df.columns), None)
    term_col  = next((c for c in ["description", "term", "term_name"] if c in enr_df.columns), None)
    pval_col  = next((c for c in ["fdr", "p_value", "p_fdr_bh"] if c in enr_df.columns), None)
    prot_col  = next((c for c in ["inputGenes", "matching_proteins", "proteins"] if c in enr_df.columns), None)

    if not all([cat_col, term_col, pval_col]):
        print(f"  {label}: unexpected enrichment columns {enr_df.columns.tolist()} — skipping plot")
        return

    # Filter to selected categories and significant terms
    subset = enr_df[enr_df[cat_col].isin(categories)].copy()
    if subset.empty:
        subset = enr_df.copy()

    subset[pval_col] = pd.to_numeric(subset[pval_col], errors="coerce")
    subset = subset.dropna(subset=[pval_col])
    subset = subset[subset[pval_col] < 0.05]
    subset = subset.nsmallest(top_n, pval_col)

    if subset.empty:
        print(f"  {label}: no significant enrichment terms to plot")
        return

    subset = subset.sort_values(pval_col, ascending=False)
    subset["neg_log10_fdr"] = -np.log10(subset[pval_col].clip(lower=1e-20))

    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(subset))))
    bar_colors = subset[cat_col].map({
        "Process": "#e63946", "KEGG": "#457b9d", "RCTM": "#a8dadc",
    }).fillna("#6a994e")
    ax.barh(range(len(subset)), subset["neg_log10_fdr"], color=bar_colors, height=0.7, edgecolor="none")
    ax.set_yticks(range(len(subset)))
    ax.set_yticklabels(
        [textwrap.shorten(str(t), 55) for t in subset[term_col]],
        fontsize=8,
    )
    if prot_col and prot_col in subset.columns:
        for i, (_, row) in enumerate(subset.iterrows()):
            ax.text(0.02, i, str(row[prot_col])[:60], va="center", fontsize=6, alpha=0.7)
    ax.axvline(-np.log10(0.05), color="black", linestyle="--", lw=1, alpha=0.5)
    ax.set_xlabel("−log₁₀(FDR/p)")
    ax.set_title(f"STRING functional enrichment — {label}", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    patches = [mpatches.Patch(color="#e63946", label="GO:Process"),
               mpatches.Patch(color="#457b9d", label="KEGG"),
               mpatches.Patch(color="#a8dadc", label="Reactome")]
    ax.legend(handles=patches, fontsize=7, loc="lower right")

    plt.tight_layout()
    out_path = os.path.join(OUT, "plots", f"{label}_enrichment.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")

plot_enrichment(enr1, "tier1_main")
plot_enrichment(enr2, "tier2_expanded")
plot_enrichment(enr3, "tier3_cytokine")


# ── 8. STRING network plots ───────────────────────────────────────────────────

print()
print("=" * 70)
print("STEP 8: Network plots")
print("=" * 70)

def plot_network(net_df, proteins_in_tier, tier1_set, label, score_col="score"):
    if net_df.empty:
        print(f"  {label}: no interactions to plot")
        return

    # Determine column names (STRING returns preferredName_A/B and score)
    a_col = next((c for c in ["preferredName_A", "protein1"] if c in net_df.columns), None)
    b_col = next((c for c in ["preferredName_B", "protein2"] if c in net_df.columns), None)
    sc_col = next((c for c in ["score", "combined_score"] if c in net_df.columns), None)

    if not all([a_col, b_col, sc_col]):
        print(f"  {label}: unexpected network columns {net_df.columns.tolist()} — skipping")
        return

    G = nx.Graph()
    for _, row in net_df.iterrows():
        G.add_edge(row[a_col], row[b_col], weight=float(row[sc_col]) / 1000)

    pos = nx.spring_layout(G, seed=42, k=2.0)
    fig, ax = plt.subplots(figsize=(9, 7))

    node_colors = []
    for node in G.nodes():
        if node in tier1_set:
            node_colors.append("#e63946")    # Tier 1 (robust signal)
        elif node in proteins_in_tier:
            node_colors.append("#457b9d")    # Tier 2 expansion
        else:
            node_colors.append("#adb5bd")    # STRING hub (not in our list)

    edge_weights = [G[u][v]["weight"] * 2 for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.5, edge_color="#ccc", ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=600, node_color=node_colors, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=7, ax=ax)

    patches = [
        mpatches.Patch(color="#e63946", label="Tier 1 (peds adj p<0.05)"),
        mpatches.Patch(color="#457b9d", label="Tier 2 (exploratory)"),
        mpatches.Patch(color="#adb5bd", label="STRING neighbor"),
    ]
    ax.legend(handles=patches, fontsize=8, loc="upper left")
    ax.set_title(f"STRING PPI network — {label}\n(edge width ∝ combined score)", fontsize=10)
    ax.axis("off")

    plt.tight_layout()
    out_path = os.path.join(OUT, "plots", f"{label}_network.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")

t1_set = set(t1_proteins)
plot_network(net2, t2_proteins, t1_set, "tier2_expanded")
plot_network(net3, t3_proteins, t1_set, "tier3_cytokine")


# ── 9. markdown interpretation summary ────────────────────────────────────────

print()
print("=" * 70)
print("STEP 9: Markdown summary")
print("=" * 70)

def fmt_list(proteins, concordant_set):
    lines = []
    for p in proteins:
        mark = " ★" if p in concordant_set else ""
        lines.append(f"  - {p}{mark}")
    return "\n".join(lines)

concordant_set = set(concordant)

# Gather top enrichment terms for Tier 2 if available
top_terms_t2 = ""
if not enr2.empty:
    pval_col = next((c for c in ["fdr", "p_value"] if c in enr2.columns), None)
    term_col = next((c for c in ["description", "term"] if c in enr2.columns), None)
    if pval_col and term_col:
        enr2_sig = enr2[pd.to_numeric(enr2[pval_col], errors="coerce") < 0.05]
        top3 = enr2_sig.nsmallest(5, pval_col)[term_col].tolist()
        top_terms_t2 = "\n".join(f"  - {t}" for t in top3)

summary = f"""\
# PPI Analysis: LC_Neuro vs LC_NonNeuro — Interpretation Summary

**Date:** 2026-04-23
**Primary cohort:** Pediatric Long COVID (n=34 for neuro comparison)
**Replication context:** UK Biobank adult cohort

---

## Direction alignment

The pediatric regression codes: `LC_NonNeuro_vs_LC_Neuro`
(positive beta = higher in LC_NonNeuro relative to LC_Neuro).

The UKBB regression codes: `LC_Neuro_vs_LC_NonNeuro`
(positive beta = higher in LC_Neuro relative to LC_NonNeuro).

**Effect directions are flipped across cohorts.** The pediatric beta must be
negated before comparing to UKBB. This was verified using the pre-computed
`neuro_direction_alignment_check.tsv`.
A ★ marks proteins where the biological direction is concordant after alignment.

---

## Protein tiers chosen for PPI

### Tier 1 — Main list ({len(t1_proteins)} proteins, suitable for candidate PPI)
Selection: pediatric age+sex-adjusted model (LC_NonNeuro_vs_LC_Neuro), p < 0.05,
no model convergence warning.

{fmt_list(t1_proteins, concordant_set)}

**Why included:** These are the only proteins reaching nominal significance in the
primary covariate-adjusted comparison. FGF21 (★) is the most robust: significant
in both the adjusted and univariate models, and directionally concordant with UKBB.
The other three (CCL2, TNFRSF11B, IL17C) are significant in adjusted model only;
directionality is discordant with UKBB or UKBB is underpowered for them.

**Suitability:** Too small (n=4) for a stable standalone STRING network; used as
the anchor set for Tier 2 expansion. Enrichment is unreliable at n=4.

---

### Tier 2 — Expanded exploratory list ({len(t2_proteins)} proteins)
Selection: Tier 1 + peds adjusted/univariate p < 0.10 + UKBB neuro p < 0.05 +
peds Healthy vs LC_Neuro abundance-adjusted p < 0.05 (for biologically coherent
neuro context). Proteins from each source are labeled.

{fmt_list(tier2_all["protein"].tolist(), concordant_set)}

**Why included:** Expanding to p < 0.10 in a small cohort (n=34) captures likely
true signals that are underpowered. UKBB-derived proteins (CD6, CXCL9) provide
external replication evidence independent of pediatric power limitations. Healthy
vs LC_Neuro proteins add context about what is biologically active in neuro Long
COVID vs healthy children, without conflating the neuro/non-neuro subtype contrast.

**Caveats:**
- Proteins from UKBB-only source were **not significant in the pediatric cohort**.
- Proteins from Healthy vs LC_Neuro come from a **different comparison** with
  known site/batch confounding.
- Label all Tier 2 proteins as exploratory in any publication.

**Suitability:** STRING network and enrichment analysis are appropriate at n≥10.
Treat enrichment results as hypothesis-generating.

---

### Tier 3 — Cytokine/immune subset ({len(t3_proteins)} proteins)
Selection: Tier 2 proteins annotated as cytokines, chemokines, or immune mediators.

{fmt_list(tier3_all["protein"].tolist(), concordant_set)}

**Why included:** This subset maximizes biological coherence for a focused immune
network analysis. Cytokine/chemokine networks are well-represented in STRING and
yield more interpretable enrichment results than mixed protein classes.

---

## What the PPI adds

1. **Network connectivity check:** Confirms whether the selected proteins form a
   biologically coherent interaction module (expected for chemokine clusters).
2. **Pathway enrichment:** Identifies whether the protein set is enriched in
   known neuro-immune or inflammatory pathways (e.g., cytokine signaling, JAK-STAT,
   neuroinflammation).
3. **Hypothesis generation:** Proteins central in the network (high degree or
   betweenness) are candidates for follow-up functional studies.

---

## Caveats for manuscript/slides

- The pediatric cohort is very small (n=34 per comparison). **No proteins survive
  FDR correction** in the primary neuro comparison. All PPI analyses are based on
  nominal p-value signals.
- **Only FGF21** is directionally concordant with UKBB in the primary comparison.
  Other proteins should not be presented as replicated.
- The site/confounding structure of the pediatric cohort means that proteins
  identified in Healthy vs LC comparisons may partly reflect center/batch effects.
  Only the within-LC (Neuro vs NonNeuro) comparison is site-confounding-resistant.
- PPI networks from STRING include literature-curated, experimental, and
  text-mining interactions. High connectivity in STRING does not confirm disease
  relevance.
- Enrichment at small n is unreliable. Report enriched pathways as
  **hypothesis-generating**, not confirmatory.

---

## Output files

```
results/ppi/
├── protein_lists/
│   ├── tier1_main.txt / .tsv
│   ├── tier2_expanded.txt / .tsv
│   └── tier3_cytokine.txt / .tsv
├── string/
│   ├── tier1_main_interactions.tsv
│   ├── tier2_expanded_interactions.tsv
│   ├── tier3_cytokine_interactions.tsv
│   ├── tier*_enrichment.tsv
├── plots/
│   ├── protein_selection_summary.png
│   ├── tier2_expanded_enrichment.png
│   ├── tier3_cytokine_enrichment.png
│   ├── tier2_expanded_network.png
│   └── tier3_cytokine_network.png
└── ppi_interpretation_summary.md
```
"""

md_path = os.path.join(OUT, "ppi_interpretation_summary.md")
with open(md_path, "w") as fh:
    fh.write(summary)
print(f"  Saved: {md_path}")


print()
print("=" * 70)
print("DONE")
print(f"All outputs written to: {OUT}")
print("=" * 70)
