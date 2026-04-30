"""
ppi_network_plots.py
Improved network visualisation and metrics for PPI results.

Reads pre-computed STRING interaction files — does NOT rerun protein selection
or enrichment.  Fixes edge rendering (STRING scores are in [0,1], not [0,1000]).

Outputs:
  results/ppi/network_metrics.tsv
  results/ppi/plots/tier2_expanded_network.png   (overwrite)
  results/ppi/plots/tier3_cytokine_network.png   (overwrite)
  results/ppi/plots/tier3_network_clean.png      (slide quality)
"""

import os
import textwrap
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LISTS = os.path.join(BASE, "results", "ppi", "protein_lists")
STR   = os.path.join(BASE, "results", "ppi", "string")
OUT   = os.path.join(BASE, "results", "ppi", "plots")
PPI   = os.path.join(BASE, "results", "ppi")

# ── load tier membership ───────────────────────────────────────────────────────

def read_list(fname):
    with open(os.path.join(LISTS, fname)) as f:
        return [l.strip() for l in f if l.strip()]

TIER1 = set(read_list("tier1_main.txt"))
TIER2 = set(read_list("tier2_expanded.txt"))
TIER3 = set(read_list("tier3_cytokine.txt"))

print(f"Tier1: {sorted(TIER1)}")
print(f"Tier3: {sorted(TIER3)}")

# Node colours (Tier 1 = red, Tier 2-only = steel blue, STRING-only = light grey)
T1_COLOR  = "#e63946"
T2_COLOR  = "#457b9d"
GREY      = "#adb5bd"

SCORE_COL_A = "preferredName_A"
SCORE_COL_B = "preferredName_B"
SCORE_COL   = "score"          # 0–1 range from STRING API


# ── helper: build networkx graph ──────────────────────────────────────────────

def build_graph(tsv_path, min_score=0.4):
    df = pd.read_csv(tsv_path, sep="\t")
    df[SCORE_COL] = pd.to_numeric(df[SCORE_COL], errors="coerce")
    df = df[df[SCORE_COL] >= min_score].dropna(subset=[SCORE_COL])
    G = nx.Graph()
    for _, r in df.iterrows():
        G.add_edge(r[SCORE_COL_A], r[SCORE_COL_B], weight=r[SCORE_COL])
    print(f"  {os.path.basename(tsv_path)}: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges (score≥{min_score})")
    return G


# ── helper: network metrics ───────────────────────────────────────────────────

def compute_metrics(G, tier_label):
    rows = []
    deg = dict(G.degree())
    dc  = nx.degree_centrality(G)
    bc  = nx.betweenness_centrality(G, weight="weight", normalized=True)
    for node in G.nodes():
        rows.append({
            "protein":              node,
            "tier_label":           tier_label,
            "degree":               deg[node],
            "degree_centrality":    round(dc[node], 4),
            "betweenness_centrality": round(bc[node], 4),
            "tier1":                node in TIER1,
            "tier2":                node in TIER2,
            "tier3":                node in TIER3,
        })
    return pd.DataFrame(rows).sort_values("degree", ascending=False)


def node_color(name, tier_in_graph):
    # tier_in_graph: set of proteins that form the tier (e.g., TIER3)
    if name in TIER1 and name in tier_in_graph:
        return T1_COLOR
    if name in tier_in_graph:
        return T2_COLOR
    return GREY


# ── helper: draw network ──────────────────────────────────────────────────────

def draw_network(G, tier_set, title, out_path,
                 score_thresh_display=0.4,
                 label_nodes=None,           # set of nodes to label; None = all
                 figsize=(10, 8),
                 seed=42):
    """
    label_nodes: if None, label all; if a set, only those nodes.
    """
    if G.number_of_nodes() == 0:
        print(f"  Skipping {out_path}: empty graph")
        return

    # Layout
    pos = nx.spring_layout(G, weight="weight", seed=seed, k=2.5 / np.sqrt(G.number_of_nodes()))

    # Node attributes
    degrees   = dict(G.degree())
    max_deg   = max(degrees.values()) if degrees else 1
    node_list = list(G.nodes())
    colors    = [node_color(n, tier_set) for n in node_list]
    # node size: base 500 + up to 1500 extra based on degree fraction
    sizes     = [500 + 1500 * (degrees[n] / max_deg) for n in node_list]

    # Edge attributes — filter to score_thresh_display
    edges_to_draw = [(u, v) for u, v, d in G.edges(data=True)
                     if d.get("weight", 0) >= score_thresh_display]
    edge_weights  = [G[u][v]["weight"] for u, v in edges_to_draw]
    # scale edge width: min 0.5, max 4
    if edge_weights:
        w_min, w_max = min(edge_weights), max(edge_weights)
        span = (w_max - w_min) if w_max > w_min else 1
        edge_widths = [0.5 + 3.5 * (w - w_min) / span for w in edge_weights]
    else:
        edge_widths = []

    fig, ax = plt.subplots(figsize=figsize)

    nx.draw_networkx_edges(
        G, pos, edgelist=edges_to_draw,
        width=edge_widths, alpha=0.45,
        edge_color="#888888", ax=ax,
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=node_list,
        node_size=sizes, node_color=colors,
        linewidths=1.0, edgecolors="white",
        ax=ax,
    )

    # Labels
    if label_nodes is None:
        labels = {n: n for n in node_list}
    else:
        labels = {n: n for n in node_list if n in label_nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_weight="bold", ax=ax)

    # Legend
    patches = [
        mpatches.Patch(color=T1_COLOR, label="Tier 1 (peds adj p<0.05)"),
        mpatches.Patch(color=T2_COLOR, label="Tier 2 / Tier 3 (exploratory)"),
    ]
    ax.legend(handles=patches, fontsize=9, loc="upper left", framealpha=0.8)
    ax.set_title(title, fontsize=11, pad=12)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ── 1. Tier 2 network ─────────────────────────────────────────────────────────

print("\n=== Tier 2: expanded exploratory ===")
G2 = build_graph(os.path.join(STR, "tier2_expanded_interactions.tsv"), min_score=0.4)
m2 = compute_metrics(G2, "tier2_expanded")
draw_network(
    G2, TIER2,
    title="STRING PPI — Tier 2 expanded (28 proteins)\nEdge width ∝ combined score",
    out_path=os.path.join(OUT, "tier2_expanded_network.png"),
    score_thresh_display=0.5,   # hide weakest edges for clarity
    label_nodes=None,
    figsize=(12, 9),
)

# ── 2. Tier 3 full network ────────────────────────────────────────────────────

print("\n=== Tier 3: cytokine/immune subset ===")
G3 = build_graph(os.path.join(STR, "tier3_cytokine_interactions.tsv"), min_score=0.4)
m3 = compute_metrics(G3, "tier3_cytokine")
draw_network(
    G3, TIER3,
    title="STRING PPI — Tier 3 cytokine/immune subset (17 proteins)\nEdge width ∝ combined score",
    out_path=os.path.join(OUT, "tier3_cytokine_network.png"),
    score_thresh_display=0.5,
    label_nodes=None,
    figsize=(10, 8),
)

# ── 3. Tier 3 slide-quality clean plot ───────────────────────────────────────

print("\n=== Tier 3: slide-quality clean plot ===")

# Use higher score threshold so only high-confidence edges are shown
G3_clean = build_graph(os.path.join(STR, "tier3_cytokine_interactions.tsv"), min_score=0.6)

# Label only: Tier 1 proteins in this network + top 5 hubs
top5_hubs = set(m3.nlargest(5, "degree")["protein"].tolist())
tier1_in_t3 = TIER1 & TIER3
label_set = tier1_in_t3 | top5_hubs
print(f"  Labelling: {sorted(label_set)}")

# Palette: consistent with full network
t3_degrees = dict(G3_clean.degree())
max_deg3   = max(t3_degrees.values()) if t3_degrees else 1

node_list3 = list(G3_clean.nodes())
colors3    = []
sizes3     = []
for n in node_list3:
    colors3.append(node_color(n, TIER3))
    deg = t3_degrees.get(n, 0)
    sizes3.append(400 + 1800 * (deg / max_deg3))

edges3     = [(u, v) for u, v in G3_clean.edges()]
weights3   = [G3_clean[u][v]["weight"] for u, v in edges3]
if weights3:
    w_min3, w_max3 = min(weights3), max(weights3)
    span3 = (w_max3 - w_min3) if w_max3 > w_min3 else 1
    ewidths3 = [0.8 + 4.5 * (w - w_min3) / span3 for w in weights3]
else:
    ewidths3 = []

pos3 = nx.spring_layout(G3_clean, weight="weight", seed=7, k=3.2 / np.sqrt(max(G3_clean.number_of_nodes(), 1)))

fig, ax = plt.subplots(figsize=(10, 8))

nx.draw_networkx_edges(
    G3_clean, pos3, edgelist=edges3,
    width=ewidths3, alpha=0.5,
    edge_color="#999999", ax=ax,
)
nx.draw_networkx_nodes(
    G3_clean, pos3, nodelist=node_list3,
    node_size=sizes3, node_color=colors3,
    linewidths=1.2, edgecolors="white", ax=ax,
)
# All labels, but differentiated by tier
for n in node_list3:
    x, y = pos3[n]
    weight = "bold" if n in label_set else "normal"
    fontsize = 9 if n in label_set else 7
    color = "black" if n in label_set else "#444444"
    ax.text(x, y + 0.05, n, ha="center", va="bottom",
            fontsize=fontsize, fontweight=weight, color=color)

# Legend
patches = [
    mpatches.Patch(color=T1_COLOR, label="Tier 1 protein (peds adj p<0.05)"),
    mpatches.Patch(color=T2_COLOR, label="Cytokine/immune (exploratory)"),
]
ax.legend(handles=patches, fontsize=9, loc="lower left", framealpha=0.9)
ax.set_title(
    "Pediatric Long COVID — LC_Neuro vs LC_NonNeuro\n"
    "Cytokine/chemokine PPI network (STRING score ≥ 0.6)\n"
    "Node size ∝ degree · bold labels = Tier 1 or top hub",
    fontsize=10, pad=14,
)
ax.axis("off")
plt.tight_layout()
clean_path = os.path.join(OUT, "tier3_network_clean.png")
plt.savefig(clean_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"  Saved: {clean_path}")


# ── 4. Save network metrics ───────────────────────────────────────────────────

print("\n=== Network metrics ===")
metrics = pd.concat([m2, m3], ignore_index=True)
metrics_path = os.path.join(PPI, "network_metrics.tsv")
metrics.to_csv(metrics_path, sep="\t", index=False)
print(f"  Saved: {metrics_path}")

# Print tier3 metrics summary
print("\nTier 3 — top 10 proteins by degree:")
print(m3[["protein", "degree", "degree_centrality", "betweenness_centrality",
          "tier1", "tier2"]].head(10).to_string(index=False))

# Connected components
comps3 = list(nx.connected_components(G3))
print(f"\nTier 3 connected components: {len(comps3)}")
for i, comp in enumerate(sorted(comps3, key=len, reverse=True)):
    print(f"  Component {i+1} ({len(comp)} nodes): {sorted(comp)}")

comps2 = list(nx.connected_components(G2))
print(f"\nTier 2 connected components: {len(comps2)}")
for i, comp in enumerate(sorted(comps2, key=len, reverse=True)):
    print(f"  Component {i+1} ({len(comp)} nodes): {sorted(comp)[:10]}{'...' if len(comp)>10 else ''}")


# ── 5. Slide interpretation text ─────────────────────────────────────────────

print("\n=== Slide interpretation ===")
top5 = m3.nlargest(5, "degree")[["protein", "degree"]].values.tolist()
top5_str = ", ".join(f"{p} (degree {d})" for p, d in top5)

interp = f"""
Slide-ready interpretation bullets (Tier 3 — cytokine/immune network):

• The 17-protein cytokine/chemokine network forms {len(comps3)} connected component(s),
  with all proteins interacting above STRING medium-confidence (score ≥ 0.40).

• Top hub proteins by degree: {top5_str}.
  These are the most centrally connected nodes and strongest candidates
  for pathway follow-up.

• CCL2 and IL17C (red nodes) are the only Tier 1 proteins in this network,
  meaning they have the strongest statistical evidence in the pediatric cohort.
  Both are embedded in the main chemokine cluster, consistent with a
  neuroinflammatory chemokine response in LC_Neuro children.

• STRING enrichment confirms this network is dominated by cytokine–receptor
  interaction (KEGG FDR 2.8×10⁻²⁶), chemokine signaling (FDR 1.9×10⁻¹⁰),
  and IL-10 signaling (Reactome FDR 2.0×10⁻⁷), suggesting immune regulatory
  dysregulation as a potential biological theme in pediatric neuro Long COVID.

Caveat: All findings are exploratory (no FDR correction survives in n=34).
Interpret as hypothesis-generating only.
"""
print(interp)

interp_path = os.path.join(PPI, "network_interpretation.md")
with open(interp_path, "w") as fh:
    fh.write(interp.strip() + "\n")
print(f"  Saved: {interp_path}")
