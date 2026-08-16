"""
SenSpot — Multi-Sample Comparison Example
=============================================

Compare senescence burden across sample groups (e.g. normal vs disease).
Replicates the Ganier et al. 2024 skin atlas analysis:
    Normal body < Normal face < BCC (KW p = 3.24e-291)
"""

import os
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from senspot import SenSpot

# ── Define samples ────────────────────────────────────────────────────────────
SAMPLES = {
    "normal_body_back":   ("Normal Body", "#4A90C4"),
    "normal_face_nose":   ("Normal Face", "#5BAD72"),
    "BCC_nose":           ("BCC",         "#C94040"),
}
DATA_DIR  = "path/to/ganier_data/"
THRESHOLD = 0.543

# ── Run SenSpot on each sample ──────────────────────────────────────────────
clf     = SenSpot(threshold=THRESHOLD, n_jobs=4, verbose=True)
results = []

for sample, (group, color) in SAMPLES.items():
    h5_path = os.path.join(DATA_DIR, sample, "filtered_feature_bc_matrix.h5")
    if not os.path.exists(h5_path):
        print(f"Skipping {sample} — file not found"); continue

    adata = sc.read_10x_h5(h5_path)
    adata.var_names_make_unique()

    probs = clf.classify(adata, verbose=False)
    prop  = sum(p > THRESHOLD for p in probs) / len(probs)

    for prob in probs:
        results.append({"sample": sample, "group": group,
                         "color": color, "probability": prob})

    print(f"  {sample} ({group}): {prop*100:.1f}% sen+")

df = pd.DataFrame(results)
df.to_csv("forecasts_multi_sample.csv", index=False)

# ── Statistics ────────────────────────────────────────────────────────────────
groups     = df["group"].unique()
group_data = [df[df["group"] == g]["probability"].values for g in groups]
kw_stat, kw_p = stats.kruskal(*group_data)
print(f"\nKruskal-Wallis: H={kw_stat:.2f}, p={kw_p:.2e}")

# ── Boxplot ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
fig.patch.set_facecolor("white")

group_order  = ["Normal Body", "Normal Face", "BCC"]
group_colors = {"Normal Body": "#4A90C4",
                "Normal Face": "#5BAD72",
                "BCC":         "#C94040"}

plot_data = [df[df["group"] == g]["probability"].values for g in group_order]
bp = ax.boxplot(plot_data, patch_artist=True, widths=0.5,
                medianprops=dict(color="black", lw=2),
                flierprops=dict(marker="o", ms=1.5, alpha=0.2))

for patch, g in zip(bp["boxes"], group_order):
    patch.set_facecolor(group_colors[g])
    patch.set_alpha(0.75)

ax.axhline(THRESHOLD, color="grey", lw=1, linestyle="--", alpha=0.5)
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(group_order, fontsize=12)
ax.set_ylabel("Senescence Probability", fontsize=12)
ax.set_title("SenSpot — Senescence by Group", fontsize=12, fontweight="bold")
ax.text(0.98, 0.02, f"KW p={kw_p:.2e}", transform=ax.transAxes,
        ha="right", fontsize=9, color="grey")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("forecasts_groups.png", dpi=150, bbox_inches="tight")
print("Saved: forecasts_groups.png")
