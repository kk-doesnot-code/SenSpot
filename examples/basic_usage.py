"""
SenSpot — Basic Usage Example
================================

This script demonstrates how to run SenSpot on a Visium dataset,
visualise results, and compare senescence burden across sample groups.

Requirements:
    pip install forecasts-senescence scanpy matplotlib
"""

import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
from senspot import SenSpot

# ── 1. Load Visium data ───────────────────────────────────────────────────────
# Option A: from a Visium directory
adata = sc.read_visium("path/to/your/visium_sample/")

# Option B: from an h5ad file
# adata = sc.read_h5ad("path/to/your/data.h5ad")

# Option C: from a raw 10x h5 file
# adata = sc.read_10x_h5("filtered_feature_bc_matrix.h5")

print(f"Loaded: {adata.shape[0]:,} spots × {adata.shape[1]:,} genes")

# ── 2. Run SenSpot ──────────────────────────────────────────────────────────
clf = SenSpot(
    threshold=0.543,   # cross-dataset default; adjust per tissue type
    n_jobs=4,          # parallelise random forest
    verbose=True,
)

# classify() returns a list of senescence probabilities (0–1)
adata.obs["sen_probability"] = clf.classify(adata)
adata.obs["sen_positive"]    = adata.obs["sen_probability"] > clf.threshold

n_pos = adata.obs["sen_positive"].sum()
print(f"\nSen+ spots: {n_pos:,}/{adata.shape[0]:,} "
      f"({n_pos/adata.shape[0]*100:.1f}%)")

# ── 3. Spatial visualisation ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Continuous probability map
sc.pl.spatial(adata, color="sen_probability", cmap="plasma",
              vmin=0.3, vmax=0.8, ax=axes[0], show=False,
              title="Senescence Probability")

# Binary sen+/sen- map
sc.pl.spatial(adata, color="sen_positive", ax=axes[1], show=False,
              title=f"Sen+ Spots (threshold={clf.threshold})")

plt.tight_layout()
plt.savefig("forecasts_spatial_map.png", dpi=150, bbox_inches="tight")
print("Saved: forecasts_spatial_map.png")

# ── 4. Save results ───────────────────────────────────────────────────────────
results = adata.obs[["sen_probability", "sen_positive"]].copy()
results.to_csv("forecasts_results.csv")
print("Saved: forecasts_results.csv")
