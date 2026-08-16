## SenSpot: Senescence in Spatial Transcriptomics

SenSpot is a supervised machine learning classifier trained on HCA2 fibroblast scRNA-seq data for identifying senescent cells in 10x Visium spatial transcriptomics datasets.

---

### Installation

**Step 1 — Clone**

```bash
git clone https://github.com/kk-doesnot-code/SenSpot.git
cd SenSpot
```

**Step 2 — Environment**

```bash
conda env create -f environment.yml
conda activate senspot
pip install -e .
senspot --version
```

---

### Input formats

**1. Visium directory**

my_sample/
├── filtered_feature_bc_matrix.h5 ← required
└── spatial/
├── tissue_positions_list.csv ← required
├── scalefactors_json.json ← optional (image overlay)
└── tissue_lowres_image.png ← optional (image overlay)

**2. AnnData (.h5ad)** — gene names must be HGNC symbols (e.g. DCN, COL1A1)

> For CELLxGENE h5ad: `adata.var.index = adata.var['feature_name'].astype(str)`

**3. 10x count matrix (.h5)**

---

### Quick start

**Basic**

```bash
senspot --input my_sample/ --output results.csv
```

**With threshold**

```bash
senspot --input data.h5ad --output results.csv --threshold 0.535
```

**Full analysis**

```bash
senspot --input my_sample/ --output results/ --analyse --threads 8
```

Full analysis produces:

results/
├── senspot_results.csv
├── senspot_summary.txt
├── spatial_map.png (white background)
├── binary_map.png (white background)
├── spatial_map_overlay.png (tissue overlay, dark background)
├── binary_map_overlay.png (tissue overlay, dark background)
├── top_drivers.png
└── driver_genes.csv

### Python API

```python
import scanpy as sc
from senspot import SenSpot

adata = sc.read_visium("path/to/visium_sample/")
clf = SenSpot()
adata.obs["sen_probability"] = clf.classify(adata)
adata.obs["sen_positive"] = adata.obs["sen_probability"] > 0.543

# Or get both in one call
results = clf.classify_with_calls(adata)
results.to_csv("results.csv")
```

---

### CLI flags

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--input` | `-i` | — | Visium dir, .h5ad, or .h5 |
| `--output` | `-o` | — | CSV or directory |
| `--threshold` | `-t` | `0.543` | Sen+ probability cutoff |
| `--threads` | `-j` | `1` | Parallel threads |
| `--analyse` | `-a` | off | Full analysis mode |
| `--quiet` | `-q` | off | Suppress output |

---

### Repository structure

SenSpot/
├── senspot/
│ ├── model.py ← SenSpot classifier
│ ├── cli.py ← command-line interface
│ └── analyse.py ← analysis module
├── Model/ ← pre-trained reference (HCA2 fibroblasts)
├── images/
├── examples/
├── tests/
├── environment.yml
└── pyproject.toml

---

### Citation

MAR GROUP (2026),
LL ,Jess ,Huiwen ,Kuber

Training data: Tang HY et al. (2019). Protein Cell, 10(5). GEO: GSE119807

Optimisation dataset: Ganier C et al. (2024). PNAS, 121(2). ArrayExpress: E-MTAB-13084