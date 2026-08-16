## SenSpot: Senescence in Spatial Transcriptomics

SenSpot is a supervised machine learning classifier trained on HCA2 fibroblast scRNA-seq data for identifying senescent cells in 10x Visium spatial transcriptomics datasets.

---

### Installation

#### Step 1 — Clone

\`\`\`bash
git clone https://github.com/lach-lan/SenSpot.git
cd SenSpot
\`\`\`

#### Step 2 — Environment

\`\`\`bash
conda env create -f environment.yml
conda activate senspot
pip install -e .
senspot --version
\`\`\`

---

### Input formats

**1. Visium directory**
\`\`\`
my_sample/
├── filtered_feature_bc_matrix.h5     ← required
└── spatial/
    ├── tissue_positions_list.csv      ← required
    ├── scalefactors_json.json         ← optional
    └── tissue_lowres_image.png        ← optional
\`\`\`

**2. AnnData (.h5ad)** — gene names must be HGNC symbols

**3. 10x count matrix (.h5)**

---

### Quick start

\`\`\`bash
# Basic
senspot --input my_sample/ --output results.csv

# With threshold
senspot --input data.h5ad --output results.csv --threshold 0.535

# Full analysis
senspot --input my_sample/ --output results/ --analyse --threads 8
\`\`\`

Full analysis produces:
\`\`\`
results/
├── senspot_results.csv
├── senspot_summary.txt
├── spatial_map.png          (white bg)
├── binary_map.png           (white bg)
├── spatial_map_overlay.png  (tissue overlay, dark bg)
├── binary_map_overlay.png   (tissue overlay, dark bg)
├── top_drivers.png
└── driver_genes.csv
\`\`\`

### Python API

\`\`\`python
import scanpy as sc
from senspot import SenSpot

adata = sc.read_visium("path/to/visium_sample/")
clf = SenSpot()
adata.obs["sen_probability"] = clf.classify(adata)
adata.obs["sen_positive"] = adata.obs["sen_probability"] > 0.543
\`\`\`

---

### How it works

SenSpot trains a random forest on HCA2 fibroblast scRNA-seq data (Tang et al. 2019, GSE119807) using young (PD=38) and old (PD=48) cells as reference. Applied to each Visium spot after joint quantile normalisation.

Top 208 predictive genes (>=10x uniform baseline) enriched for EMT (FE=20.7, FDR=6.8e-19). SASP and ECM genes dominate — not CDKN1A/CDKN2A.

---

### Key results

| Dataset | Tissue | Finding |
|---------|--------|---------|
| Ganier et al. 2024 | Skin / BCC | Body 17% < Face 26% < BCC 40%, KW p=3.24e-291 |
| Wu et al. 2021 | Breast cancer | TLS 60% > Stroma 51% > Invasive 35% |
| Schäbitz et al. 2022 | Inflammatory skin | Lesional 11.3% vs Non-lesional 4.5% |
| De Jong et al. 2025 | GBM (negative control) | Mean prob=0.515 (near-baseline) |

---

### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| --input / -i | — | Visium dir, .h5ad, or .h5 |
| --output / -o | — | CSV or directory |
| --threshold / -t | 0.543 | Sen+ cutoff |
| --threads / -j | 1 | Parallel threads |
| --analyse / -a | off | Full analysis |
| --quiet / -q | off | Suppress output |

### Thresholds by dataset

| Dataset | Threshold |
|---------|-----------|
| Skin / BCC | 0.535 |
| Breast cancer | 0.555 |
| Inflammatory skin | 0.510 |
| GBM brain | 0.505 |
| Default | 0.543 |

---

### Troubleshooting

**No common genes** — use HGNC symbols: adata.var.index = adata.var['feature_name'].astype(str)

**Low sen+ in brain/liver** — expected, SenSpot detects fibroblast senescence specifically

**use_gpu error on HPC** — remove use_gpu from scvi/cell2location calls

---

### Citation

\`\`\`
Dryburgh et al. (2025). From single cells to space: predictive identification
of senescent cells in spatial transcriptomics. [Manuscript in preparation]
\`\`\`

Training data: Tang HY et al. (2019). Protein Cell, 10(5). GEO: GSE119807

Optimisation: Ganier C et al. (2024). PNAS, 121(2). ArrayExpress: E-MTAB-13084
