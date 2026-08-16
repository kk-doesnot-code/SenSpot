"""
SenSpot command-line interface.

Usage examples:
    senspot --input path/to/visium/ --output results.csv
    senspot --input data.h5ad --output results.csv --threshold 0.535
    senspot --input data.h5ad --output results/ --analyse --threads 8
"""

import argparse
import sys
import os


def parse_args():
    parser = argparse.ArgumentParser(
        prog="senspot",
        description=(
            "SenSpot: FOREst-based ClAssification of Senescence "
            "in Spatial TranScriptomics."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Basic run — probability + binary calls
  senspot --input path/to/visium_sample/ --output results.csv

  # Full analysis — maps, distribution, driver genes
  senspot --input data.h5ad --output results/ --analyse

  # Custom threshold + parallelisation
  senspot --input data.h5ad --output results.csv --threshold 0.535 --threads 8

  # Raw counts
  senspot --input data.h5ad --output results.csv --raw

typical thresholds:
  Skin / BCC (Ganier 2024):        0.535
  Breast cancer (Wu 2021):         0.555
  Inflammatory skin (Schäbitz):    0.510
  GBM brain (De Jong 2025):        0.505
  Cross-dataset default:           0.543
        """,
    )

    parser.add_argument("--input",  "-i", required=True,
        help="Visium directory, .h5ad, or .h5 file.")
    parser.add_argument("--output", "-o", required=True,
        help="Output CSV file, or output directory if --analyse is used.")
    parser.add_argument("--threshold", "-t", type=float, default=0.543,
        help="Classification threshold (default: 0.543).")
    parser.add_argument("--threads", "-j", type=int, default=1,
        help="Parallel threads for random forest (default: 1).")
    parser.add_argument("--analyse", "-a", action="store_true",
        help=(
            "Run full analysis: spatial maps, probability distribution, "
            "top driver genes, and summary statistics. "
            "Output must be a directory when this flag is used."
        ))
    parser.add_argument("--raw", action="store_true",
        help="Input is raw counts (not yet normalised or log-transformed).")
    parser.add_argument("--layer", type=str, default=None,
        help="AnnData layer to use (default: .X).")
    parser.add_argument("--quiet", "-q", action="store_true",
        help="Suppress progress output.")
    parser.add_argument("--version", "-v", action="version",
        version="SenSpot 1.1.0")

    return parser.parse_args()


def load_input(path, raw=False, layer=None):
    import scanpy as sc
    import pandas as pd
    import anndata as ad

    path = path.rstrip("/")

    if os.path.isdir(path):
        h5_path = os.path.join(path, "filtered_feature_bc_matrix.h5")
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"No filtered_feature_bc_matrix.h5 in {path}")
        adata = sc.read_10x_h5(h5_path)
        adata.var_names_make_unique()
        pos_path = os.path.join(path, "spatial", "tissue_positions_list.csv")
        if os.path.exists(pos_path):
            pos = pd.read_csv(pos_path, header=None,
                              names=["barcode","in_tissue","array_row",
                                     "array_col","pxl_row","pxl_col"])
            pos = pos[pos["in_tissue"]==1].set_index("barcode")
            adata = adata[adata.obs_names.isin(pos.index)].copy()
            adata.obsm["spatial"] = pos.loc[
                adata.obs_names, ["pxl_col","pxl_row"]].values
        # Try loading lowres image
        import json
        sf_path  = os.path.join(path, "spatial", "scalefactors_json.json")
        img_path = os.path.join(path, "spatial", "tissue_lowres_image.png")
        if os.path.exists(sf_path) and os.path.exists(img_path):
            import numpy as np
            from PIL import Image
            with open(sf_path) as f:
                sf = json.load(f)
            img = np.array(Image.open(img_path))
            lib_id = os.path.basename(path)
            adata.uns["spatial"] = {lib_id: {
                "images": {"lowres": img},
                "scalefactors": sf
            }}
        return adata

    elif path.endswith(".h5ad"):
        return ad.read_h5ad(path)

    elif path.endswith(".h5"):
        adata = sc.read_10x_h5(path)
        adata.var_names_make_unique()
        return adata

    else:
        raise ValueError(f"Unrecognised input format: {path}")



def _is_raw_counts(adata):
    """Auto-detect if AnnData contains raw counts or normalised data."""
    import scipy.sparse as sp
    import numpy as np
    X = adata.X[:50].toarray() if sp.issparse(adata.X) else adata.X[:50]
    is_integer = np.allclose(X, X.astype(int), atol=1e-3)
    max_val    = float(X.max())
    return is_integer and max_val > 20

def main():
    args = parse_args()

    # Validate: --analyse requires a directory output
    if args.analyse and args.output.endswith(".csv"):
        print("Error: --analyse requires --output to be a directory, not a .csv file.",
              file=sys.stderr)
        print("Example: senspot --input data.h5ad --output results/ --analyse",
              file=sys.stderr)
        sys.exit(1)

    from senspot import SenSpot

    try:
        adata = load_input(args.input, raw=args.raw, layer=args.layer)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading input: {e}", file=sys.stderr)
        sys.exit(1)

    if not args.quiet:
        print(f"Loaded: {adata.shape[0]:,} spots × {adata.shape[1]:,} genes")

    # Auto-detect raw counts if --raw not specified
    is_raw = args.raw or _is_raw_counts(adata)
    if is_raw and not args.raw and not args.quiet:
        print("Auto-detected raw counts — applying normalize_total + log1p")
    elif not is_raw and not args.quiet:
        print("Auto-detected normalised data — skipping normalisation")

    clf = SenSpot(
        threshold=args.threshold,
        n_jobs=args.threads,
        verbose=not args.quiet,
    )

    results = clf.classify_with_calls(
        adata,
        normalized=not is_raw,
        logarithmized=not is_raw,
        layer=args.layer,
    )

    # ── Save results CSV ──────────────────────────────────────
    if args.analyse:
        os.makedirs(args.output, exist_ok=True)
        csv_path = os.path.join(args.output, "senspot_results.csv")
    else:
        csv_path = args.output

    results.to_csv(csv_path)
    if not args.quiet:
        n_pos = results["sen_positive"].sum()
        pct   = n_pos / len(results) * 100
        print(f"\nResults saved to: {csv_path}")
        print(f"Sen+ spots: {n_pos:,}/{len(results):,} ({pct:.1f}%)")
        print(f"Threshold:  {args.threshold}")

    # ── Full analysis ─────────────────────────────────────────
    if args.analyse:
        from senspot.analyse import run_analysis
        sample_name = os.path.basename(args.input.rstrip("/"))
        run_analysis(
            adata=adata,
            results=results,
            output_dir=args.output,
            sample_name=sample_name,
            threshold=args.threshold,
            layer=args.layer,
            verbose=not args.quiet,
        )


if __name__ == "__main__":
    main()

def detect_raw(adata):
    """Auto-detect if data is raw counts or normalised."""
    import scipy.sparse as sp
    import numpy as np
    X = adata.X[:10].toarray() if sp.issparse(adata.X) else adata.X[:10]
    # Raw counts: integers, max typically >10
    # Normalised: floats, max typically <15 after log1p
    is_integer = np.allclose(X, X.astype(int))
    max_val = X.max()
    return is_integer and max_val > 20
