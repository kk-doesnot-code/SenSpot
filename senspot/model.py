"""
SenSpot: Senescence in Spatial Transcriptomics

A random forest classifier trained on HCA2 fibroblast scRNA-seq data that
predicts senescence probability per Visium spot.

Reference:
    Tang et al. (2019). Single senescent cell sequencing reveals heterogeneity
    in senescent cells induced by telomere erosion. Protein Cell, 10(5), 370-375.
    GEO: GSE119807

Optimised using:
    Ganier et al. (2024). Multiscale spatial mapping of cell populations across
    anatomical sites in healthy human skin and basal cell carcinoma.
    PNAS, 121(2), e2313326120.
"""

import os
import pickle
import time
from pathlib import Path
from typing import Union, Optional

import numpy as np
import pandas as pd
import scanpy as sc
import qnorm
import anndata as ad
from anndata import AnnData
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Path to bundled model files
_MODEL_DIR = Path(__file__).parent.parent / "Model"


class SenSpot:
    """
    Senescence in Spatial Transcriptomics.

    Classifies spatial transcriptomics spots as senescent or non-senescent
    using a random forest trained on HCA2 fibroblast scRNA-seq data.

    Parameters
    ----------
    threshold : float, optional
        Classification threshold for sen+ calls. Default 0.543 (optimised).
        Lower values increase sensitivity, higher values increase specificity.
    n_estimators : int, optional
        Number of trees in the random forest. Default 500.
    n_jobs : int, optional
        Number of parallel jobs for the random forest. Default 1.
    random_state : int, optional
        Random seed for reproducibility. Default 42.
    verbose : bool, optional
        Print progress messages. Default True.

    Examples
    --------
    Basic usage with AnnData:

    >>> import scanpy as sc
    >>> from senspot import SenSpot
    >>> adata = sc.read_visium("path/to/visium/")
    >>> clf = SenSpot()
    >>> adata.obs['sen_probability'] = clf.classify(adata)
    >>> adata.obs['sen_positive']   = adata.obs['sen_probability'] > 0.543

    Command line:

    .. code-block:: bash

        senspot --input my_visium/ --output results.csv

    """

    def __init__(
        self,
        threshold: float = 0.543,
        n_estimators: int = 500,
        n_jobs: int = 1,
        random_state: int = 42,
        verbose: bool = True,
    ):
        self.threshold    = threshold
        self.n_estimators = n_estimators
        self.n_jobs       = n_jobs
        self.random_state = random_state
        self.verbose      = verbose

        self._load_training_data()

    def _load_training_data(self):
        """Load and preprocess the HCA2 fibroblast reference data."""
        model_dir = _MODEL_DIR

        genes_path = model_dir / "model_genes.pkl"
        yng_path   = model_dir / "model_yng.pkl"
        old_path   = model_dir / "model_old.pkl"

        for p in [genes_path, yng_path, old_path]:
            if not p.exists():
                raise FileNotFoundError(
                    f"Model file not found: {p}\n"
                    f"Ensure the Model/ directory is present alongside the "
                    f"senspot/ package."
                )

        with open(genes_path, "rb") as f:
            genes = pickle.load(f)
        with open(yng_path, "rb") as f:
            yng = pd.DataFrame(pickle.load(f).todense(),
                               index=[0] * 400, columns=genes)
        with open(old_path, "rb") as f:
            old = pd.DataFrame(pickle.load(f).todense(),
                               index=[1] * 400, columns=genes)

        # Normalise reference data
        yng_a = ad.AnnData(yng)
        old_a = ad.AnnData(old)
        sc.pp.normalize_total(yng_a, exclude_highly_expressed=True)
        sc.pp.normalize_total(old_a, exclude_highly_expressed=True)
        sc.pp.log1p(yng_a)
        sc.pp.log1p(old_a)
        yng = yng_a.to_df().astype(float)
        old = old_a.to_df().astype(float)

        # Train/test split
        yng_trn, yng_tst = train_test_split(yng, test_size=0.1,
                                             random_state=self.random_state)
        old_trn, old_tst = train_test_split(old, test_size=0.1,
                                             random_state=self.random_state)
        self._trn_X = pd.concat([yng_trn, old_trn])
        self._trn_y = list(self._trn_X.index)
        self._tst_X = pd.concat([yng_tst, old_tst])
        self._tst_y = list(self._tst_X.index)

    def _preprocess(
        self,
        st_df: pd.DataFrame,
        normalized: bool,
        logarithmized: bool,
    ):
        """
        Align spatial data with reference via quantile normalisation.

        Performs library-size normalisation and log transformation if needed,
        subsets to shared genes, then applies joint quantile normalisation
        across reference and spatial data followed by standard scaling.
        """
        # Normalise if raw counts provided
        if not normalized:
            adata = ad.AnnData(X=st_df)
            sc.pp.normalize_total(adata, exclude_highly_expressed=True)
            st_df = adata.to_df()

        # Log transform if not already done
        if not logarithmized:
            st_df = pd.DataFrame(
                data=sc.pp.log1p(np.asarray(st_df)),
                index=st_df.index,
                columns=st_df.columns,
            )

        # Subset to shared genes
        common_genes = self._trn_X.columns.intersection(st_df.columns)
        if len(common_genes) == 0:
            raise ValueError(
                "No common genes found between input data and model genes. "
                "Check that gene names are HGNC symbols (e.g. DCN, COL1A1)."
            )

        trn_X = self._trn_X[common_genes]
        tst_X = self._tst_X[common_genes]
        st_df = st_df[common_genes]
        st_df = st_df.loc[:, ~st_df.columns.duplicated(keep="first")]

        if self.verbose:
            print(f"  {len(common_genes):,} genes shared with model "
                  f"(of {len(self._trn_X.columns):,} model genes)")

        # Joint quantile normalisation
        all_samples = pd.concat([trn_X, tst_X, st_df], axis=0)
        qn = qnorm.quantile_normalize(all_samples, axis=0)

        n_trn = self._trn_X.shape[0]
        n_tst = self._tst_X.shape[0]
        trn_X = qn[:n_trn]
        tst_X = qn[n_trn: n_trn + n_tst]
        st_df = qn[n_trn + n_tst:]

        # Standard scaling
        scaler = StandardScaler()
        trn_X = scaler.fit_transform(trn_X)
        tst_X = scaler.fit_transform(tst_X)
        st_df = scaler.fit_transform(st_df)

        return trn_X, tst_X, st_df

    def classify(
        self,
        data: Union[pd.DataFrame, AnnData],
        normalized: bool = True,
        logarithmized: bool = True,
        layer: Optional[str] = None,
    ) -> list:
        """
        Classify spots as senescent or non-senescent.

        Parameters
        ----------
        data : pd.DataFrame or AnnData
            Spatial transcriptomics data. Rows = spots/barcodes,
            columns = genes (HGNC symbols). If AnnData, uses .X or
            the specified layer.
        normalized : bool
            Whether library-size normalisation has already been applied.
            Set False to apply normalize_total internally.
        logarithmized : bool
            Whether log1p transformation has already been applied.
            Set False to apply log1p internally.
        layer : str, optional
            Layer in AnnData to use. Default None (uses .X).

        Returns
        -------
        list of float
            Senescence probability per spot (0–1).
            Apply threshold (default 0.543) to get binary calls.

        Notes
        -----
        Senescence probability > threshold → sen+
        Typical thresholds by dataset:
            Skin (Ganier 2024):       0.535
            Breast cancer (Wu 2021):  0.555
            Inflammatory (Schäbitz): 0.510
            GBM (De Jong 2025):       0.505
            Cross-dataset default:    0.543
        """
        if not isinstance(data, (pd.DataFrame, AnnData)):
            raise TypeError("data must be a pandas DataFrame or AnnData object.")

        if self.verbose:
            print("SenSpot: Pre-processing...")
            start = time.time()

        # Extract expression matrix
        if isinstance(data, AnnData):
            st_df = data.to_df(layer=layer)
        else:
            st_df = data

        trn_X, tst_X, st_s = self._preprocess(
            st_df, normalized=normalized, logarithmized=logarithmized)

        if self.verbose:
            print(f"  Pre-processing complete in {time.time()-start:.1f}s")
            print("SenSpot: Training classifier...")
            start = time.time()

        # Train random forest
        clf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=2,
            max_features=173,
            max_samples=0.401,
            criterion="gini",
            min_samples_split=2,
            min_samples_leaf=1,
            ccp_alpha=0.0,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        ).fit(trn_X, self._trn_y)

        if self.verbose:
            preds    = clf.predict(tst_X)
            accuracy = np.mean(np.array(preds) == np.array(self._tst_y))
            print(f"  Hold-out scRNA-seq accuracy: {accuracy*100:.1f}%")

        probabilities = [x[1] for x in clf.predict_proba(st_s)]

        if self.verbose:
            n_pos   = sum(p > self.threshold for p in probabilities)
            pct_pos = n_pos / len(probabilities) * 100
            print(f"  Modelling complete in {time.time()-start:.1f}s")
            print(f"  {n_pos:,}/{len(probabilities):,} spots sen+ "
                  f"({pct_pos:.1f}%, threshold={self.threshold})")

        return probabilities

    def classify_with_calls(
        self,
        data: Union[pd.DataFrame, AnnData],
        normalized: bool = True,
        logarithmized: bool = True,
        layer: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Classify and return a DataFrame with probabilities and binary calls.

        Returns
        -------
        pd.DataFrame
            Columns: senescence_probability, sen_positive
        """
        probs = self.classify(data, normalized=normalized,
                              logarithmized=logarithmized, layer=layer)

        if isinstance(data, AnnData):
            index = data.obs_names
        else:
            index = data.index

        return pd.DataFrame({
            "senescence_probability": probs,
            "sen_positive": [p > self.threshold for p in probs],
        }, index=index)
