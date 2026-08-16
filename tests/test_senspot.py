"""Basic tests for SenSpot classifier."""

import pytest
import numpy as np
import pandas as pd


def make_dummy_data(n_spots=50, n_genes=500, seed=42):
    """Create minimal dummy expression data for testing."""
    rng = np.random.default_rng(seed)
    # Use some real SenSpot gene names
    genes = ["DCN", "COL1A1", "FN1", "VIM", "FBLN1",
             "SPARC", "RPS8", "RPL5", "CDKN1A", "CDKN2A"]
    genes += [f"GENE_{i}" for i in range(n_genes - len(genes))]
    data = pd.DataFrame(
        rng.poisson(5, size=(n_spots, n_genes)).astype(float),
        columns=genes,
        index=[f"BARCODE_{i}-1" for i in range(n_spots)],
    )
    return data


class TestSenSpot:
    def test_import(self):
        from senspot import SenSpot
        assert SenSpot is not None

    def test_init_default(self):
        from senspot import SenSpot
        clf = SenSpot(verbose=False)
        assert clf.threshold == 0.543
        assert clf.n_estimators == 500

    def test_init_custom(self):
        from senspot import SenSpot
        clf = SenSpot(threshold=0.510, n_jobs=2, verbose=False)
        assert clf.threshold == 0.510
        assert clf.n_jobs == 2

    def test_classify_dataframe(self):
        from senspot import SenSpot
        clf  = SenSpot(verbose=False)
        data = make_dummy_data(n_spots=20)
        probs = clf.classify(data, normalized=False, logarithmized=False)
        assert len(probs) == 20
        assert all(0 <= p <= 1 for p in probs)

    def test_classify_with_calls(self):
        from senspot import SenSpot
        clf    = SenSpot(verbose=False)
        data   = make_dummy_data(n_spots=20)
        result = clf.classify_with_calls(data, normalized=False,
                                          logarithmized=False)
        assert "senescence_probability" in result.columns
        assert "sen_positive" in result.columns
        assert len(result) == 20
        assert result["sen_positive"].dtype == bool

    def test_classify_anndata(self):
        import anndata as ad
        from senspot import SenSpot
        data  = make_dummy_data(n_spots=20)
        adata = ad.AnnData(data)
        clf   = SenSpot(verbose=False)
        probs = clf.classify(adata, normalized=False, logarithmized=False)
        assert len(probs) == 20

    def test_invalid_input_type(self):
        from senspot import SenSpot
        clf = SenSpot(verbose=False)
        with pytest.raises(TypeError):
            clf.classify([[1, 2, 3], [4, 5, 6]])

    def test_no_common_genes(self):
        from senspot import SenSpot
        clf  = SenSpot(verbose=False)
        # DataFrame with fake genes that don't exist in model
        data = pd.DataFrame(
            np.ones((10, 5)),
            columns=["FAKE1", "FAKE2", "FAKE3", "FAKE4", "FAKE5"],
        )
        with pytest.raises(ValueError, match="No common genes"):
            clf.classify(data, normalized=False, logarithmized=False)
