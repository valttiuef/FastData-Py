from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str((Path(__file__).parent.parent / "src").resolve()))

from backend.services.dimensionality_reduction_service import DimensionalityReductionService


def test_available_dimensionality_reducers_list():
    service = DimensionalityReductionService()
    reducers = service.available_dimensionality_reducers()
    reducer_keys = {key for key, _label, _defaults in reducers}

    expected_reducers = {
        "none",
        "pca",
        "pls_regression",
        "fast_ica",
        "factor_analysis",
        "truncated_svd",
    }
    assert expected_reducers.issubset(reducer_keys), f"Missing reducers: {expected_reducers - reducer_keys}"


def test_build_pca_reducer_transforms():
    service = DimensionalityReductionService()
    label, reducer, cfg = service.build_reducer(
        "pca",
        {"n_components": 1},
        n_features=3,
        n_samples=10,
    )

    assert label == "PCA"
    assert reducer is not None
    assert cfg["n_components"] == 1

    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 3))
    transformed = reducer.fit_transform(X)

    assert transformed.shape == (10, 1)


def test_build_pls_reducer_transforms():
    service = DimensionalityReductionService()
    label, reducer, cfg = service.build_reducer(
        "pls_regression",
        {"n_components": 1},
        n_features=3,
        n_samples=10,
    )

    assert label == "PLSRegression"
    assert reducer is not None
    assert cfg["n_components"] == 1

    rng = np.random.default_rng(1)
    X = rng.normal(size=(10, 3))
    y = rng.normal(size=(10,))
    reducer.fit(X, y)
    transformed = reducer.transform(X)

    assert transformed.shape == (10, 1)


def test_truncated_svd_component_bounds():
    service = DimensionalityReductionService()
    _label, _reducer, cfg = service.build_reducer(
        "truncated_svd",
        {"n_components": 10},
        n_features=2,
        n_samples=10,
    )

    assert cfg["n_components"] == 1
