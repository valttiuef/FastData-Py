
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import threading

import numpy as np
import pandas as pd

from minisom import MiniSom, _build_iteration_indexes

from .clustering import (

    ClusteringService,
    FeatureClusteringResult,
    NeuronClusteringResult,
    ScoreType,
    ClusteringMethodSpec,
)

import logging
logger = logging.getLogger(__name__)

class MiniSomWithCallback(MiniSom):
    # --- TRUE BATCH EPOCH (vectorized) ---
    def _batch_epoch(self, X: np.ndarray, sigma: float, chunk: int = 2048):
        """
        One batch epoch:
          1) BMUs for all samples (vectorized, chunked)
          2) Neighborhood weights H[:, j] = exp(-||r_i - r_bmu(j)||^2 / (2*sigma^2))
          3) W_new = (H @ X) / (H @ 1)
        X: (n, d), weights: (mx, my, d)
        """
        W = self._weights  # (mx, my, d)
        mx, my, d = W.shape
        m = mx * my
        W2 = W.reshape(m, d)

        # neuron coordinates grid (cache this if you like)
        rx, ry = np.meshgrid(np.arange(mx), np.arange(my), indexing="ij")
        R = np.stack([rx.reshape(-1), ry.reshape(-1)], axis=1).astype(np.float32)  # (m, 2)

        # ---- BMU indices for all samples (vectorized in chunks) ----
        n = X.shape[0]
        bmu_idx = np.empty(n, dtype=np.int32)
        for start in range(0, n, chunk):
            stop = min(n, start + chunk)
            Xc = X[start:stop].astype(W2.dtype, copy=False)  # (c, d)
            # squared distances to all neurons: ||X - W||^2 = ||X||^2 + ||W||^2 - 2 XW^T
            # compute with matrix ops in chunks
            x2 = np.sum(Xc * Xc, axis=1, keepdims=True)          # (c, 1)
            w2 = np.sum(W2 * W2, axis=1, keepdims=True).T        # (1, m)
            # (c, m)
            d2 = x2 + w2 - (2.0 * (Xc @ W2.T))
            bmu_idx[start:stop] = np.argmin(d2, axis=1).astype(np.int32)

        # ---- Neighborhood matrix H (m, n) built via BMU coords ----
        bmu_rc = R[bmu_idx]                           # (n, 2)
        # squared grid distance from each neuron to each sample’s BMU
        # (m, n): ||R_i - r_bmu(j)||^2
        dRc2 = (R[:, None, :] - bmu_rc[None, :, :])
        dRc2 = (dRc2 * dRc2).sum(axis=2)
        # Gaussian neighborhood
        denom = 2.0 * (float(sigma) ** 2) + 1e-12
        H = np.exp(-dRc2 / denom).astype(np.float32, copy=False)  # (m, n)

        # ---- Batch update ----
        # W_new = (H @ X) / (H @ 1)
        num = H @ X.astype(np.float32, copy=False)                # (m, d)
        den = H.sum(axis=1, keepdims=True) + 1e-12                # (m, 1)
        W2[:] = (num / den).astype(W2.dtype, copy=False)

    def train(self, data, num_iteration,
              random_order=False, verbose=False,
              use_epochs=False, fixed_points=None,
              callback=None, callback_every: int = 10, report_qe: bool = False,
              mode: str = "online",  # "online" (default) | "batch"
              sigma_schedule: tuple = (6.0, 1.0)):
        """
        Override with a real 'batch' mode.

        - mode="online": original MiniSom behavior with per-sample updates.
        - mode="batch" : true batch epochs using vectorized updates.
        - num_iteration: if use_epochs=True, this is EPOCHS. Otherwise, it's raw updates (online) or epochs (batch).
        """
        mode = (mode or "online").lower()
        self._check_iteration_number(num_iteration)
        self._check_input_len(data)

        X = np.asarray(data)
        if X.ndim != 2:
            raise ValueError("data must be 2-D (n_samples, n_features)")

        if mode == "batch":
            # ---- TRUE BATCH TRAINING ----
            epochs = int(num_iteration if use_epochs or num_iteration > 0 else 1)
            mx, my, _ = self._weights.shape

            sigma0, sigmaf = sigma_schedule  # e.g., (6.0, 1.0)
            # exponential decay per epoch (MATLAB-like)
            def sigma_at(e):
                if epochs <= 1:
                    return sigmaf
                frac = e / float(epochs - 1)
                return float(sigma0) * ((float(sigmaf) / float(sigma0)) ** frac)

            last_prog = -1
            for e in range(epochs):
                sigma_t = sigma_at(e)
                self._batch_epoch(X, sigma=sigma_t)

                if callback:
                    # progress ~ epochs
                    pct = int(round(100.0 * (e + 1) / float(epochs)))
                    if pct != last_prog:
                        last_prog = pct
                        qe = self.quantization_error(X) if report_qe else None
                        callback(pct / 100.0, e, None, e, self, qe)

            if verbose:
                print('\n quantization error:', self.quantization_error(X))
            return

        # ---- ONLINE / RANDOM path (original behavior with callbacks) ----
        random_generator = None
        if random_order:
            random_generator = self._random_generator

        iterations = _build_iteration_indexes(
            len(X), num_iteration, verbose, random_generator, use_epochs
        )

        if use_epochs:
            total_updates = len(X) * num_iteration
            def get_decay_rate(iteration_index, data_len):
                return int(iteration_index / data_len)
        else:
            total_updates = num_iteration
            def get_decay_rate(iteration_index, data_len):
                return int(iteration_index)

        if fixed_points:
            self._check_fixed_points(fixed_points, X)
        else:
            fixed_points = {}

        for t, iteration in enumerate(iterations):
            decay_rate = get_decay_rate(t, len(X))
            self.update(
                X[iteration],
                fixed_points.get(iteration, self.winner(X[iteration])),
                decay_rate,
                num_iteration
            )
            if callback and (t + 1) % max(1, int(callback_every)) == 0:
                progress = (t + 1) / float(max(1, total_updates))
                qe = self.quantization_error(X) if report_qe else None
                callback(progress, t, iteration, decay_rate, self, qe)

        if verbose:
            print('\n quantization error:', self.quantization_error(X))

    def pca_weights_init(self, data: np.ndarray) -> None:
        """
        Robust PCA weight initialization for MiniSom.

        Expects `data` shaped (n_samples, n_features) and already normalized the
        same way you will train. Initializes self._weights[:, :, :] accordingly.
        """
        # ---- 0) shape / guards ----
        self._check_input_len(data)
        if self._input_len < 2:
            raise ValueError("Need at least 2 features for PCA initialization.")
        if len(self._neigx) == 1 or len(self._neigy) == 1:
            # keep behavior but do not abort; warn once
            from warnings import warn
            warn("PCA init on a 1D SOM grid is not very meaningful.")

        X = np.asarray(data)

        # ---- 1) sanitize rows and drop zero-variance columns ----
        finite_rows = np.isfinite(X).all(axis=1)
        X = X[finite_rows]
        if X.shape[0] == 0:
            raise ValueError("No finite rows left for PCA initialization.")

        var = X.var(axis=0)
        keep = var > 0.0
        if keep.sum() < 2:
            # fall back to random if fewer than 2 informative features
            self.random_weights_init(data[finite_rows])
            return
        Xr = X[:, keep].astype(np.float64, copy=False)

        # ---- 2) center and SVD (robust PCA) ----
        mu_r = Xr.mean(axis=0, dtype=np.float64)
        Xc = Xr - mu_r
        # economy SVD; for (3721 x 50) it’s fast
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        if Vt.shape[0] < 2:
            # not enough PCs – fallback
            self.random_weights_init(data[finite_rows])
            return
        pc1_r = Vt[0, :]                       # in reduced feature space
        pc2_r = Vt[1, :]

        # ---- 3) expand back to full feature space ----
        D = data.shape[1]
        mu = np.zeros((D,), dtype=np.float64)
        pc1 = np.zeros((D,), dtype=np.float64)
        pc2 = np.zeros((D,), dtype=np.float64)
        mu[keep]  = mu_r
        pc1[keep] = pc1_r
        pc2[keep] = pc2_r

        # ---- 4) lay weights on the PC plane: mean + c1*pc1 + c2*pc2 ----
        mx, my = self._weights.shape[:2]
        gx = np.linspace(-1.0, 1.0, mx)
        gy = np.linspace(-1.0, 1.0, my)
        # write in the dtype MiniSom uses
        W = self._weights
        for i, c1 in enumerate(gx):
            for j, c2 in enumerate(gy):
                W[i, j, :] = (mu + c1 * pc1 + c2 * pc2).astype(W.dtype, copy=False)


@dataclass
class SomResult:
    """Container with the outputs of a trained Self-Organising Map."""

    map_shape: Tuple[int, int]
    component_planes: Dict[str, pd.DataFrame]
    feature_positions: pd.DataFrame
    row_bmus: pd.DataFrame
    bmu_counts: pd.DataFrame
    distance_map: Optional[pd.DataFrame]
    activation_response: Optional[pd.DataFrame]
    quantization_map: Optional[pd.DataFrame]
    correlations: pd.DataFrame
    quantization_error: float
    topographic_error: float
    normalized_dataframe: pd.DataFrame
    scaler: Dict[str, Dict[str, float]]
    # The underlying MiniSom object. Frontend components may leverage
    # the trained instance for additional visualisations if desired.
    som_object: Any = None


@dataclass
class ClusteringInputs:
    """Prepared state required for clustering operations."""

    codebook: np.ndarray
    map_shape: Tuple[int, int]
    bmu_indices: Optional[np.ndarray]
    feature_names: List[str]
class SOMService:
    """Utility class to train and inspect Self-Organising Maps on pandas data."""

    def __init__(self, clustering_service: Optional[ClusteringService] = None):
        self._som: Optional[Any] = None
        self._clustering = clustering_service or ClusteringService()
        self._last_norm_df: Optional[pd.DataFrame] = None

    def load_from_state(
        self,
        *,
        weights: np.ndarray,
        sigma: float,
        learning_rate: float,
        neighborhood_function: str = "gaussian",
        random_seed: Optional[int] = None,
        normalized_dataframe: Optional[pd.DataFrame] = None,
    ) -> Any:
        if weights is None or weights.ndim != 3:
            raise ValueError("SOM weights must be a 3D array")
        width, height, input_len = weights.shape
        som = MiniSomWithCallback(
            int(width),
            int(height),
            int(input_len),
            sigma=float(sigma),
            learning_rate=float(learning_rate),
            neighborhood_function=neighborhood_function,
            random_seed=random_seed,
        )
        som._weights = weights.astype(float, copy=False)
        self._som = som
        self._last_norm_df = None if normalized_dataframe is None else normalized_dataframe.copy()
        return som

    @property
    def som(self) -> Optional[Any]:
        return self._som

    def _get_codebook(self) -> np.ndarray:
        """Return codebook as (n_units, n_features)."""
        if self._som is None:
            raise RuntimeError("SOM not trained.")
        W = self._som.get_weights()               # (width, height, features)
        codebook = W.reshape(-1, W.shape[2])      # (n_units, n_features)
        return codebook

    def _bmus_for_rows(self, data_array: np.ndarray) -> np.ndarray:
        """Return BMU indices (flattened unit index) for each row in data_array."""
        W = self._som.get_weights()               # (w,h,f)
        w, h, f = W.shape
        flat_W = W.reshape(-1, f)                 # (n_units, f)
        # compute BMUs
        dists = ((data_array[:, None, :] - flat_W[None, :, :])**2).sum(axis=2)  # (n_samples, n_units)
        return np.argmin(dists, axis=1)           # (n_samples,)

    def clustering_inputs(self) -> ClusteringInputs:
        """Return pre-computed state required for clustering operations."""

        if self._som is None:
            raise RuntimeError("SOM not trained.")

        codebook = self._get_codebook()
        weights = self._som.get_weights()
        map_shape = (int(weights.shape[0]), int(weights.shape[1]))

        if self._last_norm_df is not None:
            X = self._last_norm_df.to_numpy(dtype=float)
            if X.size:
                bmu_indices = self._bmus_for_rows(X)
            else:
                bmu_indices = np.array([], dtype=int)
            cols = list(self._last_norm_df.columns)
            if len(cols) == codebook.shape[1]:
                feature_names = list(cols)
            else:
                feature_names = [f"f{i}" for i in range(codebook.shape[1])]
        else:
            bmu_indices = None
            feature_names = [f"f{i}" for i in range(codebook.shape[1])]

        return ClusteringInputs(
            codebook=codebook,
            map_shape=map_shape,
            bmu_indices=bmu_indices,
            feature_names=feature_names,
        )

    def available_clustering_methods(self) -> List[ClusteringMethodSpec]:
        """Expose clustering methods available from the clustering service."""

        return self._clustering.available_methods()

    # ------------------------------------------------------------------
    @staticmethod
    def _prepare_dataframe(df: pd.DataFrame, *, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
        if df is None:
            raise ValueError("DataFrame is required")
        if columns:
            missing = [c for c in columns if c not in df.columns]
            if missing:
                raise KeyError(f"Columns not found in dataframe: {missing}")
            df = df.loc[:, list(columns)]

        if df.empty:
            raise ValueError("DataFrame has no rows")

        numeric_df = df.select_dtypes(include=["number", "float", "int", "int64", "float64"]).copy()
        if numeric_df.empty:
            raise ValueError("DataFrame does not contain numeric columns")

        numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)
        numeric_df = numeric_df.dropna(how="any")
        if numeric_df.empty:
            raise ValueError("No rows remain after dropping NaNs")
        return numeric_df

    @staticmethod
    def _normalise(df: pd.DataFrame, method: str) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
        method = (method or "none").lower()
        info: Dict[str, Dict[str, float]] = {}
        if method == "zscore":
            means = df.mean(axis=0)
            stds = df.std(axis=0)
            stds_replaced = stds.replace(0, 1.0)
            norm = (df - means) / stds_replaced
            for col in df.columns:
                info[col] = {
                    "method": "zscore",
                    "center": float(means[col]),
                    "scale": float(stds_replaced[col]),
                }
            return norm, info
        if method == "minmax":
            mins = df.min(axis=0)
            maxs = df.max(axis=0)
            ranges = (maxs - mins).replace(0, 1.0)
            norm = (df - mins) / ranges
            for col in df.columns:
                info[col] = {
                    "method": "minmax",
                    "min": float(mins[col]),
                    "max": float(maxs[col]),
                    "range": float(ranges[col]),
                }
            return norm, info
        # none -> copy as float
        return df.astype(float), {c: {"method": "none"} for c in df.columns}

    @staticmethod
    def _inverse_scale(values: np.ndarray, info: Dict[str, float]) -> np.ndarray:
        method = info.get("method", "none")
        if method == "zscore":
            return (values * info.get("scale", 1.0)) + info.get("center", 0.0)
        if method == "minmax":
            return (values * info.get("range", 1.0)) + info.get("min", 0.0)
        return values

    # ------------------------------------------------------------------
    def train(
        self,
        df: pd.DataFrame,
        *,
        columns: Optional[Sequence[str]] = None,
        map_shape: Tuple[int, int] = (10, 10),
        sigma: float = 6.0,
        learning_rate: float = 0.5,
        num_epochs: Optional[int] = None,
        neighborhood_function: str = "gaussian",
        normalisation: str = "zscore",
        random_seed: Optional[int] = None,
        training_mode: str = "batch",
        # --- PURE BACKEND CALLBACKS (optional) ---
        progress_callback: Optional[Callable[[int], None]] = None,  # 0..100
        status_callback: Optional[Callable[[str], None]] = None,    # free text
        stop_event: Optional[threading.Event] = None,
        init_mode = "pca"  # "pca" | "data" | "random"
    ) -> SomResult:
        """
        Train a SOM on the provided dataframe and return structured outputs.

        Parameters
        ----------
        df: DataFrame with observations in rows and features in columns.
        columns: Optional subset of column names to use.
        map_shape: Tuple[int, int] describing the SOM grid (width, height).
        sigma: Initial neighbourhood radius passed to MiniSom.
        learning_rate: Learning rate used by MiniSom for random training.
        num_epochs: Training epochs; defaults to 100.
        neighborhood_function: MiniSom neighbourhood function (e.g. 'gaussian').
        normalisation: one of {'zscore', 'minmax', 'none'}.
        random_seed: Optional seed for MiniSom weight initialisation.
        training_mode: Either 'batch' (default) or 'random'.
        """

        # light wrappers (remain backend-only)
        def _emit_progress(pct: int) -> None:
            if progress_callback:
                try:
                    progress_callback(int(max(0, min(100, pct))))
                except Exception:
                    logger.warning("Exception in _emit_progress", exc_info=True)

        def _emit_status(msg: str) -> None:
            if status_callback:
                try:
                    status_callback(str(msg))
                except Exception:
                    logger.warning("Exception in _emit_status", exc_info=True)

        _emit_status("Preparing data…")

        try:
            data = self._prepare_dataframe(df, columns=columns)

            _emit_status("Normalising features…")
            norm_df, scaler = self._normalise(data, normalisation)

            #keep copy so easier to cluster per row
            self._last_norm_df = norm_df.copy()

            #default to 100 epochs if not specified
            if num_epochs is None or num_epochs <= 0:
                num_epochs = 100

            width, height = map_shape
            if width <= 0 or height <= 0:
                raise ValueError("map_shape dimensions must be positive")

            data_array = norm_df.to_numpy(dtype=float)
            input_len = data_array.shape[1]

            _emit_status("Configuring SOM grid…")
            som = MiniSomWithCallback(
                int(width),
                int(height),
                input_len,
                sigma=sigma,
                learning_rate=learning_rate,
                neighborhood_function=neighborhood_function,
                random_seed=42,
            )

            _emit_status("Initializing weights…")
            # ---- Weight init on the SAME normalized data ----
            mode = (init_mode or "pca").lower()
            if mode == "pca":
                # If user requested no normalization, PCA init is ill-conditioned:
                if normalisation.lower() in ("none", "no", "off"):
                    _emit_status("Warning: PCA init with unnormalized data. Consider normalisation='zscore'.")
                som.pca_weights_init(data_array)
            elif mode in ("data", "random_sample"):
                som.random_weights_init(data_array)
            else:
                logger.warning("Unknown init mode '%s', using default MiniSom init", mode)

            # pick mode
            mode = (training_mode or "batch").lower()
            if mode not in {"batch", "random"}:
                mode = "batch"

            # --- compute callback cadence (no UI calls here) ---
            updates_per_epoch = len(data_array)
            total_updates = max(1, updates_per_epoch * int(num_epochs))
            callback_every = max(1, total_updates // 100)  # ~100 ticks

            last_pct = -1
            training_status_steps = [
                (0, "Initialising SOM training…"),
                (10, "Training SOM…"),
                (55, "Refining neuron organisation…"),
                (85, "Finalising training…"),
            ]
            status_stage_index = 0

            def _som_cb(progress: float, t: int, iteration: int, decay_rate: int, som_obj, qe: Optional[float]):
                # cooperative cancel
                if stop_event is not None and stop_event.is_set():
                    raise RuntimeError("Training cancelled")

                pct = int(progress * 100.0)
                nonlocal last_pct, status_stage_index
                if pct != last_pct:
                    _emit_progress(pct)
                    last_pct = pct

                    while (
                        status_stage_index + 1 < len(training_status_steps)
                        and pct >= training_status_steps[status_stage_index + 1][0]
                    ):
                        status_stage_index += 1
                        _emit_status(training_status_steps[status_stage_index][1])

            # initial tick (pure callback)
            _emit_status(training_status_steps[0][1])
            _emit_progress(0)

            # ---- TRAIN with callback ----
            tm = (training_mode or "batch").lower()
            if tm == "batch":
                # true batch, with epoch count = num_epochs
                som.train(
                    data_array,
                    int(num_epochs),
                    use_epochs=True,            # epochs semantics
                    verbose=False,
                    mode="batch",
                    callback=_som_cb,           # will tick once per epoch
                    callback_every=1,
                    report_qe=False,
                    sigma_schedule=(sigma, 1.0),  # similar to MATLAB som_batchtrain
                )
            else:
                # online/random, as before (but now explicit)
                som.train(
                    data_array,
                    int(num_epochs),
                    random_order=(tm == "random"),
                    use_epochs=True,
                    verbose=False,
                    mode="online",
                    callback=_som_cb,
                    callback_every=max(1, (len(data_array) * int(num_epochs)) // 100),
                    report_qe=False,
                )

            self._som = som

            _emit_status("Calculating component planes…")

            weights = som.get_weights()  # shape (width, height, features)
            width = weights.shape[0]
            height = weights.shape[1]

            component_planes: Dict[str, pd.DataFrame] = {}
            for idx, column in enumerate(norm_df.columns):
                plane = weights[:, :, idx].T  # rows -> x, columns -> y
                restored = self._inverse_scale(plane, scaler[column]) if scaler else plane
                component_planes[column] = pd.DataFrame(
                    restored,
                    index=pd.RangeIndex(start=0, stop=height, step=1, name="x"),
                    columns=pd.RangeIndex(start=0, stop=width, step=1, name="y"),
                )

            _emit_status("Locating feature positions…")
            feature_rows = []
            for column, plane_df in component_planes.items():
                if plane_df.empty:
                    continue
                plane_values = plane_df.to_numpy(dtype=float)
                flat_index = int(np.nanargmax(np.abs(plane_values)))
                row_idx, col_idx = np.unravel_index(flat_index, plane_values.shape)
                feature_rows.append(
                    {
                        "feature": column,
                        "x": int(row_idx),
                        "y": int(col_idx),
                        "max_value": float(plane_values[row_idx, col_idx]),
                        "mean_value": float(np.nanmean(plane_values)),
                    }
                )

            feature_positions = pd.DataFrame(feature_rows)

            _emit_status("Assigning rows to best matching units…")
            row_records = []
            distance_accumulator = np.zeros((height, width), dtype=float)
            hit_counts = np.zeros((height, width), dtype=float)
            for step, (idx, vector) in enumerate(zip(norm_df.index, data_array)):
                winner = som.winner(vector)
                x_idx, y_idx = int(winner[0]), int(winner[1])
                weight_vector = weights[x_idx, y_idx]
                row_idx = int(y_idx)
                col_idx = int(x_idx)
                distance = float(np.linalg.norm(vector - weight_vector))
                row_records.append(
                    {
                        "index": str(idx),
                        "step": int(step),
                        "bmu_x": row_idx,
                        "bmu_y": col_idx,
                        "distance": distance,
                    }
                )
                hit_counts[row_idx, col_idx] += 1
                distance_accumulator[row_idx, col_idx] += distance

            row_bmus = pd.DataFrame(row_records, columns=["index", "step", "bmu_x", "bmu_y", "distance"])

            if row_bmus.empty:
                bmu_counts = pd.DataFrame(columns=["bmu_x", "bmu_y", "count"])
            else:
                bmu_counts = (
                    row_bmus.groupby(["bmu_x", "bmu_y"], as_index=False)["index"]
                    .count()
                    .rename(columns={"index": "count"})
                    .sort_values("count", ascending=False)
                    .reset_index(drop=True)
                )

            _emit_status("Calculating quantisation map…")
            with np.errstate(divide="ignore", invalid="ignore"):
                qe_matrix = np.divide(
                    distance_accumulator,
                    hit_counts,
                    out=np.zeros_like(distance_accumulator),
                    where=hit_counts > 0,
                )

            quantization_map = pd.DataFrame(
                qe_matrix,
                index=pd.RangeIndex(start=0, stop=height, step=1, name="x"),
                columns=pd.RangeIndex(start=0, stop=width, step=1, name="y"),
            )

            _emit_status("Building activation response…")
            activation_counts = pd.DataFrame(
                hit_counts,
                index=pd.RangeIndex(start=0, stop=height, step=1, name="x"),
                columns=pd.RangeIndex(start=0, stop=width, step=1, name="y"),
            )

            _emit_status("Computing distance map…")
            distance_map_array = self._compute_umatrix(weights)
            distance_map = pd.DataFrame(
                distance_map_array,
                index=pd.RangeIndex(start=0, stop=height, step=1, name="x"),
                columns=pd.RangeIndex(start=0, stop=width, step=1, name="y"),
            )

            _emit_status("Deriving correlations and errors…")
            correlations = data.corr().fillna(0.0)

            qe = float(som.quantization_error(data_array)) if len(data_array) else float("nan")
            topo = float(self._topographic_error(som, data_array))

            _emit_progress(100)

            result = SomResult(
                map_shape=(height, width),
                component_planes=component_planes,
                feature_positions=feature_positions,
                row_bmus=row_bmus,
                bmu_counts=bmu_counts,
                distance_map=distance_map,
                activation_response=activation_counts,
                quantization_map=quantization_map,
                correlations=correlations,
                quantization_error=qe,
                topographic_error=topo,
                normalized_dataframe=norm_df,
                scaler=scaler,
                som_object=som,
            )
        except Exception:
            _emit_status("SOM ready (last run failed)")
            raise
        else:
            _emit_status("SOM ready")
            return result
    
    def cluster_neurons(
        self,
        *,
        n_clusters: Optional[int] = None,
        k_list_max: int = 16,
        random_state: int = 42,
        scoring: ScoreType = "silhouette",
        progress_callback: Optional[Callable[[int], None]] = None,
        method: str = "kmeans",
        method_params: Optional[Mapping[str, object]] = None,
    ) -> NeuronClusteringResult:
        """Cluster SOM neurons via the shared clustering service."""
        inputs = self.clustering_inputs()

        return self._clustering.cluster_neurons(
            codebook=inputs.codebook,
            map_shape=inputs.map_shape,
            n_clusters=n_clusters,
            k_list_max=k_list_max,
            random_state=random_state,
            scoring=scoring,
            bmu_indices=inputs.bmu_indices,
            progress_callback=progress_callback,
            method=method,
            method_params=method_params,
        )

    def cluster_features(
        self,
        *,
        n_clusters: Optional[int] = None,
        k_list_max: Optional[int] = None,
        random_state: int = 0,
        scoring: ScoreType = "silhouette",
        progress_callback: Optional[Callable[[int], None]] = None,
        method: str = "kmeans",
        method_params: Optional[Mapping[str, object]] = None,
    ) -> FeatureClusteringResult:
        """
        Groups features by similarity of their component-plane patterns.
        If n_clusters is None, chooses K by 'scoring' over K in [2..k_list_max].
        """
        clustering_state = self.clustering_inputs()
        codebook = clustering_state.codebook
        feature_names = clustering_state.feature_names
        return self._clustering.cluster_features(
            codebook=codebook,
            feature_names=feature_names,
            n_clusters=n_clusters,
            k_list_max=k_list_max,
            random_state=random_state,
            scoring=scoring,
            progress_callback=progress_callback,
            method=method,
            method_params=method_params,
        )

    @staticmethod
    def _compute_umatrix(weights: np.ndarray) -> np.ndarray:
        width, height, _ = weights.shape
        umat = np.zeros((height, width), dtype=float)
        for x in range(width):
            for y in range(height):
                center = weights[x, y]
                neighbors = []
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        neighbors.append(weights[nx, ny])
                if neighbors:
                    distances = [float(np.linalg.norm(center - nbr)) for nbr in neighbors]
                    umat[y, x] = float(np.mean(distances))
                else:
                    umat[y, x] = 0.0
        return umat

    @staticmethod
    def _topographic_error(som: MiniSomWithCallback, data: np.ndarray) -> float:
        if data.size == 0:
            return float("nan")

        weights = som.get_weights()
        if weights is None:
            return float("nan")

        errors = 0
        grid_shape = (weights.shape[0], weights.shape[1])
        for vector in data:
            distances = np.linalg.norm(weights - vector, axis=2)
            flat = distances.ravel()
            if flat.size < 2:
                continue
            best = int(np.argmin(flat))
            masked = flat.copy()
            masked[best] = np.inf
            second = int(np.argmin(masked))
            x1, y1 = np.unravel_index(best, grid_shape)
            x2, y2 = np.unravel_index(second, grid_shape)
            if abs(x1 - x2) + abs(y1 - y2) > 1:
                errors += 1
        return errors / data.shape[0]
