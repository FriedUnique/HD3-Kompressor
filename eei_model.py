import numpy as np
import pandas as pd

from config import ScopeConfig


def ew(numerator: pd.Series, weight: pd.Series) -> float:
    """Energy-weighted mean."""
    total_weight = weight.sum()
    return float((numerator * weight).sum() / total_weight) if total_weight else float("nan")


def eei_load_curve(baseline_df: pd.DataFrame, cfg: ScopeConfig):
    """Generates an energy-weighted efficiency curve vs. cooling load with load bins
    (binning + linear interpolation method, kept for comparison).
    """
    df_copy = baseline_df.copy()
    try:
        df_copy["q_bin"] = pd.qcut(df_copy["Q"], cfg.n_load_bins, duplicates="drop")
    except ValueError:
        df_copy["q_bin"] = 0

    bin_results = []
    for _, bin_group in df_copy.groupby("q_bin", observed=True):
        q_center = ew(bin_group["Q"], bin_group["E"])
        eei_center = ew(bin_group["EEI"], bin_group["E"])
        bin_results.append((q_center, eei_center))
    bin_results.sort()

    q_centers = np.array([res[0] for res in bin_results])
    eei_values = np.array([res[1] for res in bin_results])

    return q_centers, eei_values








def eei_load_regression(baseline_df: pd.DataFrame, cfg: ScopeConfig):
    """Fits a weighted polynomial regression of EEI on load (Q), weighted by
    energy (E) with least squares.
    """
    Q = baseline_df["Q"].values         # independent variable
    EEI = baseline_df["EEI"].values     # dependent variable
    W = baseline_df["E"].values         # weight

    degree = min(cfg.regression_degree, max(len(np.unique(Q)) - 1, 0))

    if degree < 1:
        return np.array([ew(baseline_df["EEI"], baseline_df["E"])])

    X = np.vander(Q, degree + 1, increasing=True)  # columns: [1, Q, Q^2, ...] -> polynomial c0 + c1*Q + c2*Q^2 + ...
    sqrt_w = np.sqrt(np.clip(W, 0, None))
    Xw = X * sqrt_w[:, None]
    yw = EEI * sqrt_w

    coeffs, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)
    return coeffs












def eei_regression_predict(coeffs: np.ndarray, Q: np.ndarray, q_min: float = None, q_max: float = None) -> np.ndarray:
    """Prediction with regression: EEI_matrix = X_matrix * coefficients_vector."""
    Q = np.asarray(Q, dtype=float)
    if q_min is not None or q_max is not None:
        Q = np.clip(Q, 
                    q_min if q_min is not None else -np.inf, 
                    q_max if q_max is not None else np.inf)
        
    X = np.vander(Q, len(coeffs), increasing=True)
    return X @ coeffs


def cop_norm(eval_df: pd.DataFrame, eei_series: pd.Series) -> float:
    """Aggregate baseline-equivalent COP for a EEI series."""
    cop_baseline_normalized = eei_series.values * eval_df["cop_carnot"].values
    total_q = eval_df["Q"].sum()
    total_equivalent_e = (eval_df["Q"].values / cop_baseline_normalized).sum()
    return total_q / total_equivalent_e
