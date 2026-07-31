import numpy as np


def weighted_r2(y_actual: np.ndarray, y_pred: np.ndarray, weight: np.ndarray) -> float:
    """Energy-weighted R^2 of a fit against actual values."""
    y_actual = np.asarray(y_actual, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    weight = np.asarray(weight, dtype=float)
    y_mean = np.average(y_actual, weights=weight)
    ss_res = np.sum(weight * (y_actual - y_pred) ** 2)
    ss_tot = np.sum(weight * (y_actual - y_mean) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot else float("nan")


def standard_error(f_actual: np.ndarray, f_interp: np.ndarray) -> float:
    """Standard error (RMSE) of an fitted curve against actual values.
    """
    f_actual = np.asarray(f_actual, dtype=float)
    f_interp = np.asarray(f_interp, dtype=float)
    mask = ~(np.isnan(f_actual) | np.isnan(f_interp))
    if mask.sum() == 0:
        return float("nan")
    residuals = f_actual[mask] - f_interp[mask]
    return float(np.sqrt(np.mean(residuals ** 2)))


def ashrae_metrics(y_actual: np.ndarray, y_pred: np.ndarray, polynomial_degree: int = 1) -> dict:
    """Calculates ASHRAE Guideline 14 NMBE and CV(RMSE) percentages.
    -> A model calibrated against monthly utility bills must land within ±5% NMBE and 15% CV(RMSE); against hourly interval data, within ±10% NMBE and 30% CV(RMSE).
    values are in percent (*100)

    CV(RMSE) tells you how big the errors are compared to the predicted values
    
    """
    y_actual = np.asarray(y_actual, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = ~(np.isnan(y_actual) | np.isnan(y_pred))
    y_actual = y_actual[mask]
    y_pred = y_pred[mask]
    n = len(y_actual)

    if n <= polynomial_degree:
        return {"NMBE": float("nan"), "CV_RMSE": float("nan")}

    y_mean = np.mean(y_actual)
    if y_mean == 0:
        return {"NMBE": float("nan"), "CV_RMSE": float("nan")}

    errors = y_pred - y_actual
    nmbe = (np.sum(errors) / (n * y_mean)) * 100.0
    rmse = np.sqrt(np.sum(errors ** 2) / (n - polynomial_degree))
    cv_rmse = (rmse / y_mean) * 100.0

    return {"NMBE": float(nmbe), "CV_RMSE": float(cv_rmse)}
