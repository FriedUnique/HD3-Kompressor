import numpy as np
import pandas as pd

from config import ScopeConfig
from data_io import KELVIN, load_data, steady, window
from eei_model import eei_load_curve, eei_load_regression, eei_regression_predict


def build_usage_forecast(cfg: ScopeConfig, method: str = "regression", resample_freq: str = "1h") -> pd.DataFrame:
    """

    """
    df = load_data(cfg)
    base_window = window(df, cfg, cfg.baseline)
    eval_window = window(df, cfg, cfg.evaluation)

    base_steady = steady(base_window, cfg).copy()
    base_steady["cop_carnot"] = (base_steady["TEVAP"] + KELVIN) / base_steady["lift"]
    base_steady["EEI"] = base_steady["COP"] / base_steady["cop_carnot"]
    q_base_min = float(base_steady["Q"].min())
    q_base_max = float(base_steady["Q"].max())

    # Only forecast where the unit is actually operating in steady state - during
    # transients/off periods (Q or lift near zero) the model isn't valid and
    # division blows up, so those points are left as gaps rather than plotted.
    eval_steady = steady(eval_window, cfg).copy()
    eval_steady["cop_carnot"] = (eval_steady["TEVAP"] + KELVIN) / eval_steady["lift"]

    if method == "regression":
        coeffs = eei_load_regression(base_steady, cfg)
        eei_forecast = eei_regression_predict(coeffs, eval_steady["Q"].values, q_base_min, q_base_max)
    elif method == "bin":
        q_centers, eei_curve = eei_load_curve(base_steady, cfg)
        eei_forecast = np.interp(eval_steady["Q"].values, q_centers, eei_curve)
    else:
        raise ValueError(f"Unknown method '{method}', expected 'regression' or 'bin'")


    # cop_norm, without the aggregation
    cop_forecast = eei_forecast * eval_steady["cop_carnot"].values
    e_forecast_steady = pd.Series(
        np.where(cop_forecast > 0, eval_steady["Q"].values / cop_forecast, np.nan),
        index=eval_steady.index,
    )
    e_forecast = e_forecast_steady.reindex(eval_window.index)

    out = pd.DataFrame({
        "ts": pd.concat([base_window["ts"], eval_window["ts"]], ignore_index=True),
        "phase": ["baseline"] * len(base_window) + ["evaluation"] * len(eval_window),
        "E_actual": pd.concat([base_window["E"], eval_window["E"]], ignore_index=True).values,
        "E_forecast": np.concatenate([np.full(len(base_window), np.nan), e_forecast.values]),
    })



    if resample_freq:
        out = (
            out.set_index("ts")
            .resample(resample_freq)
            .agg({"E_actual": "mean", "E_forecast": "mean", "phase": "first"})
            .dropna(subset=["phase"])
            .reset_index()
        )

    # test = out.copy()

    # x = pd.to_datetime("2026-06-12")

    # if test["ts"].dt.tz is not None:
    #     x = x.tz_localize(test["ts"].dt.tz)

    # # Filter test for timestamps strictly greater than x
    # test = test[test["ts"] > x].copy()

    # print((test["E_forecast"] - test["E_actual"]).sum()/1000, "kWh")

    return out
