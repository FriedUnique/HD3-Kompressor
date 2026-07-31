from typing import Tuple

import numpy as np
import pandas as pd

from config import ScopeConfig

KELVIN = 273.15


def load_data(cfg: ScopeConfig) -> pd.DataFrame:
    cols = cfg.cols
    wanted_columns = list(dict.fromkeys(cols.values()))
    df = pd.read_csv(cfg.csv_path, usecols=lambda col: col in wanted_columns)

    missing_cols = [name for key, name in cols.items()
                     if key in ("ts", "Q", "E", "TCOND", "TEVAP") and name not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns for '{cfg.scope_prefix}': {missing_cols}")

    df = df.rename(columns={val: key for key, val in cols.items() if val in df.columns})
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(cfg.tz)
    df["lift"] = df["TCOND"] - df["TEVAP"]


    # for steady state
    if "COP" not in df.columns:
        df["COP"] = df["Q"] / df["E"].replace(0, np.nan)
    if "FREQ" not in df.columns:
        df["FREQ"] = np.where(df["Q"] > 0, cfg.freq_min + 1.0, 0.0)

    return df


def sample_hours(df: pd.DataFrame) -> float:
    delta_seconds = df["ts"].diff().dt.total_seconds().median()
    return (delta_seconds if delta_seconds and delta_seconds > 0 else 60.0) / 3600.0


def window(df: pd.DataFrame, cfg: ScopeConfig, window_range: Tuple[str, str]) -> pd.DataFrame:
    start_ts = pd.Timestamp(window_range[0], tz=cfg.tz)
    end_ts = pd.Timestamp(window_range[1], tz=cfg.tz)
    return df[(df["ts"] >= start_ts) & (df["ts"] < end_ts)]


def steady_state(df: pd.DataFrame, cfg: ScopeConfig) -> pd.DataFrame:
    df_filter = ((df["FREQ"] > cfg.freq_min) & (df["Q"] > 0) & (df["E"] > 0)
                 & (df["COP"].between(cfg.cop_min, cfg.cop_max))
                 & (df["lift"].between(cfg.lift_min, cfg.lift_max)))
    return df[df_filter].copy()
