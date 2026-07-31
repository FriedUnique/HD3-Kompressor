from forecast import build_usage_forecast
from main import build_config

import matplotlib.pyplot as plt
from config import ScopeConfig
import pandas as pd

# --- run settings -----------------------------------------------------
METHOD = "regression"   # "regression" or "bin"
SAVE_PATH = None        # e.g. "usage_forecast.png", or None to skip saving
# ------------------------------------------------------------------------


def main():
    cfg = build_config()
    forecast_df = build_usage_forecast(cfg, method=METHOD)
    plot_usage_forecast(forecast_df, cfg, show=True)
    
def plot_usage_forecast(forecast_df: pd.DataFrame, cfg: ScopeConfig, save_path: str = None, show: bool = True):
    """Plots historical (baseline) usage, actual evaluation usage, and forecasted
    (baseline-equivalent) evaluation usage on one time axis.
    """
    baseline = forecast_df[forecast_df["phase"] == "baseline"]
    evaluation = forecast_df[forecast_df["phase"] == "evaluation"]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(baseline["ts"], baseline["E_actual"], color="tab:blue", linewidth=2, label="Historical Training Data")
    ax.plot(evaluation["ts"], evaluation["E_actual"], color="green", linewidth=2, label="Actual Usage")
    ax.plot(evaluation["ts"], evaluation["E_forecast"], color="red", linewidth=2, linestyle="--", label="Forcasted Usage")

    ax.set_title(f"Hourly Electricity Usage & Forecast :: {cfg.scope_prefix}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Hourly Electricity Usage (kW)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()

    return fig



if __name__ == "__main__":
    main()
