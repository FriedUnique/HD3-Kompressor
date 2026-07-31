from dataclasses import dataclass
from typing import Tuple


@dataclass
class ScopeConfig:
    csv_path: str
    scope_prefix: str                       # "S1_B1_CS1_COMPHIGH3"
    baseline: Tuple[str, str]               # [start, end)
    evaluation: Tuple[str, str]             # [start, end)

    ts_col: str = "ts"
    q_suffix: str = "COOLPOWER"
    e_suffix: str = "EP_IN"
    cop_suffix: str = "COOLCOP"
    freq_suffix: str = "FREQ"
    tcond_suffix: str = "T_COND"
    tevap_suffix: str = "T_EVAP"

    tz: str = "Europe/Amsterdam"

    # load / steady-state filter (transients give nonsense COP; exclude them)
    freq_min: float = 10.0
    cop_min: float = 1.5
    cop_max: float = 8.0
    lift_min: float = 20.0
    lift_max: float = 60.0
    n_load_bins: int = 100                  # bins for the EEI(load) curve (binning method)
    regression_degree: int = 4              # polynomial degree for the EEI(load) regression method

    @property
    def cols(self) -> dict:
        return {
            "ts": self.ts_col,
            "Q": f"{self.scope_prefix}_{self.q_suffix}",
            "E": f"{self.scope_prefix}_{self.e_suffix}",
            "COP": f"{self.scope_prefix}_{self.cop_suffix}",
            "FREQ": f"{self.scope_prefix}_{self.freq_suffix}",
            "TCOND": f"{self.scope_prefix}_{self.tcond_suffix}",
            "TEVAP": f"{self.scope_prefix}_{self.tevap_suffix}",
            "CAP": f"{self.scope_prefix}_CAP",
        }
