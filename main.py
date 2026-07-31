from config import ScopeConfig
from savings import compute_savings, summary

# --- run settings -----------------------------------------------------
CSV_PATH = "./data/test.csv"
SCOPE_PREFIX = "S1_B1_CS1_COMPHIGH3"
BASELINE = ("2026-05-01", "2026-06-12")     # [start, end)
EVALUATION = ("2026-06-12", "2026-06-30")   # [start, end)
REGRESSION_DEGREE = 4
TIMEZONE = "Europe/Amsterdam"
# ------------------------------------------------------------------------


def build_config() -> ScopeConfig:
    return ScopeConfig(
        csv_path=CSV_PATH,
        scope_prefix=SCOPE_PREFIX,
        baseline=BASELINE,
        evaluation=EVALUATION,
        tz=TIMEZONE,
        regression_degree=REGRESSION_DEGREE,
    )


def main():
    cfg = build_config()
    results = compute_savings(cfg)
    print()
    print(summary(results))


if __name__ == "__main__":
    main()
