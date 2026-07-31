import numpy as np
import pandas as pd

from config import ScopeConfig
from data_io import KELVIN, load_data, sample_hours, steady_state, window
from eei_model import cop_norm, ew, eei_load_curve, eei_load_regression, eei_regression_predict
from metrics import ashrae_metrics, standard_error, weighted_r2


def compute_savings(cfg: ScopeConfig) -> dict:
    df = load_data(cfg)
    sample_duration_hours = sample_hours(df)
    base_df = steady_state(window(df, cfg, cfg.baseline), cfg)
    eval_df = steady_state(window(df, cfg, cfg.evaluation), cfg)

    for period_df in (base_df, eval_df):
        period_df["cop_carnot"] = (period_df["TEVAP"] + KELVIN) / period_df["lift"]
        period_df["EEI"] = period_df["COP"] / period_df["cop_carnot"]

    cop_base_raw = base_df["Q"].sum() / base_df["E"].sum()
    cop_eval = eval_df["Q"].sum() / eval_df["E"].sum()
    EEI_base = ew(base_df["EEI"], base_df["E"])

    # (1) lift-only: constant baseline EEI at each after-samples Carnot limit
    cop_norm_lift = cop_norm(eval_df, pd.Series(EEI_base, index=eval_df.index))



    # (2) lift + load, BINNING method: baseline EEI as function of load, via bins + interpolation
    q_centers, EEI_curve = eei_load_curve(base_df, cfg)
    EEI_load_bin = pd.Series(np.interp(eval_df["Q"].values, q_centers, EEI_curve), index=eval_df.index)
    cop_norm_loadlift_bin = cop_norm(eval_df, EEI_load_bin)

    EEI_base_bin_pred = np.interp(base_df["Q"].values, q_centers, EEI_curve)
    EEI_bin_se = standard_error(base_df["EEI"].values, EEI_base_bin_pred)
    EEI_bin_r2 = weighted_r2(base_df["EEI"].values, EEI_base_bin_pred, base_df["E"].values)




    # (3) lift + load, REGRESSION method: weighted polynomial fit of EEI vs load
    q_base_min = float(base_df["Q"].min())
    q_base_max = float(base_df["Q"].max())
    reg_coeffs = eei_load_regression(base_df, cfg)
    EEI_load_reg = pd.Series(
        eei_regression_predict(reg_coeffs, eval_df["Q"].values, q_base_min, q_base_max),
        index=eval_df.index,
    )
    cop_norm_loadlift_reg = cop_norm(eval_df, EEI_load_reg)

    EEI_base_reg_pred = eei_regression_predict(reg_coeffs, base_df["Q"].values, q_base_min, q_base_max)
    EEI_reg_se = standard_error(base_df["EEI"].values, EEI_base_reg_pred)
    EEI_reg_r2 = weighted_r2(base_df["EEI"].values, EEI_base_reg_pred, base_df["E"].values)






    q_eval = eval_df["Q"].sum() * sample_duration_hours
    e_actual = eval_df["E"].sum() * sample_duration_hours
    pct_clipped = float((eval_df["Q"] > q_centers.max()).mean() * 100)


    bin_metrics = ashrae_metrics(base_df["EEI"].values, EEI_base_bin_pred, polynomial_degree=1)
    reg_metrics = ashrae_metrics(base_df["EEI"].values, EEI_base_reg_pred, polynomial_degree=len(reg_coeffs))

    print(bin_metrics)

    return dict(
        scope=cfg.scope_prefix,
        baseline=cfg.baseline,
        evaluation=cfg.evaluation,

        n_base=len(base_df), n_eval=len(eval_df),
        run_h=len(base_df) * sample_duration_hours,

        cop_base_raw=cop_base_raw,
        cop_norm_lift=cop_norm_lift,
        cop_norm_loadlift_bin=cop_norm_loadlift_bin,
        cop_norm_loadlift_reg=cop_norm_loadlift_reg,
        cop_eval=cop_eval,

        lift_base=float(base_df["lift"].mean()),
        lift_eval=float(eval_df["lift"].mean()),
        load_base=float(base_df["Q"].mean()),
        load_eval=float(eval_df["Q"].mean()),

        EEI_base=EEI_base,
        EEI_bin_se=EEI_bin_se,
        EEI_bin_r2=EEI_bin_r2,

        EEI_reg_se=EEI_reg_se,
        EEI_reg_r2=EEI_reg_r2,

        bin_NMBE = bin_metrics["NMBE"], bin_CV_RMSE = bin_metrics["CV_RMSE"],
        reg_NMBE = reg_metrics["NMBE"], reg_CV_RMSE = reg_metrics["CV_RMSE"],


        reg_degree=len(reg_coeffs) - 1,

        q_eval=q_eval,
        e_actual=e_actual,
        pct_load_clipped=pct_clipped,

        saved_raw=q_eval / cop_base_raw - e_actual,
        saved_norm_lift=q_eval / cop_norm_lift - e_actual,
        saved_norm_loadlift_bin=q_eval / cop_norm_loadlift_bin - e_actual,
        saved_norm_loadlift_reg=q_eval / cop_norm_loadlift_reg - e_actual,

        EEI_load_curve=list(zip(q_centers.tolist(), EEI_curve.tolist())),
        EEI_regression_coeffs=reg_coeffs.tolist(),
    )


def summary(results: dict) -> str:
    return f"""================ SAVINGS :: {results['scope']} ================
baseline {results['baseline'][0]} -> {results['baseline'][1]}   eval {results['evaluation'][0]} -> {results['evaluation'][1]}
running steady samples : baseline n={results['n_base']}, eval n={results['n_eval']}

Prediction Quality:
  bins + interpolation\t\t\tRMSE={results['EEI_bin_se']:.4f}\tR^2={results['EEI_bin_r2']:.3f}\tNMBE={results['bin_NMBE']:.2f}%\tCV(RMSE)={results['bin_CV_RMSE']:.2f}%
  polynomial regression(d={results['reg_degree']})\t\tRMSE={results['EEI_reg_se']:.4f}\tR^2={results['EEI_reg_r2']:.3f}\tNMBE={results['reg_NMBE']:.2f}%\tCV(RMSE)={results['reg_CV_RMSE']:.2f}%


pct outside base range (binning):  {results['pct_load_clipped']:,.0f} %

delivered cooling eval : {results['q_eval']:,.0f} kWh_th    actual elec : {results['e_actual']:,.0f} kWh

 method                                     COP_base        ENERGY SAVED
 raw                                        {results['cop_base_raw']:7.3f}\t{results['saved_raw']:>11,.0f} kWh
 normalized (lift only)                     {results['cop_norm_lift']:7.3f}\t{results['saved_norm_lift']:>11,.0f} kWh
 normalized (lift+load, bins)               {results['cop_norm_loadlift_bin']:7.3f}\t{results['saved_norm_loadlift_bin']:>11,.0f} kWh
 normalized (lift+load, regression)         {results['cop_norm_loadlift_reg']:7.3f}\t{results['saved_norm_loadlift_reg']:>11,.0f} kWh
 COP_eval = {results['cop_eval']:.3f};
========================================================="""
