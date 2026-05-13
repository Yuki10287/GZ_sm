from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor

import task1_price_mechanism as base


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs" / "task1_multisource_statespace"


@dataclass(frozen=True)
class MultiSourceConfig:
    start_date: str = "2016-01-01"
    train_end: str = "2022-12-31"
    window_size: int = 10
    pricing_lag_days: int = 1
    factor_order: int = 1


OBS_SETS = {
    "A_oil_only": [
        "log_brent_ma10",
        "log_wti_ma10",
        "log_dubai_ma10",
    ],
    "B_oil_fx": [
        "log_brent_ma10",
        "log_wti_ma10",
        "log_dubai_ma10",
        "log_usd_cny_ma10",
    ],
    "C_cost_factor": [
        "log_brent_ma10",
        "log_wti_ma10",
        "log_dubai_ma10",
        "log_usd_cny_ma10",
        "log_import_cny_per_ton",
        "log_import_tons",
    ],
    "D_full_factor": [
        "log_brent_ma10",
        "log_wti_ma10",
        "log_dubai_ma10",
        "log_usd_cny_ma10",
        "log_import_cny_per_ton",
        "log_import_tons",
        "cpi_yoy",
        "ppi_yoy",
        "pmi_manufacturing",
        "gdp_yoy",
    ],
}

POLICY_VARS = [
    "cpi_yoy",
    "ppi_yoy",
    "cpi_ppi_gap",
    "pmi_manufacturing",
    "pmi_nonmanufacturing",
    "gdp_yoy",
]


def month_end_from_ym(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series.astype("string") + "-01", errors="coerce") + pd.offsets.MonthEnd(0)


def make_monthly_available(df: pd.DataFrame) -> pd.DataFrame:
    # Without exact release dates, monthly data are conservatively assumed usable
    # only from the first day of the following month.
    out = df.copy()
    out["available_date"] = out["date"] + pd.offsets.Day(1)
    return out.drop(columns=["date"]).sort_values("available_date")


def make_quarterly_available(df: pd.DataFrame) -> pd.DataFrame:
    # GDP uses the most recent completed quarter. Without release dates, a quarter
    # is assumed usable only after the quarter-end date.
    out = df.copy()
    out["available_date"] = out["date"] + pd.offsets.Day(1)
    return out.drop(columns=["date"]).sort_values("available_date")


def read_customs_monthly() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "海关进出口数量数据" / "进口原油数量和金额_合并总表.csv", encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["数据年月"].astype(str) + "01", format="%Y%m%d", errors="coerce") + pd.offsets.MonthEnd(0)
    numeric_cols = ["第一数量", "金额_人民币", "金额_美元", "每吨人民币", "每吨美元"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    out = df[["date", "第一数量", "金额_人民币", "金额_美元", "每吨人民币", "每吨美元"]].copy()
    out = out.rename(
        columns={
            "第一数量": "import_kg",
            "金额_人民币": "import_amount_cny",
            "金额_美元": "import_amount_usd",
            "每吨人民币": "import_cny_per_ton",
            "每吨美元": "import_usd_per_ton",
        }
    )
    out["import_tons"] = out["import_kg"] / 1000.0
    return make_monthly_available(out.sort_values("date"))


def read_macro_monthly() -> pd.DataFrame:
    cpi = pd.read_excel(ROOT / "国内的一些数据" / "2026-2008_cpi.xlsx")
    cpi["date"] = month_end_from_ym(cpi["月份"])
    cpi = cpi.rename(
        columns={
            "全国_当月": "cpi_index",
            "全国_同比增长": "cpi_yoy",
            "全国_环比增长": "cpi_mom",
        }
    )[["date", "cpi_index", "cpi_yoy", "cpi_mom"]]

    ppi = pd.read_excel(ROOT / "国内的一些数据" / "2026-2006_ppi.xlsx")
    ppi["date"] = month_end_from_ym(ppi["月份"])
    ppi = ppi.rename(columns={"PPI_当月": "ppi_index", "PPI_当月同比增长": "ppi_yoy"})[
        ["date", "ppi_index", "ppi_yoy"]
    ]

    pmi = pd.read_excel(ROOT / "国内的一些数据" / "2008-2026_pmi.xlsx")
    pmi["date"] = month_end_from_ym(pmi["月份"])
    pmi = pmi.rename(
        columns={
            "制造业_指数": "pmi_manufacturing",
            "非制造业_指数": "pmi_nonmanufacturing",
            "制造业_同比增长": "pmi_manufacturing_yoy",
        }
    )[["date", "pmi_manufacturing", "pmi_nonmanufacturing", "pmi_manufacturing_yoy"]]

    macro = cpi.merge(ppi, on="date", how="outer").merge(pmi, on="date", how="outer")
    return make_monthly_available(macro.sort_values("date"))


def read_gdp_quarterly() -> pd.DataFrame:
    df = pd.read_excel(ROOT / "国内的一些数据" / "2006-2026_gdp.xlsx")
    year = df["标准季度"].astype(str).str.slice(0, 4).astype(int)
    quarter = df["标准季度"].astype(str).str.extract(r"Q(\d)").iloc[:, 0].astype(int)
    month = quarter * 3
    df["date"] = pd.to_datetime(dict(year=year, month=month, day=1)) + pd.offsets.MonthEnd(0)
    df = df.rename(columns={"GDP_当季绝对值": "gdp_current", "GDP_累计同比": "gdp_yoy"})
    return make_quarterly_available(df[["date", "gdp_current", "gdp_yoy"]].sort_values("date"))


def asof_merge_available(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    return pd.merge_asof(
        left.sort_values("adjust_date"),
        right.sort_values("available_date"),
        left_on="adjust_date",
        right_on="available_date",
        direction="backward",
    ).drop(columns=["available_date"], errors="ignore")


def build_window_dataset(config: MultiSourceConfig) -> pd.DataFrame:
    mechanism_config = base.MechanismConfig(
        start_date=config.start_date,
        train_end=config.train_end,
        window_size=config.window_size,
        pricing_lag_days=config.pricing_lag_days,
    )
    panel = base.read_oil_panel()
    domestic = base.read_domestic_adjustments(config.start_date)
    mechanism = base.simulate_mechanism(domestic, panel, "kalman_index_usd", mechanism_config)

    daily_features = []
    for _, row in mechanism.iterrows():
        pricing_end = pd.Timestamp(row["pricing_end"])
        win = base.trailing_window(panel, pricing_end, "kalman_index_usd", config.window_size)
        raw_win = panel[panel["date"].isin(win["date"])]
        record = {"adjust_date": row["adjust_date"]}
        for col in ["brent", "wti", "dubai", "usd_cny"]:
            record[f"{col}_ma10"] = float(raw_win[col].mean())
        daily_features.append(record)

    features = mechanism.merge(pd.DataFrame(daily_features), on="adjust_date", how="left")
    features = asof_merge_available(features, read_customs_monthly())
    features = asof_merge_available(features, read_macro_monthly())
    features = asof_merge_available(features, read_gdp_quarterly())
    features = features.sort_values("adjust_date").reset_index(drop=True)

    for col in [
        "brent_ma10",
        "wti_ma10",
        "dubai_ma10",
        "usd_cny_ma10",
        "import_cny_per_ton",
        "import_usd_per_ton",
        "import_tons",
        "cpi_index",
        "ppi_index",
        "pmi_manufacturing",
        "pmi_nonmanufacturing",
        "gdp_current",
    ]:
        if col in features:
            values = pd.to_numeric(features[col], errors="coerce").clip(lower=1e-9)
            features[f"log_{col}"] = np.log(values)

    features["import_cny_per_ton_change"] = features["import_cny_per_ton"].pct_change()
    features["import_usd_per_ton_change"] = features["import_usd_per_ton"].pct_change()
    features["cpi_ppi_gap"] = features["cpi_yoy"] - features["ppi_yoy"]
    return features


def extract_loadings(result, obs_cols: list[str], sign: float) -> pd.DataFrame:
    rows = []
    params = dict(zip(result.param_names, np.asarray(result.params, dtype=float)))
    for col in obs_cols:
        value = params.get(f"loading.f1.{col}")
        if value is not None:
            rows.append({"variable": col, "loading": float(value * sign), "abs_loading": float(abs(value))})
    return pd.DataFrame(rows).sort_values("abs_loading", ascending=False)


def fit_dynamic_factor(
    features: pd.DataFrame,
    config: MultiSourceConfig,
    obs_cols: list[str],
    variant_name: str,
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    obs = features[obs_cols].copy()
    obs = obs.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    standardized = pd.DataFrame(
        StandardScaler().fit_transform(obs),
        index=pd.DatetimeIndex(features["adjust_date"]),
        columns=obs_cols,
    )
    model = DynamicFactor(
        standardized,
        k_factors=1,
        factor_order=config.factor_order,
        error_cov_type="diagonal",
    )
    result = model.fit(method="lbfgs", maxiter=1000, disp=False)
    factor = np.asarray(result.factors.smoothed[0], dtype=float)
    sign = 1.0
    reference = standardized[obs_cols[0]].to_numpy(dtype=float)
    if np.corrcoef(factor, reference)[0, 1] < 0:
        sign = -1.0
        factor = -factor

    out = features.copy()
    out["factor_1"] = factor
    out["factor_1_diff"] = out["factor_1"].diff().fillna(0.0)
    loadings = extract_loadings(result, obs_cols, sign)
    fit_info = {
        "variant": variant_name,
        "obs_cols": obs_cols,
        "llf": float(result.llf),
        "aic": float(result.aic),
        "bic": float(result.bic),
        "converged": bool(result.mle_retvals.get("converged", False)),
        "iterations": int(result.mle_retvals.get("iterations", -1)),
    }
    return out, fit_info, loadings


def train_predict_product(
    df: pd.DataFrame,
    product: str,
    config: MultiSourceConfig,
    include_policy_vars: bool,
) -> tuple[pd.DataFrame, dict[str, object]]:
    result = df.copy()
    target = f"{product}_actual_delta"
    base_col = f"{product}_theory_delta"
    feature_cols = [
        base_col,
        "oil_change_rate",
        "factor_1",
        "factor_1_diff",
        "import_cny_per_ton_change",
        "import_usd_per_ton_change",
    ]
    if include_policy_vars:
        feature_cols += POLICY_VARS

    feature_cols = [col for col in feature_cols if col in result.columns]
    result[feature_cols] = result[feature_cols].replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
    train_mask = result["adjust_date"] <= pd.Timestamp(config.train_end)
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 25))),
        ]
    )
    model.fit(result.loc[train_mask, feature_cols], result.loc[train_mask, target])
    pred_col = f"{product}_multisource_delta"
    result[pred_col] = model.predict(result[feature_cols])
    ridge = model.named_steps["ridge"]
    return result, {
        "alpha": float(ridge.alpha_),
        "features": feature_cols,
        "coef": dict(zip(feature_cols, ridge.coef_.astype(float))),
    }


def metric_block(df: pd.DataFrame, product: str, pred_col: str) -> dict[str, float | int | None]:
    block = base.metric_block(df, product, pred_col)
    actual = df[f"{product}_actual_delta"]
    pred = df[pred_col]
    denom = actual.abs().sum()
    block["wmape"] = float((pred - actual).abs().sum() / denom) if denom else None
    return block


def evaluate_variant(
    features: pd.DataFrame,
    config: MultiSourceConfig,
    variant_name: str,
    obs_cols: list[str],
    include_policy_vars: bool,
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    state_df, fit_info, loadings = fit_dynamic_factor(features, config, obs_cols, variant_name)
    state_df, gas_info = train_predict_product(state_df, "gasoline", config, include_policy_vars)
    state_df, diesel_info = train_predict_product(state_df, "diesel", config, include_policy_vars)

    samples = {
        "all": state_df,
        "test_after_train": state_df[state_df["adjust_date"] > pd.Timestamp(config.train_end)],
        "normal_exclude_explicit": state_df[
            (state_df["zone"] == "normal")
            & (~state_df["adjust_date"].isin(pd.to_datetime(["2026-03-24", "2026-04-08"])))
        ],
    }
    metrics = {}
    for sample_name, sample in samples.items():
        metrics[sample_name] = {
            "gasoline_mechanism": metric_block(sample, "gasoline", "gasoline_theory_delta"),
            "diesel_mechanism": metric_block(sample, "diesel", "diesel_theory_delta"),
            "gasoline_multisource": metric_block(sample, "gasoline", "gasoline_multisource_delta"),
            "diesel_multisource": metric_block(sample, "diesel", "diesel_multisource_delta"),
        }

    summary = {
        "fit": fit_info,
        "include_policy_vars_in_ridge": include_policy_vars,
        "transmission_models": {"gasoline": gas_info, "diesel": diesel_info},
        "metrics": metrics,
    }
    return state_df, summary, loadings


def plot_factor(df: pd.DataFrame, variant_name: str, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(pd.to_datetime(df["adjust_date"]), df["factor_1"], color="#1f77b4", linewidth=1.8)
    periods = {
        "2020 shock": ("2020-02-01", "2020-06-30"),
        "2022 high oil": ("2022-02-01", "2022-10-31"),
        "2026 conflict": ("2026-03-01", "2026-05-31"),
    }
    colors = ["#f4a261", "#e76f51", "#2a9d8f"]
    for (label, (start, end)), color in zip(periods.items(), colors):
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.16, color=color, label=label)
    ax.set_title(f"Latent Cost Pressure Factor - {variant_name}")
    ax.set_xlabel("Adjustment date")
    ax.set_ylabel("Smoothed factor")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / f"factor_path_{variant_name}.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-source dynamic-factor state-space experiments.")
    parser.add_argument("--start-date", default=MultiSourceConfig.start_date)
    parser.add_argument("--train-end", default=MultiSourceConfig.train_end)
    parser.add_argument("--window-size", type=int, default=MultiSourceConfig.window_size)
    parser.add_argument("--pricing-lag-days", type=int, default=MultiSourceConfig.pricing_lag_days)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MultiSourceConfig(
        start_date=args.start_date,
        train_end=args.train_end,
        window_size=args.window_size,
        pricing_lag_days=args.pricing_lag_days,
    )
    features = build_window_dataset(config)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    variants = {
        "A_oil_only": (OBS_SETS["A_oil_only"], False),
        "B_oil_fx": (OBS_SETS["B_oil_fx"], False),
        "C_cost_factor": (OBS_SETS["C_cost_factor"], False),
        "D_cost_factor_policy_ridge": (OBS_SETS["C_cost_factor"], True),
        "E_full_variable_factor": (OBS_SETS["D_full_factor"], False),
    }

    summaries = {}
    ablation_rows = []
    main_validation = None
    for variant_name, (obs_cols, include_policy_vars) in variants.items():
        state_df, summary, loadings = evaluate_variant(
            features,
            config,
            variant_name,
            obs_cols,
            include_policy_vars,
        )
        summaries[variant_name] = summary
        state_df.to_csv(OUTPUT_DIR / f"validation_{variant_name}.csv", index=False, encoding="utf-8-sig")
        loadings.to_csv(OUTPUT_DIR / f"loadings_{variant_name}.csv", index=False, encoding="utf-8-sig")
        plot_factor(state_df, variant_name, OUTPUT_DIR)
        if variant_name == "D_cost_factor_policy_ridge":
            main_validation = state_df
            state_df.to_csv(OUTPUT_DIR / "multisource_state_space_validation.csv", index=False, encoding="utf-8-sig")
            loadings.to_csv(OUTPUT_DIR / "loadings_cost_factor.csv", index=False, encoding="utf-8-sig")

        test = summary["metrics"]["test_after_train"]
        ablation_rows.append(
            {
                "variant": variant_name,
                "factor_variables": ", ".join(obs_cols),
                "policy_vars_in_ridge": include_policy_vars,
                "converged": summary["fit"]["converged"],
                "gasoline_test_mae": test["gasoline_multisource"]["mae"],
                "gasoline_test_wmape": test["gasoline_multisource"]["wmape"],
                "diesel_test_mae": test["diesel_multisource"]["mae"],
                "diesel_test_wmape": test["diesel_multisource"]["wmape"],
            }
        )

    ablation = pd.DataFrame(ablation_rows)
    ablation.to_csv(OUTPUT_DIR / "ablation_test_metrics.csv", index=False, encoding="utf-8-sig")
    summary = {"config": asdict(config), "variants": summaries, "ablation_test_metrics": ablation_rows}
    with (OUTPUT_DIR / "summary_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Multi-source dynamic-factor state-space experiments")
    print(ablation.to_string(index=False))
    best_idx = (
        (ablation["gasoline_test_mae"] + ablation["diesel_test_mae"])
        .astype(float)
        .idxmin()
    )
    best_variant = str(ablation.loc[best_idx, "variant"])
    best = summaries[best_variant]["metrics"]
    print(f"\nBest test-MAE variant: {best_variant}")
    print("All gasoline:", best["all"]["gasoline_multisource"])
    print("All diesel:", best["all"]["diesel_multisource"])
    print("Cost-factor control loadings saved to loadings_cost_factor.csv")
    print(f"\nOutputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
