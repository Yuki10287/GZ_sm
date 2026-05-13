from __future__ import annotations

"""任务一基础机制模拟脚本。

本脚本建立最容易解释的基准模型：
1. 用 Brent、WTI、Dubai 推断公开可得的国际原油基准价格；
2. 用 10 个工作日移动均价模拟我国成品油调价规则；
3. 将理论汽柴油调幅与历史公告调幅比较。

该模型定位是“机制复现模型”，不是高频精确预测模型。
"""

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant


ROOT = Path(__file__).resolve().parent
OIL_DIR = ROOT / "国际原油价格数据"
DOMESTIC_PATH = ROOT / "国内柴油汽油调价" / "柴油汽油调价2002-2026.xlsx"
FX_PATH = ROOT / "汇率" / "DEXCHUS.csv"
OUTPUT_DIR = ROOT / "outputs" / "task1"


@dataclass(frozen=True)
class MechanismConfig:
    """机制模拟中使用的政策参数和单位换算参数。"""

    start_date: str = "2016-01-01"
    window_size: int = 10
    pricing_lag_days: int = 1
    threshold_yuan_per_ton: float = 50.0
    floor_usd_per_bbl: float = 40.0
    ceiling_usd_per_bbl: float = 130.0
    ceiling_up_factor: float = 0.2
    tax_factor: float = 1.13
    gasoline_bbl_per_ton: float = 8.6
    diesel_bbl_per_ton: float = 7.3
    train_end: str = "2022-12-31"


def _clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype("string").str.replace(",", "", regex=False).str.replace("%", "", regex=False),
        errors="coerce",
    )


def read_eia_oil_csv(path: Path, value_name: str) -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=4)
    df = df.rename(columns={df.columns[0]: "date", df.columns[1]: value_name})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df[value_name] = _clean_numeric(df[value_name])
    return df[["date", value_name]].dropna().sort_values("date")


def read_dubai_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"Date": "date", "Price": "dubai"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["dubai"] = _clean_numeric(df["dubai"])
    return df[["date", "dubai"]].dropna().sort_values("date")


def read_fx(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"observation_date": "date", "DEXCHUS": "usd_cny"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["usd_cny"] = _clean_numeric(df["usd_cny"])
    return df[["date", "usd_cny"]].dropna().sort_values("date")


def read_oil_panel() -> pd.DataFrame:
    """读取 Brent、WTI、Dubai 和美元兑人民币汇率，并合并成日度面板。"""

    brent = read_eia_oil_csv(OIL_DIR / "Europe_Brent_Spot_Price_FOB.csv", "brent")
    wti = read_eia_oil_csv(OIL_DIR / "Cushing_OK_WTI_Spot_Price_FOB.csv", "wti")
    dubai = read_dubai_csv(OIL_DIR / "Dubai Crude Oil (Platts) Financial Futures Historical Data 2010-2026.csv")
    fx = read_fx(FX_PATH)

    panel = brent.merge(wti, on="date", how="outer").merge(dubai, on="date", how="outer")
    panel = panel.merge(fx, on="date", how="outer").sort_values("date")
    panel[["brent", "wti", "dubai", "usd_cny"]] = panel[["brent", "wti", "dubai", "usd_cny"]].ffill()
    panel = panel.dropna(subset=["brent", "wti", "dubai", "usd_cny"]).reset_index(drop=True)

    panel["basket_usd"] = 0.4 * panel["brent"] + 0.1 * panel["wti"] + 0.5 * panel["dubai"]
    panel, weights = add_pca_kalman_index(panel)
    panel.attrs["pca_weights"] = weights
    return panel


def local_level_kalman_smooth(observed: np.ndarray) -> np.ndarray:
    """用局部水平 Kalman 模型平滑一维油价指数。

    这里故意保持简单：潜在油价水平服从随机游走，观测到的 PCA 指数
    被看作该潜在水平的带噪声测量。
    """

    observed = np.asarray(observed, dtype=float)
    diff_var = float(np.nanvar(np.diff(observed)))
    if not np.isfinite(diff_var) or diff_var <= 0:
        return observed.copy()

    q = max(diff_var * 0.05, 1e-8)
    r = max(diff_var * 0.50, 1e-8)
    n = observed.size

    level = np.zeros(n)
    level_var = np.zeros(n)
    pred = np.zeros(n)
    pred_var = np.zeros(n)

    level[0] = observed[0]
    level_var[0] = r
    for t in range(1, n):
        pred[t] = level[t - 1]
        pred_var[t] = level_var[t - 1] + q
        gain = pred_var[t] / (pred_var[t] + r)
        level[t] = pred[t] + gain * (observed[t] - pred[t])
        level_var[t] = (1.0 - gain) * pred_var[t]

    smooth = level.copy()
    smooth_var = level_var.copy()
    for t in range(n - 2, -1, -1):
        gain = level_var[t] / (level_var[t] + q)
        smooth[t] = level[t] + gain * (smooth[t + 1] - level[t])
        smooth_var[t] = level_var[t] + gain * gain * (smooth_var[t + 1] - level_var[t] - q)
    return smooth


def add_pca_kalman_index(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """构造官方未公开一揽子油价的两个公开代理变量。

    ``basket_usd`` 是文献式固定权重篮子；``kalman_index_usd`` 先用 PCA
    提取三种原油对数价格的共同成分，再做 Kalman 平滑。
    """

    result = panel.copy()
    pca_prices = result[["brent", "wti", "dubai"]].copy()
    pca_prices = pca_prices.mask(pca_prices <= 0)
    pca_prices = pca_prices.interpolate(limit_direction="both").ffill().bfill()
    log_prices = np.log(pca_prices)
    scaled = StandardScaler().fit_transform(log_prices)
    pca = PCA(n_components=1)
    pca.fit(scaled)
    loadings = pca.components_[0]
    if loadings.sum() < 0:
        loadings = -loadings
    weights = np.maximum(loadings, 0)
    if weights.sum() == 0:
        weights = np.abs(loadings)
    weights = weights / weights.sum()
    weight_map = {name: float(weight) for name, weight in zip(["brent", "wti", "dubai"], weights.round(6))}

    log_index = log_prices.to_numpy(dtype=float) @ weights
    result["pca_index_usd"] = np.exp(log_index)
    result["kalman_index_usd"] = np.exp(local_level_kalman_smooth(log_index))
    return result, weight_map


def read_domestic_adjustments(start_date: str) -> pd.DataFrame:
    df = pd.read_excel(DOMESTIC_PATH, sheet_name="Sheet1")
    df = df.rename(
        columns={
            "调整日期": "adjust_date",
            "汽油_价格": "gasoline_price",
            "汽油_涨跌": "gasoline_actual_delta",
            "柴油_价格": "diesel_price",
            "柴油_涨跌": "diesel_actual_delta",
        }
    )
    df["adjust_date"] = pd.to_datetime(df["adjust_date"], errors="coerce")
    numeric_cols = [
        "gasoline_price",
        "gasoline_actual_delta",
        "diesel_price",
        "diesel_actual_delta",
    ]
    for col in numeric_cols:
        df[col] = _clean_numeric(df[col])
    df = df.dropna(subset=["adjust_date"]).sort_values("adjust_date").reset_index(drop=True)
    df = df[df["adjust_date"] >= pd.Timestamp(start_date)].reset_index(drop=True)
    return df


def last_value_before(panel: pd.DataFrame, date: pd.Timestamp, col: str) -> float:
    values = panel.loc[panel["date"] <= date, col]
    if values.empty:
        return float("nan")
    return float(values.iloc[-1])


def window_values(panel: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, col: str, size: int) -> pd.DataFrame:
    window = panel[(panel["date"] > start) & (panel["date"] <= end)].copy()
    if window.empty:
        window = panel[panel["date"] <= end].tail(size).copy()
    else:
        window = window.tail(size)
    return window[["date", col, "usd_cny"]].dropna()


def trailing_window(panel: pd.DataFrame, end: pd.Timestamp, col: str, size: int) -> pd.DataFrame:
    return panel.loc[panel["date"] <= end, ["date", col, "usd_cny"]].dropna().tail(size).copy()


def zone_for_price(price: float, config: MechanismConfig) -> str:
    if price <= config.floor_usd_per_bbl:
        return "floor"
    if price >= config.ceiling_usd_per_bbl:
        return "ceiling"
    return "normal"


def apply_zone_rule(delta: float, ma_usd: float, config: MechanismConfig) -> float:
    if ma_usd <= config.floor_usd_per_bbl and delta < 0:
        return 0.0
    if ma_usd >= config.ceiling_usd_per_bbl and delta > 0:
        return delta * config.ceiling_up_factor
    return delta


def simulate_mechanism(
    domestic: pd.DataFrame,
    panel: pd.DataFrame,
    price_col: str,
    config: MechanismConfig,
) -> pd.DataFrame:
    """在历史调价窗口上模拟现行成品油调价机制。

    关键建模选择：油价变化率用“本窗口 10 日均价”和“上一调价窗口
    10 日均价”比较。这是历史验证中表现较好的优化控制层。
    """

    rows: list[dict[str, object]] = []
    carry_gasoline = 0.0
    carry_diesel = 0.0

    for i in range(1, len(domestic)):
        prev_date = domestic.loc[i - 1, "adjust_date"]
        date = domestic.loc[i, "adjust_date"]
        pricing_end = date - pd.Timedelta(days=config.pricing_lag_days)
        base_pricing_end = prev_date - pd.Timedelta(days=config.pricing_lag_days)
        window = trailing_window(panel, pricing_end, price_col, config.window_size)
        if window.empty:
            continue

        current_ma = float(window[price_col].mean())
        current_ma_ref = max(current_ma, config.floor_usd_per_bbl)
        base_window = trailing_window(panel, base_pricing_end, price_col, config.window_size)
        base_usd = float(base_window[price_col].mean()) if not base_window.empty else float("nan")
        base_ref = max(base_usd, config.floor_usd_per_bbl)
        fx_avg = float(window["usd_cny"].mean())

        if not np.isfinite(base_ref) or base_ref == 0:
            continue

        change_rate = current_ma_ref / base_ref - 1.0
        gasoline_raw = (
            current_ma_ref
            * config.gasoline_bbl_per_ton
            * fx_avg
            * change_rate
            * config.tax_factor
        )
        diesel_raw = current_ma_ref * config.diesel_bbl_per_ton * fx_avg * change_rate * config.tax_factor

        gasoline_policy = apply_zone_rule(gasoline_raw, current_ma, config)
        diesel_policy = apply_zone_rule(diesel_raw, current_ma, config)

        gasoline_total = gasoline_policy + carry_gasoline
        diesel_total = diesel_policy + carry_diesel
        gasoline_theory = gasoline_total if abs(gasoline_total) >= config.threshold_yuan_per_ton else 0.0
        diesel_theory = diesel_total if abs(diesel_total) >= config.threshold_yuan_per_ton else 0.0
        carry_gasoline = 0.0 if gasoline_theory else gasoline_total
        carry_diesel = 0.0 if diesel_theory else diesel_total

        rows.append(
            {
                "adjust_date": date,
                "prev_adjust_date": prev_date,
                "oil_index": price_col,
                "pricing_end": pricing_end,
                "window_observations": len(window),
                "window_start": window["date"].min(),
                "window_end": window["date"].max(),
                "base_usd_per_bbl": base_usd,
                "base_window_observations": len(base_window),
                "base_window_start": base_window["date"].min() if not base_window.empty else pd.NaT,
                "base_window_end": base_window["date"].max() if not base_window.empty else pd.NaT,
                "ma_usd_per_bbl": current_ma,
                "ref_ma_usd_per_bbl": current_ma_ref,
                "avg_usd_cny": fx_avg,
                "oil_change_rate": change_rate,
                "zone": zone_for_price(current_ma, config),
                "gasoline_formula_delta": gasoline_raw,
                "diesel_formula_delta": diesel_raw,
                "gasoline_theory_delta": gasoline_theory,
                "diesel_theory_delta": diesel_theory,
                "gasoline_carry_after": carry_gasoline,
                "diesel_carry_after": carry_diesel,
                "gasoline_actual_delta": domestic.loc[i, "gasoline_actual_delta"],
                "diesel_actual_delta": domestic.loc[i, "diesel_actual_delta"],
                "gasoline_price": domestic.loc[i, "gasoline_price"],
                "diesel_price": domestic.loc[i, "diesel_price"],
            }
        )

    return pd.DataFrame(rows)


def fit_scale(train_actual: pd.Series, train_theory: pd.Series) -> float:
    """估计一个无截距缩放系数，用于可选的调幅校准。"""

    mask = train_actual.notna() & train_theory.notna() & (train_theory.abs() > 1e-9)
    if mask.sum() < 5:
        return 1.0
    x = train_theory[mask].to_numpy(dtype=float)
    y = train_actual[mask].to_numpy(dtype=float)
    return float(np.dot(x, y) / np.dot(x, x))


def add_calibrated_predictions(df: pd.DataFrame, config: MechanismConfig) -> tuple[pd.DataFrame, dict[str, float]]:
    result = df.copy()
    train_mask = result["adjust_date"] <= pd.Timestamp(config.train_end)
    scales = {
        "gasoline": fit_scale(
            result.loc[train_mask, "gasoline_actual_delta"],
            result.loc[train_mask, "gasoline_theory_delta"],
        ),
        "diesel": fit_scale(
            result.loc[train_mask, "diesel_actual_delta"],
            result.loc[train_mask, "diesel_theory_delta"],
        ),
    }
    result["gasoline_calibrated_delta"] = result["gasoline_theory_delta"] * scales["gasoline"]
    result["diesel_calibrated_delta"] = result["diesel_theory_delta"] * scales["diesel"]
    return result, scales


def metric_block(df: pd.DataFrame, product: str, pred_col: str) -> dict[str, float]:
    """计算预测调幅相对实际调幅的评价指标。"""

    actual_col = f"{product}_actual_delta"
    mask = df[actual_col].notna() & df[pred_col].notna()
    actual = df.loc[mask, actual_col].to_numpy(dtype=float)
    pred = df.loc[mask, pred_col].to_numpy(dtype=float)
    if actual.size == 0:
        return {"n": 0}
    err = pred - actual
    nonzero_mask = actual != 0
    direction_accuracy = np.mean(np.sign(pred[nonzero_mask]) == np.sign(actual[nonzero_mask])) if nonzero_mask.any() else np.nan
    threshold_accuracy = np.mean((np.abs(pred) >= 50) == (np.abs(actual) >= 50))
    return {
        "n": int(actual.size),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(math.sqrt(np.mean(err * err))),
        "mean_error": float(np.mean(err)),
        "direction_accuracy": float(direction_accuracy) if np.isfinite(direction_accuracy) else None,
        "threshold_accuracy": float(threshold_accuracy),
        "corr": float(np.corrcoef(actual, pred)[0, 1]) if actual.size > 1 and np.std(pred) > 0 else None,
    }


def build_metrics(df: pd.DataFrame, config: MechanismConfig) -> dict[str, object]:
    metrics: dict[str, object] = {}
    for sample_name, sample in {
        "all": df,
        "test_after_train": df[df["adjust_date"] > pd.Timestamp(config.train_end)],
    }.items():
        metrics[sample_name] = {
            "gasoline_formula": metric_block(sample, "gasoline", "gasoline_theory_delta"),
            "diesel_formula": metric_block(sample, "diesel", "diesel_theory_delta"),
            "gasoline_calibrated": metric_block(sample, "gasoline", "gasoline_calibrated_delta"),
            "diesel_calibrated": metric_block(sample, "diesel", "diesel_calibrated_delta"),
        }

    zone_summary = []
    for zone, group in df.groupby("zone", dropna=False):
        row = {"zone": zone, "n": int(len(group))}
        row.update({f"gasoline_{k}": v for k, v in metric_block(group, "gasoline", "gasoline_calibrated_delta").items()})
        row.update({f"diesel_{k}": v for k, v in metric_block(group, "diesel", "diesel_calibrated_delta").items()})
        zone_summary.append(row)
    metrics["zone_summary"] = zone_summary
    return metrics


def make_lagged(series: pd.Series, lag: int) -> pd.Series:
    return series.shift(lag)


def run_nardl_like_test(df: pd.DataFrame, product: str) -> dict[str, object]:
    """在验证表上运行一个简化版 NARDL 非对称检验。

    这是任务一分析用的诊断近似：把油价上涨和下跌拆开，检验二者的
    短期和长期系数是否存在差异。
    """

    price_col = f"{product}_price"
    data = df[["adjust_date", price_col, "ref_ma_usd_per_bbl", f"{product}_actual_delta"]].copy()
    data["d_price"] = data[price_col].diff()
    data["d_oil"] = data["ref_ma_usd_per_bbl"].diff()
    data["d_oil_pos"] = data["d_oil"].clip(lower=0)
    data["d_oil_neg"] = data["d_oil"].clip(upper=0)
    data["oil_pos_cum"] = data["d_oil_pos"].cumsum()
    data["oil_neg_cum"] = data["d_oil_neg"].cumsum()

    regressors = pd.DataFrame(
        {
            "price_lag1": make_lagged(data[price_col], 1),
            "oil_pos_cum_lag1": make_lagged(data["oil_pos_cum"], 1),
            "oil_neg_cum_lag1": make_lagged(data["oil_neg_cum"], 1),
            "d_price_lag1": make_lagged(data["d_price"], 1),
            "d_oil_pos": data["d_oil_pos"],
            "d_oil_neg": data["d_oil_neg"],
            "d_oil_pos_lag1": make_lagged(data["d_oil_pos"], 1),
            "d_oil_neg_lag1": make_lagged(data["d_oil_neg"], 1),
        }
    )
    y = data["d_price"]
    model_data = pd.concat([y.rename("y"), regressors], axis=1).dropna()
    if len(model_data) < 20:
        return {"n": int(len(model_data)), "error": "not enough observations"}

    x = add_constant(model_data.drop(columns=["y"]))
    model = OLS(model_data["y"], x).fit(cov_type="HC1")

    short_terms = ["d_oil_pos", "d_oil_pos_lag1", "d_oil_neg", "d_oil_neg_lag1"]
    restriction = np.zeros((1, len(model.params)))
    names = list(model.params.index)
    for name in ["d_oil_pos", "d_oil_pos_lag1"]:
        restriction[0, names.index(name)] = 1.0
    for name in ["d_oil_neg", "d_oil_neg_lag1"]:
        restriction[0, names.index(name)] = -1.0
    short_wald = model.wald_test(restriction, scalar=True)

    long_restriction = np.zeros((1, len(model.params)))
    long_restriction[0, names.index("oil_pos_cum_lag1")] = 1.0
    long_restriction[0, names.index("oil_neg_cum_lag1")] = -1.0
    long_wald = model.wald_test(long_restriction, scalar=True)

    rho = model.params.get("price_lag1", np.nan)
    phi_pos = model.params.get("oil_pos_cum_lag1", np.nan)
    phi_neg = model.params.get("oil_neg_cum_lag1", np.nan)
    long_pos = float(-phi_pos / rho) if np.isfinite(rho) and abs(rho) > 1e-12 else None
    long_neg = float(-phi_neg / rho) if np.isfinite(rho) and abs(rho) > 1e-12 else None

    return {
        "n": int(model.nobs),
        "r_squared": float(model.rsquared),
        "long_run_positive": long_pos,
        "long_run_negative": long_neg,
        "short_wald_pvalue": float(short_wald.pvalue),
        "long_wald_pvalue": float(long_wald.pvalue),
        "params": {k: float(v) for k, v in model.params.items()},
    }


def write_outputs(
    panel: pd.DataFrame,
    simulations: dict[str, pd.DataFrame],
    summaries: dict[str, object],
    config: MechanismConfig,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUTPUT_DIR / "oil_panel_with_indices.csv", index=False, encoding="utf-8-sig")
    for name, df in simulations.items():
        df.to_csv(OUTPUT_DIR / f"mechanism_validation_{name}.csv", index=False, encoding="utf-8-sig")
    with (OUTPUT_DIR / "summary_metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"config": asdict(config), "summaries": summaries}, f, ensure_ascii=False, indent=2)


def print_short_report(summaries: dict[str, object]) -> None:
    print("Task 1 price mechanism validation")
    for name, summary in summaries.items():
        print(f"\n[{name}]")
        print("PCA weights:", summary["pca_weights"])
        print("Calibration scales:", summary["calibration_scales"])
        metrics = summary["metrics"]["all"]
        for key, values in metrics.items():
            print(
                f"{key}: n={values.get('n')}, "
                f"MAE={values.get('mae'):.2f}, "
                f"RMSE={values.get('rmse'):.2f}, "
                f"DirAcc={values.get('direction_accuracy')}"
            )
        print("NARDL-like gasoline:", summary["nardl_like"]["gasoline"])
        print("NARDL-like diesel:", summary["nardl_like"]["diesel"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate China's refined-oil price adjustment mechanism.")
    parser.add_argument("--start-date", default=MechanismConfig.start_date)
    parser.add_argument("--train-end", default=MechanismConfig.train_end)
    parser.add_argument("--window-size", type=int, default=MechanismConfig.window_size)
    parser.add_argument("--pricing-lag-days", type=int, default=MechanismConfig.pricing_lag_days)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MechanismConfig(
        start_date=args.start_date,
        train_end=args.train_end,
        window_size=args.window_size,
        pricing_lag_days=args.pricing_lag_days,
    )
    panel = read_oil_panel()
    domestic = read_domestic_adjustments(config.start_date)

    simulations: dict[str, pd.DataFrame] = {}
    summaries: dict[str, object] = {}
    for name, price_col in {
        "fixed_basket": "basket_usd",
        "pca_kalman": "kalman_index_usd",
    }.items():
        simulated = simulate_mechanism(domestic, panel, price_col, config)
        simulated, scales = add_calibrated_predictions(simulated, config)
        simulations[name] = simulated
        summaries[name] = {
            "oil_index_column": price_col,
            "pca_weights": panel.attrs.get("pca_weights", {}),
            "calibration_scales": scales,
            "metrics": build_metrics(simulated, config),
            "nardl_like": {
                "gasoline": run_nardl_like_test(simulated, "gasoline"),
                "diesel": run_nardl_like_test(simulated, "diesel"),
            },
        }

    write_outputs(panel, simulations, summaries, config)
    print_short_report(summaries)
    print(f"\nOutputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
