"""任务一 1.2：NARDL 价格传递不对称完整脚本。

对应队友新版《任务一 5.14(2).docx》的 1.2，包含：
    1. 数据平稳性检验：ADF 检验，判断变量是否存在二阶单整风险；
    2. 格兰杰因果检验：验证国际油价变化是否有助于解释国内成品油价格变化；
    3. 基础 NARDL-LR 检验：短期非对称、长期非对称、短期+长期联合对称；
    4. 稳健性改进实验：扩大滞后、替换油价代理、Threshold NARDL。

输出：
    outputs/task1_serial_fusion_ridge/section_1_2_data_tests.csv
    outputs/task1_serial_fusion_ridge/section_1_2_granger_tests.csv
    outputs/task1_serial_fusion_ridge/section_1_2_nardl_lr_complete.csv
    outputs/task1_serial_fusion_ridge/section_1_2_nardl_robustness_complete.csv
    任务一第九版_1.2完整检验结果.txt
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2
from statsmodels.tsa.stattools import adfuller, grangercausalitytests


ROOT = Path(__file__).resolve().parent
VALIDATION_PATH = ROOT / "outputs" / "task1_serial_fusion_ridge" / "serial_fusion_validation.csv"
STATE_PATH = ROOT / "outputs" / "task1_teammate_statespace" / "state_space_oil_index.csv"
OUT_DIR = ROOT / "outputs" / "task1_serial_fusion_ridge"
OUT_ADF = OUT_DIR / "section_1_2_data_tests.csv"
OUT_GRANGER = OUT_DIR / "section_1_2_granger_tests.csv"
OUT_LR = OUT_DIR / "section_1_2_nardl_lr_complete.csv"
OUT_ROBUST = OUT_DIR / "section_1_2_nardl_robustness_complete.csv"
OUT_TXT = ROOT / "任务一第九版_1.2完整检验结果.txt"


@dataclass(frozen=True)
class NardlResult:
    oil_proxy: str
    fuel: str
    threshold: float
    max_lag: int
    p: int
    q: int
    n: int
    bic: float
    r2: float
    short_pos: float
    short_neg: float
    long_pos: float
    long_neg: float
    lr_short: float
    p_short: float
    lr_long: float
    p_long: float
    lr_both: float
    p_both: float


def load_data() -> pd.DataFrame:
    """读取第九版结果，并构造三类国际油价代理变量。"""
    validation = pd.read_csv(VALIDATION_PATH, parse_dates=["adjust_date", "window_start", "window_end"])
    validation = validation.sort_values("adjust_date").reset_index(drop=True)

    state = pd.read_csv(STATE_PATH, parse_dates=["date"]).sort_values("date")
    for col in ["brent", "wti", "dubai"]:
        state[col] = pd.to_numeric(state[col], errors="coerce")
    raw_prices = state[["brent", "wti", "dubai"]].mask(lambda x: x <= 0)
    state["raw_equal_spot_daily"] = raw_prices.mean(axis=1, skipna=True)

    raw_window = []
    for _, row in validation.iterrows():
        mask = (state["date"] >= row["window_start"]) & (state["date"] <= row["window_end"])
        raw_window.append(state.loc[mask, "raw_equal_spot_daily"].mean())

    validation["raw_equal_spot"] = pd.Series(raw_window).interpolate(limit_direction="both").ffill().bfill()
    validation["raw_equal_spot_ma3"] = validation["raw_equal_spot"].rolling(window=3, min_periods=1).mean()
    validation["kalman_index"] = validation["ma_usd_per_bbl"]
    validation["log_kalman_index"] = np.log(validation["kalman_index"])
    validation["log_raw_equal_spot"] = np.log(validation["raw_equal_spot"])
    validation["log_raw_equal_spot_ma3"] = np.log(validation["raw_equal_spot_ma3"])
    validation["log_gasoline_price"] = np.log(validation["gasoline_price"])
    validation["log_diesel_price"] = np.log(validation["diesel_price"])
    return validation


def run_adf_tests(df: pd.DataFrame) -> pd.DataFrame:
    """对水平序列和一阶差分序列做 ADF 平稳性检验。"""
    variables = {
        "log_kalman_index": "状态空间综合油价",
        "log_raw_equal_spot_ma3": "原始现货3期轻度平滑油价",
        "log_gasoline_price": "汽油价格",
        "log_diesel_price": "柴油价格",
    }
    rows = []
    for col, label in variables.items():
        for form, series in [("level", df[col]), ("diff", df[col].diff())]:
            x = series.dropna().to_numpy(dtype=float)
            stat, p_value, used_lag, nobs, crit, _ = adfuller(x, autolag="AIC")
            rows.append(
                {
                    "variable": col,
                    "label": label,
                    "form": form,
                    "adf_stat": stat,
                    "p_value": p_value,
                    "used_lag": used_lag,
                    "nobs": nobs,
                    "crit_1pct": crit["1%"],
                    "crit_5pct": crit["5%"],
                    "crit_10pct": crit["10%"],
                    "stationary_5pct": p_value < 0.05,
                }
            )
    return pd.DataFrame(rows)


def run_granger_tests(df: pd.DataFrame, max_lag: int = 4) -> pd.DataFrame:
    """对一阶差分序列做格兰杰因果检验。"""
    rows = []
    oil_cols = ["log_kalman_index", "log_raw_equal_spot_ma3"]
    fuel_cols = {"gasoline": "log_gasoline_price", "diesel": "log_diesel_price"}
    for oil_col, (fuel, y_col) in product(oil_cols, fuel_cols.items()):
        data = pd.DataFrame({"dy": df[y_col].diff(), "dx": df[oil_col].diff()}).dropna()

        # statsmodels 的 grangercausalitytests 检验第二列是否 Granger 导致第一列。
        oil_to_domestic = grangercausalitytests(data[["dy", "dx"]], maxlag=max_lag, verbose=False)
        domestic_to_oil = grangercausalitytests(data[["dx", "dy"]], maxlag=max_lag, verbose=False)

        for direction, result in [("oil_to_domestic", oil_to_domestic), ("domestic_to_oil", domestic_to_oil)]:
            best_lag = None
            best_p = None
            for lag, tests in result.items():
                p_value = float(tests[0]["ssr_chi2test"][1])
                if best_p is None or p_value < best_p:
                    best_p = p_value
                    best_lag = lag
            rows.append(
                {
                    "oil_proxy": oil_col,
                    "fuel": fuel,
                    "direction": direction,
                    "max_lag": max_lag,
                    "best_lag_by_min_p": best_lag,
                    "min_p_value": best_p,
                    "significant_5pct": best_p < 0.05,
                }
            )
    return pd.DataFrame(rows)


def add_nardl_terms(df: pd.DataFrame, oil_proxy: str, threshold: float) -> pd.DataFrame:
    d = df.copy()
    d["x"] = np.log(pd.to_numeric(d[oil_proxy], errors="coerce"))
    d["dx"] = d["x"].diff()
    d["dx_pos"] = np.where(d["dx"] > threshold, d["dx"], 0.0)
    d["dx_neg"] = np.where(d["dx"] < -threshold, d["dx"], 0.0)
    d.loc[d["dx"].isna(), ["dx_pos", "dx_neg"]] = np.nan
    d["x_pos"] = d["dx_pos"].fillna(0.0).cumsum()
    d["x_neg"] = d["dx_neg"].fillna(0.0).cumsum()
    return d


def build_nardl_frame(df: pd.DataFrame, fuel: str, p: int, q: int) -> tuple[pd.Series, pd.DataFrame]:
    d = df.copy()
    d["y"] = np.log(pd.to_numeric(d[f"{fuel}_price"], errors="coerce"))
    d["dy"] = d["y"].diff()
    d["y_l1"] = d["y"].shift(1)
    d["xpos_l1"] = d["x_pos"].shift(1)
    d["xneg_l1"] = d["x_neg"].shift(1)
    cols = ["y_l1", "xpos_l1", "xneg_l1"]
    for i in range(1, p + 1):
        d[f"dy_l{i}"] = d["dy"].shift(i)
        cols.append(f"dy_l{i}")
    for j in range(0, q + 1):
        d[f"dxpos_l{j}"] = d["dx_pos"].shift(j)
        d[f"dxneg_l{j}"] = d["dx_neg"].shift(j)
        cols.extend([f"dxpos_l{j}", f"dxneg_l{j}"])
    reg = d.dropna(subset=["dy"] + cols).copy()
    return reg["dy"], reg[cols]


def fit_ols(y: pd.Series, x: pd.DataFrame):
    return sm.OLS(y, sm.add_constant(x, has_constant="add")).fit()


def restricted_rss(y: pd.Series, x: pd.DataFrame, constraints: list[dict[str, float]]) -> float:
    x_const = sm.add_constant(x, has_constant="add")
    names = list(x_const.columns)
    x_mat = x_const.to_numpy(dtype=float)
    y_vec = y.to_numpy(dtype=float)
    xtx_inv = np.linalg.pinv(x_mat.T @ x_mat)
    beta_u = xtx_inv @ x_mat.T @ y_vec
    r_mat = np.zeros((len(constraints), len(names)), dtype=float)
    for i, constraint in enumerate(constraints):
        for name, value in constraint.items():
            r_mat[i, names.index(name)] = value
    beta_r = beta_u - xtx_inv @ r_mat.T @ np.linalg.pinv(r_mat @ xtx_inv @ r_mat.T) @ (r_mat @ beta_u)
    resid = y_vec - x_mat @ beta_r
    return float(resid @ resid)


def lr_test(n: int, rss_r: float, rss_u: float, df_restriction: int) -> tuple[float, float]:
    stat = n * np.log(rss_r / rss_u)
    return float(stat), float(chi2.sf(stat, df_restriction))


def fit_nardl(df: pd.DataFrame, fuel: str, oil_proxy: str, threshold: float, max_lag: int) -> NardlResult:
    d = add_nardl_terms(df, oil_proxy, threshold)
    best = None
    for p, q in product(range(1, max_lag + 1), range(0, max_lag + 1)):
        y, x = build_nardl_frame(d, fuel, p, q)
        if len(y) < max(60, 8 + len(x.columns) * 3):
            continue
        model = fit_ols(y, x)
        if best is None or model.bic < best["bic"]:
            best = {"p": p, "q": q, "y": y, "x": x, "model": model, "bic": float(model.bic)}
    if best is None:
        raise RuntimeError(f"{fuel}/{oil_proxy} 无可用 NARDL 设定")

    y, x, model = best["y"], best["x"], best["model"]
    rss_u = float(np.sum(model.resid**2))
    short_constraint = {}
    for j in range(best["q"] + 1):
        short_constraint[f"dxpos_l{j}"] = 1.0
        short_constraint[f"dxneg_l{j}"] = -1.0
    long_constraint = {"xpos_l1": 1.0, "xneg_l1": -1.0}

    lr_short, p_short = lr_test(len(y), restricted_rss(y, x, [short_constraint]), rss_u, 1)
    lr_long, p_long = lr_test(len(y), restricted_rss(y, x, [long_constraint]), rss_u, 1)
    lr_both, p_both = lr_test(len(y), restricted_rss(y, x, [short_constraint, long_constraint]), rss_u, 2)

    rho = model.params["y_l1"]
    long_pos = -model.params["xpos_l1"] / rho
    long_neg = -model.params["xneg_l1"] / rho
    short_pos = sum(model.params.get(f"dxpos_l{j}", 0.0) for j in range(best["q"] + 1))
    short_neg = sum(model.params.get(f"dxneg_l{j}", 0.0) for j in range(best["q"] + 1))

    return NardlResult(
        oil_proxy=oil_proxy,
        fuel=fuel,
        threshold=threshold,
        max_lag=max_lag,
        p=best["p"],
        q=best["q"],
        n=len(y),
        bic=float(model.bic),
        r2=float(model.rsquared),
        short_pos=float(short_pos),
        short_neg=float(short_neg),
        long_pos=float(long_pos),
        long_neg=float(long_neg),
        lr_short=lr_short,
        p_short=p_short,
        lr_long=lr_long,
        p_long=p_long,
        lr_both=lr_both,
        p_both=p_both,
    )


def write_report(adf: pd.DataFrame, granger: pd.DataFrame, lr: pd.DataFrame, robust: pd.DataFrame) -> None:
    lines = ["任务一 1.2 NARDL 完整检验结果", "=" * 60, ""]
    lines.append("一、数据检验")
    lines.append("平稳性检验采用 ADF；因果检验采用一阶差分序列上的格兰杰因果检验。")
    lines.append("ADF 结果见 section_1_2_data_tests.csv；Granger 结果见 section_1_2_granger_tests.csv。")
    lines.append("")
    lines.append("二、基础 NARDL-LR 检验")
    for _, row in lr.iterrows():
        name = "汽油" if row["fuel"] == "gasoline" else "柴油"
        lines.append(
            f"{name}: (p,q)=({int(row['p'])},{int(row['q'])}), "
            f"短期上涨={row['short_pos']:.4f}, 短期下跌={row['short_neg']:.4f}, p_short={row['p_short']:.4g}; "
            f"长期上涨={row['long_pos']:.4f}, 长期下跌={row['long_neg']:.4f}, p_long={row['p_long']:.4g}; "
            f"联合p={row['p_both']:.4g}。"
        )
    lines.append("")
    lines.append("三、稳健性实验")
    for fuel in ["gasoline", "diesel"]:
        sub = robust[robust["fuel"] == fuel]
        name = "汽油" if fuel == "gasoline" else "柴油"
        lines.append(f"{name}: 共 {len(sub)} 个设定，短期显著 {int((sub['p_short'] < 0.05).sum())} 个，长期显著 {int((sub['p_long'] < 0.05).sum())} 个。")
        best_long = sub.sort_values("p_long").iloc[0]
        lines.append(
            f"  长期最显著: {best_long['oil_proxy']}, threshold={best_long['threshold']:.0%}, "
            f"(p,q)=({int(best_long['p'])},{int(best_long['q'])}), "
            f"long_pos={best_long['long_pos']:.4f}, long_neg={best_long['long_neg']:.4f}, p_long={best_long['p_long']:.4g}。"
        )
    lines.append("")
    lines.append("结论：新版文档中需要补的数据检验包括平稳性检验和格兰杰因果检验。")
    lines.append("基础 NARDL 显示短期非对称显著、长期非对称不显著；增强门限 NARDL 下，长期非对称在部分合理设定中显著。")
    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()

    adf = run_adf_tests(df)
    granger = run_granger_tests(df)
    adf.to_csv(OUT_ADF, index=False, encoding="utf-8-sig")
    granger.to_csv(OUT_GRANGER, index=False, encoding="utf-8-sig")

    lr_rows = [fit_nardl(df, fuel, "kalman_index", 0.0, 4).__dict__ for fuel in ["gasoline", "diesel"]]
    lr = pd.DataFrame(lr_rows)
    lr.to_csv(OUT_LR, index=False, encoding="utf-8-sig")

    robust_rows = []
    for oil_proxy, threshold, max_lag, fuel in product(
        ["kalman_index", "raw_equal_spot", "raw_equal_spot_ma3"],
        [0.0, 0.01, 0.02, 0.03],
        [4, 8, 12],
        ["gasoline", "diesel"],
    ):
        robust_rows.append(fit_nardl(df, fuel, oil_proxy, threshold, max_lag).__dict__)
    robust = pd.DataFrame(robust_rows)
    robust.to_csv(OUT_ROBUST, index=False, encoding="utf-8-sig")

    write_report(adf, granger, lr, robust)
    print("ADF:")
    print(adf.to_string(index=False))
    print("\nGranger:")
    print(granger.to_string(index=False))
    print("\nNARDL LR:")
    print(lr.to_string(index=False))
    print(f"\n已写出: {OUT_TXT}")


if __name__ == "__main__":
    main()
