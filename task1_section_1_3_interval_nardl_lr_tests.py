"""任务一 1.3：区间 NARDL 的 LR 似然比检验。

严格对应队友新版《任务一 5.14(2).docx》的 1.3：
    1. 按 40/130 美元制度边界定义低油价、正常油价、高油价三个虚拟变量；
    2. 将区间虚拟变量与 NARDL 长期正负累计项相乘；
    3. 对每个区间检验“涨价传导系数 = 降价传导系数”；
    4. 检验方法使用 LR 似然比检验，与 1.2 保持一致。

输出：
    outputs/task1_serial_fusion_ridge/section_1_3_interval_nardl_lr_tests.csv
    任务一第九版_1.3区间NARDL_LR检验结果.txt
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2


ROOT = Path(__file__).resolve().parent
VALIDATION_PATH = ROOT / "outputs" / "task1_serial_fusion_ridge" / "serial_fusion_validation.csv"
STATE_PATH = ROOT / "outputs" / "task1_teammate_statespace" / "state_space_oil_index.csv"
OUT_CSV = ROOT / "outputs" / "task1_serial_fusion_ridge" / "section_1_3_interval_nardl_lr_tests.csv"
OUT_TXT = ROOT / "任务一第九版_1.3区间NARDL_LR检验结果.txt"


@dataclass(frozen=True)
class Scenario:
    name: str
    oil_proxy: str
    threshold: float
    max_lag: int
    note: str


SCENARIOS = [
    Scenario("baseline_kalman_0pct", "kalman_index", 0.0, 4, "基础口径：状态空间综合油价，0%门限。"),
    Scenario("robust_raw_ma3_2pct", "raw_equal_spot_ma3", 0.02, 4, "增强口径：原始三油种等权价3期轻度平滑，2%门限。"),
]

ZONE_ORDER = ["low", "normal", "high"]
ZONE_NAME = {"low": "低油价区间(<40)", "normal": "正常油价区间(40-130)", "high": "高油价区间(>=130)"}


def load_data() -> pd.DataFrame:
    """读取第九版输出，并补充原始三油种现货价代理。"""
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
    return validation


def add_terms(df: pd.DataFrame, oil_proxy: str, threshold: float) -> pd.DataFrame:
    """构造 Threshold NARDL 项和三个区间虚拟变量。"""
    d = df.copy()
    d["x_level"] = pd.to_numeric(d[oil_proxy], errors="coerce")
    d["x"] = np.log(d["x_level"])
    d["dx"] = d["x"].diff()
    d["dx_pos"] = np.where(d["dx"] > threshold, d["dx"], 0.0)
    d["dx_neg"] = np.where(d["dx"] < -threshold, d["dx"], 0.0)
    d.loc[d["dx"].isna(), ["dx_pos", "dx_neg"]] = np.nan
    d["x_pos"] = d["dx_pos"].fillna(0.0).cumsum()
    d["x_neg"] = d["dx_neg"].fillna(0.0).cumsum()

    d["D_low"] = (d["x_level"] < 40).astype(int)
    d["D_normal"] = ((d["x_level"] >= 40) & (d["x_level"] < 130)).astype(int)
    d["D_high"] = (d["x_level"] >= 130).astype(int)

    for zone in ZONE_ORDER:
        d[f"{zone}_pos_base"] = d[f"D_{zone}"] * d["x_pos"]
        d[f"{zone}_neg_base"] = d[f"D_{zone}"] * d["x_neg"]
    return d


def build_frame(df: pd.DataFrame, fuel: str, p: int, q: int) -> tuple[pd.Series, pd.DataFrame]:
    """构造区间 NARDL 误差修正模型矩阵。"""
    d = df.copy()
    d["y"] = np.log(pd.to_numeric(d[f"{fuel}_price"], errors="coerce"))
    d["dy"] = d["y"].diff()
    d["y_l1"] = d["y"].shift(1)

    cols = ["y_l1"]
    for zone in ZONE_ORDER:
        for sign in ["pos", "neg"]:
            col = f"{zone}_{sign}_l1"
            d[col] = d[f"{zone}_{sign}_base"].shift(1)
            if d[col].abs().sum(skipna=True) > 1e-12:
                cols.append(col)

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
    """计算 R beta = 0 约束下的 OLS RSS。"""
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


def lr_pvalue(n: int, rss_r: float, rss_u: float, df_restriction: int) -> tuple[float, float]:
    stat = n * np.log(rss_r / rss_u)
    return float(stat), float(chi2.sf(stat, df_restriction))


def choose_lags(df: pd.DataFrame, fuel: str, max_lag: int) -> tuple[int, int]:
    best = None
    for p, q in product(range(1, max_lag + 1), range(0, max_lag + 1)):
        y, x = build_frame(df, fuel, p, q)
        if len(y) < max(60, 8 + len(x.columns) * 3):
            continue
        model = fit_ols(y, x)
        if best is None or model.bic < best[0]:
            best = (float(model.bic), p, q)
    if best is None:
        raise RuntimeError(f"{fuel} 没有可用滞后阶数组合")
    return best[1], best[2]


def run_one(df: pd.DataFrame, scenario: Scenario, fuel: str) -> list[dict[str, object]]:
    d = add_terms(df, scenario.oil_proxy, scenario.threshold)
    p, q = choose_lags(d, fuel, scenario.max_lag)
    y, x = build_frame(d, fuel, p, q)
    model = fit_ols(y, x)
    rss_u = float(np.sum(model.resid**2))
    rho = model.params["y_l1"]

    rows = []
    for zone in ZONE_ORDER:
        pos_col = f"{zone}_pos_l1"
        neg_col = f"{zone}_neg_l1"
        count = int(d[f"D_{zone}"].sum())
        base = {
            "scenario": scenario.name,
            "scenario_note": scenario.note,
            "fuel": fuel,
            "oil_proxy": scenario.oil_proxy,
            "threshold": scenario.threshold,
            "p": p,
            "q": q,
            "n_regression": int(len(y)),
            "zone": zone,
            "zone_name": ZONE_NAME[zone],
            "zone_window_count": count,
            "model_r2": float(model.rsquared),
            "model_bic": float(model.bic),
        }
        if pos_col not in x.columns or neg_col not in x.columns:
            rows.append(base | {"available": False, "reason": "该区间没有可识别的正负长期交互项。"})
            continue

        rss_r = restricted_rss(y, x, [{pos_col: 1.0, neg_col: -1.0}])
        lr_stat, p_value = lr_pvalue(len(y), rss_r, rss_u, 1)
        long_pos = -model.params[pos_col] / rho
        long_neg = -model.params[neg_col] / rho
        rows.append(
            base
            | {
                "available": True,
                "reason": "",
                "long_pos": float(long_pos),
                "long_neg": float(long_neg),
                "lr_stat": lr_stat,
                "p_value": p_value,
                "decision_5pct": "拒绝对称" if p_value < 0.05 else "不能拒绝对称",
            }
        )
    return rows


def write_report(results: pd.DataFrame) -> None:
    fuel_name = {"gasoline": "汽油", "diesel": "柴油"}
    lines = [
        "任务一 1.3 区间 NARDL 的 LR 似然比检验结果",
        "=" * 60,
        "",
        "检验对象：每个油价区间内，长期上涨传导系数是否等于长期下跌传导系数。",
        "原假设 H0：该区间涨价传导系数 = 该区间降价传导系数。",
        "",
    ]
    for scenario in SCENARIOS:
        sub_s = results[results["scenario"] == scenario.name]
        lines.append(f"情景：{scenario.name}")
        lines.append(f"说明：{scenario.note}")
        for fuel in ["gasoline", "diesel"]:
            sub = sub_s[sub_s["fuel"] == fuel]
            if sub.empty:
                continue
            first = sub.iloc[0]
            lines.append(f"{fuel_name[fuel]}：最优滞后 (p,q)=({int(first['p'])},{int(first['q'])})")
            for _, row in sub.iterrows():
                lines.append(f"  {row['zone_name']}：窗口数 {int(row['zone_window_count'])}")
                if not bool(row["available"]):
                    lines.append(f"    不可检验：{row['reason']}")
                else:
                    lines.append(
                        f"    长期上涨={row['long_pos']:.4f}，长期下跌={row['long_neg']:.4f}，"
                        f"LR={row['lr_stat']:.4f}，p={row['p_value']:.4g}，结论：{row['decision_5pct']}。"
                    )
            lines.append("")
    lines.append("说明：样本期没有 130 美元/桶以上窗口，因此高油价天花板区间不能直接实证检验。")
    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df = load_data()
    rows = []
    for scenario, fuel in product(SCENARIOS, ["gasoline", "diesel"]):
        rows.extend(run_one(df, scenario, fuel))
    results = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    write_report(results)
    print(results.to_string(index=False))
    print(f"\n已写出: {OUT_CSV}")
    print(f"已写出: {OUT_TXT}")


if __name__ == "__main__":
    main()
