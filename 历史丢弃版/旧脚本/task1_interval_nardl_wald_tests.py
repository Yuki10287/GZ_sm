"""任务一 1.3：严格按队友文档思路做区间 NARDL Wald 假设检验。

对应《任务一 5.12 18点.docx》的 1.3：
    1. 根据 40 美元地板价、130 美元天花板价定义三个区间虚拟变量；
    2. 将区间虚拟变量与基础 NARDL 的长期正负累计项相乘；
    3. 分别检验每个区间内“涨价传导系数 = 降价传导系数”。

注意：
    样本期最高窗口油价未达到 130 美元/桶，因此高油价天花板区间没有样本。
    脚本会保留该区间的检验位置，但将结果标记为 unavailable。

输出：
    outputs/task1_serial_fusion_ridge/section_1_3_interval_nardl_wald_tests.csv
    任务一第九版_1.3区间NARDL_Wald检验结果.txt
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parent
VALIDATION_PATH = ROOT / "outputs" / "task1_serial_fusion_ridge" / "serial_fusion_validation.csv"
STATE_PATH = ROOT / "outputs" / "task1_teammate_statespace" / "state_space_oil_index.csv"
OUT_CSV = ROOT / "outputs" / "task1_serial_fusion_ridge" / "section_1_3_interval_nardl_wald_tests.csv"
OUT_TXT = ROOT / "任务一第九版_1.3区间NARDL_Wald检验结果.txt"


@dataclass(frozen=True)
class Scenario:
    """一组 1.3 区间 NARDL 检验设定。"""

    name: str
    oil_proxy: str
    threshold: float
    max_lag: int
    note: str


SCENARIOS = [
    Scenario(
        name="doc_baseline_kalman_0pct",
        oil_proxy="kalman_index",
        threshold=0.0,
        max_lag=4,
        note="严格基础口径：使用第九版状态空间综合油价，0门限NARDL。",
    ),
    Scenario(
        name="robust_raw_ma3_2pct",
        oil_proxy="raw_equal_spot_ma3",
        threshold=0.02,
        max_lag=4,
        note="增强口径：使用原始三油种等权价3期轻度平滑，并采用2%门限。",
    ),
]


ZONE_ORDER = ["low", "normal", "high"]
ZONE_NAME = {
    "low": "低油价区间(<40)",
    "normal": "正常油价区间(40-130)",
    "high": "高油价区间(>=130)",
}


def load_data() -> pd.DataFrame:
    """读取第九版验证结果，并补充原始三油种等权油价。"""
    validation = pd.read_csv(VALIDATION_PATH, parse_dates=["adjust_date", "window_start", "window_end"])
    validation = validation.sort_values("adjust_date").reset_index(drop=True)

    state = pd.read_csv(STATE_PATH, parse_dates=["date"]).sort_values("date")
    for col in ["brent", "wti", "dubai"]:
        state[col] = pd.to_numeric(state[col], errors="coerce")

    # 原始现货价口径：不做卡尔曼滤波。WTI 负值或非正值不进入均值。
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


def add_threshold_nardl_terms(df: pd.DataFrame, oil_proxy: str, threshold: float) -> pd.DataFrame:
    """构造正负冲击累计项和三个制度区间虚拟变量。"""
    d = df.copy()
    d["x_level"] = pd.to_numeric(d[oil_proxy], errors="coerce")
    d["x"] = np.log(d["x_level"])
    d["dx"] = d["x"].diff()

    # threshold=0 时退化为传统 NARDL；threshold>0 时为门限 NARDL。
    d["dx_pos"] = np.where(d["dx"] > threshold, d["dx"], 0.0)
    d["dx_neg"] = np.where(d["dx"] < -threshold, d["dx"], 0.0)
    d.loc[d["dx"].isna(), ["dx_pos", "dx_neg"]] = np.nan
    d["x_pos"] = d["dx_pos"].fillna(0.0).cumsum()
    d["x_neg"] = d["dx_neg"].fillna(0.0).cumsum()

    d["D_low"] = (d["x_level"] < 40).astype(int)
    d["D_high"] = (d["x_level"] >= 130).astype(int)
    d["D_normal"] = ((d["x_level"] >= 40) & (d["x_level"] < 130)).astype(int)

    for zone in ZONE_ORDER:
        dummy = f"D_{zone}"
        d[f"{zone}_pos_l1_base"] = d[dummy] * d["x_pos"]
        d[f"{zone}_neg_l1_base"] = d[dummy] * d["x_neg"]
    return d


def build_interval_nardl_frame(df: pd.DataFrame, fuel: str, p: int, q: int) -> tuple[pd.Series, pd.DataFrame]:
    """构造区间 NARDL 的误差修正回归矩阵。"""
    d = df.copy()
    d["y"] = np.log(pd.to_numeric(d[f"{fuel}_price"], errors="coerce"))
    d["dy"] = d["y"].diff()
    d["y_l1"] = d["y"].shift(1)

    cols = ["y_l1"]

    # 长期项：区间虚拟变量 × 正负累计冲击项，并滞后一阶。
    for zone in ZONE_ORDER:
        for sign in ["pos", "neg"]:
            col = f"{zone}_{sign}_l1"
            d[col] = d[f"{zone}_{sign}_l1_base"].shift(1)
            # 全零列不可识别，例如样本期没有高油价区间时 high 列会全零。
            if d[col].abs().sum(skipna=True) > 1e-12:
                cols.append(col)

    # 短期动态项保留基础 NARDL 的正负冲击滞后，不做区间交互。
    for i in range(1, p + 1):
        d[f"dy_l{i}"] = d["dy"].shift(i)
        cols.append(f"dy_l{i}")
    for j in range(0, q + 1):
        d[f"dxpos_l{j}"] = d["dx_pos"].shift(j)
        d[f"dxneg_l{j}"] = d["dx_neg"].shift(j)
        cols.extend([f"dxpos_l{j}", f"dxneg_l{j}"])

    reg = d.dropna(subset=["dy"] + cols).copy()
    return reg["dy"], reg[cols]


def fit_model(y: pd.Series, x: pd.DataFrame):
    """使用异方差稳健标准误拟合区间 NARDL。"""
    return sm.OLS(y, sm.add_constant(x, has_constant="add")).fit(cov_type="HC3")


def choose_lags(df: pd.DataFrame, fuel: str, max_lag: int) -> tuple[int, int]:
    """用 BIC 在区间 NARDL 中选择最优 (p,q)。"""
    best = None
    for p, q in product(range(1, max_lag + 1), range(0, max_lag + 1)):
        y, x = build_interval_nardl_frame(df, fuel, p, q)
        min_required = max(60, 8 + len(x.columns) * 3)
        if len(y) < min_required:
            continue
        try:
            model = fit_model(y, x)
        except np.linalg.LinAlgError:
            continue
        if best is None or model.bic < best[0]:
            best = (float(model.bic), p, q)
    if best is None:
        raise RuntimeError(f"{fuel} 无法选择有效滞后阶数")
    return best[1], best[2]


def run_one(df: pd.DataFrame, scenario: Scenario, fuel: str) -> list[dict[str, object]]:
    """执行一个情景、一个品种的区间 Wald 检验。"""
    d = add_threshold_nardl_terms(df, scenario.oil_proxy, scenario.threshold)
    p, q = choose_lags(d, fuel, scenario.max_lag)
    y, x = build_interval_nardl_frame(d, fuel, p, q)
    model = fit_model(y, x)

    params = model.params
    rho = params["y_l1"]
    rows: list[dict[str, object]] = []

    for zone in ZONE_ORDER:
        pos_col = f"{zone}_pos_l1"
        neg_col = f"{zone}_neg_l1"
        zone_count = int(d[f"D_{zone}"].sum())
        row: dict[str, object] = {
            "scenario": scenario.name,
            "scenario_note": scenario.note,
            "fuel": fuel,
            "oil_proxy": scenario.oil_proxy,
            "threshold": scenario.threshold,
            "max_lag": scenario.max_lag,
            "p": p,
            "q": q,
            "n_regression": int(len(y)),
            "zone": zone,
            "zone_name": ZONE_NAME[zone],
            "zone_window_count": zone_count,
            "model_r2": float(model.rsquared),
            "model_bic": float(model.bic),
        }

        if pos_col not in x.columns or neg_col not in x.columns:
            row.update(
                {
                    "available": False,
                    "reason": "该区间没有可识别的正负长期交互项，通常是样本数为0或变量全为0。",
                    "theta_pos": np.nan,
                    "theta_neg": np.nan,
                    "long_pos": np.nan,
                    "long_neg": np.nan,
                    "wald_stat": np.nan,
                    "p_value": np.nan,
                    "decision_5pct": "不可检验",
                }
            )
            rows.append(row)
            continue

        # 文档中的原假设：该区间涨价传导系数 = 降价传导系数。
        # 因为长期乘数都是 -theta/rho，所以等价于 theta_pos = theta_neg。
        wald = model.wald_test(f"{pos_col} = {neg_col}", scalar=True)
        p_value = float(wald.pvalue)
        row.update(
            {
                "available": True,
                "reason": "",
                "theta_pos": float(params[pos_col]),
                "theta_neg": float(params[neg_col]),
                "long_pos": float(-params[pos_col] / rho),
                "long_neg": float(-params[neg_col] / rho),
                "wald_stat": float(wald.statistic),
                "p_value": p_value,
                "decision_5pct": "拒绝对称" if p_value < 0.05 else "不能拒绝对称",
            }
        )
        rows.append(row)
    return rows


def write_report(results: pd.DataFrame) -> None:
    """写出中文结果说明。"""
    fuel_name = {"gasoline": "汽油", "diesel": "柴油"}
    lines: list[str] = []
    lines.append("任务一 1.3 区间 NARDL Wald 检验结果")
    lines.append("=" * 60)
    lines.append("")
    lines.append("检验思路严格对应队友文档 1.3：")
    lines.append("将低油价、正常油价、高油价虚拟变量与 NARDL 长期正负累计项相乘，")
    lines.append("分别检验每个区间内“涨价传导系数 = 降价传导系数”。")
    lines.append("")
    lines.append("三个制度区间：")
    lines.append("低油价区间：X < 40 美元/桶；")
    lines.append("正常油价区间：40 <= X < 130 美元/桶；")
    lines.append("高油价区间：X >= 130 美元/桶。")
    lines.append("")

    for scenario in SCENARIOS:
        sub_s = results[results["scenario"] == scenario.name]
        lines.append(f"情景：{scenario.name}")
        lines.append(f"说明：{scenario.note}")
        lines.append("")
        for fuel in ["gasoline", "diesel"]:
            sub = sub_s[sub_s["fuel"] == fuel]
            if sub.empty:
                continue
            first = sub.iloc[0]
            lines.append(f"{fuel_name[fuel]}：最优滞后 (p,q)=({int(first['p'])},{int(first['q'])})，回归样本数 {int(first['n_regression'])}")
            for _, row in sub.iterrows():
                lines.append(f"  {row['zone_name']}：窗口数 {int(row['zone_window_count'])}")
                if not bool(row["available"]):
                    lines.append(f"    结果：不可检验。原因：{row['reason']}")
                    continue
                lines.append(
                    f"    长期上涨系数={row['long_pos']:.4f}，长期下跌系数={row['long_neg']:.4f}，"
                    f"Wald p值={row['p_value']:.4g}，结论：{row['decision_5pct']}。"
                )
            lines.append("")

    lines.append("总体说明：")
    lines.append("样本期内没有出现 130 美元/桶以上窗口，因此高油价天花板区间无法实证检验。")
    lines.append("低油价区间窗口数很少，检验结果需要谨慎解释。")
    lines.append("正常油价区间是样本主体，其 Wald 检验结果更适合作为 1.3 的主要实证依据。")
    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df = load_data()
    rows: list[dict[str, object]] = []
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
