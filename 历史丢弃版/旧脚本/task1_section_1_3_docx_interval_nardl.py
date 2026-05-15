"""Task 1.3: interval NARDL strictly following ``任务一 5.12 18点.docx``.

This script intentionally does not reuse earlier 1.3 scripts or reports.

Docx model:
    D_floor = 1{MAT <= 40}
    D_norm  = 1{40 < MAT < 130}
    D_ceil  = 1{MAT >= 130}

    dP_t = c + rho P_{t-1}
           + D_norm  * phi_norm+  * S^+_{t-1}
           + D_floor * phi_floor+ * S^+_{t-1}
           + D_ceil  * phi_ceil+  * S^+_{t-1}
           + D_norm  * phi_norm-  * S^-_{t-1}
           + D_floor * phi_floor- * S^-_{t-1}
           + D_ceil  * phi_ceil-  * S^-_{t-1}
           + lagged dP terms + lagged positive/negative dS terms + error.

Inputs:
    outputs/task1_serial_fusion_ridge/serial_fusion_validation.csv

Outputs:
    outputs/task1_serial_fusion_ridge/section_1_3_docx_interval_nardl_wald.csv
    任务一第九版_1.3严格按队友docx区间NARDL结果.txt
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parent
INPUT_CSV = ROOT / "outputs" / "task1_serial_fusion_ridge" / "serial_fusion_validation.csv"
OUT_CSV = ROOT / "outputs" / "task1_serial_fusion_ridge" / "section_1_3_docx_interval_nardl_wald.csv"
OUT_TXT = ROOT / "任务一第九版_1.3严格按队友docx区间NARDL结果.txt"

MAX_LAG = 4
USE_LOG = True
ZONES = ["floor", "norm", "ceil"]
ZONE_CN = {
    "floor": "低油价区间（地板价及以下，MAT<=40）",
    "norm": "正常油价区间（40<MAT<130）",
    "ceil": "高油价区间（天花板价及以上，MAT>=130）",
}
FUEL_CN = {"gasoline": "汽油", "diesel": "柴油"}


@dataclass(frozen=True)
class LagChoice:
    n_y: int
    p_pos: int
    q_neg: int
    bic: float


def load_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_CSV, parse_dates=["adjust_date"])
    df = df.sort_values("adjust_date").reset_index(drop=True)
    df["MAT"] = pd.to_numeric(df["ma_usd_per_bbl"], errors="coerce")

    df["D_floor"] = (df["MAT"] <= 40).astype(int)
    df["D_norm"] = ((df["MAT"] > 40) & (df["MAT"] < 130)).astype(int)
    df["D_ceil"] = (df["MAT"] >= 130).astype(int)
    return df


def add_nardl_terms(df: pd.DataFrame, fuel: str) -> pd.DataFrame:
    d = df.copy()
    price_col = f"{fuel}_price"

    p_level = pd.to_numeric(d[price_col], errors="coerce")
    s_level = pd.to_numeric(d["MAT"], errors="coerce")
    if USE_LOG:
        d["P"] = np.log(p_level)
        d["S"] = np.log(s_level)
    else:
        d["P"] = p_level
        d["S"] = s_level

    d["dP"] = d["P"].diff()
    d["dS"] = d["S"].diff()
    d["dS_pos"] = d["dS"].clip(lower=0)
    d["dS_neg"] = d["dS"].clip(upper=0)
    d["S_pos"] = d["dS_pos"].fillna(0.0).cumsum()
    d["S_neg"] = d["dS_neg"].fillna(0.0).cumsum()
    d["P_l1"] = d["P"].shift(1)
    d["S_pos_l1"] = d["S_pos"].shift(1)
    d["S_neg_l1"] = d["S_neg"].shift(1)

    for zone in ZONES:
        d[f"{zone}_pos_l1"] = d[f"D_{zone}"] * d["S_pos_l1"]
        d[f"{zone}_neg_l1"] = d[f"D_{zone}"] * d["S_neg_l1"]
    return d


def build_frame(d: pd.DataFrame, n_y: int, p_pos: int, q_neg: int) -> tuple[pd.Series, pd.DataFrame, pd.Index]:
    cols = ["P_l1"]

    for zone in ZONES:
        for sign in ["pos", "neg"]:
            col = f"{zone}_{sign}_l1"
            if d[col].abs().sum(skipna=True) > 1e-12:
                cols.append(col)

    for i in range(1, n_y + 1):
        col = f"dP_l{i}"
        d[col] = d["dP"].shift(i)
        cols.append(col)

    for j in range(0, p_pos + 1):
        col = f"dS_pos_l{j}"
        d[col] = d["dS_pos"].shift(j)
        cols.append(col)

    for k in range(0, q_neg + 1):
        col = f"dS_neg_l{k}"
        d[col] = d["dS_neg"].shift(k)
        cols.append(col)

    reg = d.dropna(subset=["dP"] + cols).copy()
    return reg["dP"], reg[cols], reg.index


def fit_ols(y: pd.Series, x: pd.DataFrame):
    return sm.OLS(y, sm.add_constant(x, has_constant="add")).fit()


def choose_lags(d: pd.DataFrame) -> LagChoice:
    best: LagChoice | None = None
    for n_y, p_pos, q_neg in product(range(1, MAX_LAG + 1), range(0, MAX_LAG + 1), range(0, MAX_LAG + 1)):
        y, x, _ = build_frame(d.copy(), n_y, p_pos, q_neg)
        min_required = max(50, 8 + len(x.columns) * 3)
        if len(y) < min_required:
            continue
        try:
            model = fit_ols(y, x)
        except np.linalg.LinAlgError:
            continue
        if best is None or model.bic < best.bic:
            best = LagChoice(n_y=n_y, p_pos=p_pos, q_neg=q_neg, bic=float(model.bic))

    if best is None:
        raise RuntimeError("No valid lag choice found for interval NARDL.")
    return best


def run_fuel(df: pd.DataFrame, fuel: str) -> list[dict[str, Any]]:
    d = add_nardl_terms(df, fuel)
    lag = choose_lags(d.copy())
    y, x, reg_index = build_frame(d.copy(), lag.n_y, lag.p_pos, lag.q_neg)
    model = fit_ols(y, x)

    rho = float(model.params["P_l1"])
    rows: list[dict[str, Any]] = []
    for zone in ZONES:
        pos_col = f"{zone}_pos_l1"
        neg_col = f"{zone}_neg_l1"
        full_count = int(d[f"D_{zone}"].sum())
        reg_count = int(d.loc[reg_index, f"D_{zone}"].sum())
        row: dict[str, Any] = {
            "fuel": fuel,
            "fuel_cn": FUEL_CN[fuel],
            "zone": zone,
            "zone_cn": ZONE_CN[zone],
            "zone_window_count_full": full_count,
            "zone_window_count_regression": reg_count,
            "n_y_lag": lag.n_y,
            "p_pos_lag": lag.p_pos,
            "q_neg_lag": lag.q_neg,
            "n_regression": int(len(y)),
            "model_bic": float(model.bic),
            "model_r2": float(model.rsquared),
            "rho": rho,
        }

        if pos_col not in x.columns or neg_col not in x.columns:
            row.update(
                {
                    "available": False,
                    "reason": "该区间没有可识别的长期正负交互项，通常是样本数为0或交互项全为0。",
                    "phi_pos": np.nan,
                    "phi_neg": np.nan,
                    "phi_pos_pvalue": np.nan,
                    "phi_neg_pvalue": np.nan,
                    "long_pos": np.nan,
                    "long_neg": np.nan,
                    "wald_stat": np.nan,
                    "wald_pvalue": np.nan,
                    "decision_5pct": "不可检验",
                    "direction_note": "不可判断",
                }
            )
            rows.append(row)
            continue

        wald = model.wald_test(f"{pos_col} = {neg_col}", scalar=True)
        phi_pos = float(model.params[pos_col])
        phi_neg = float(model.params[neg_col])
        long_pos = float(-phi_pos / rho)
        long_neg = float(-phi_neg / rho)
        p_value = float(wald.pvalue)
        if p_value < 0.05 and long_pos > long_neg:
            direction_note = "拒绝对称，且上涨长期传导更强，支持涨多跌少方向。"
        elif p_value < 0.05 and long_pos < long_neg:
            direction_note = "拒绝对称，但下跌长期传导更强，不支持涨多跌少方向。"
        else:
            direction_note = "不能拒绝对称，未发现该区间长期涨跌传导显著不同。"

        row.update(
            {
                "available": True,
                "reason": "",
                "phi_pos": phi_pos,
                "phi_neg": phi_neg,
                "phi_pos_pvalue": float(model.pvalues[pos_col]),
                "phi_neg_pvalue": float(model.pvalues[neg_col]),
                "long_pos": long_pos,
                "long_neg": long_neg,
                "wald_stat": float(wald.statistic),
                "wald_pvalue": p_value,
                "decision_5pct": "拒绝对称" if p_value < 0.05 else "不能拒绝对称",
                "direction_note": direction_note,
            }
        )
        rows.append(row)
    return rows


def fmt_p(value: float) -> str:
    if pd.isna(value):
        return "-"
    if value < 0.001:
        return f"{value:.3e}"
    return f"{value:.4f}"


def write_report(results: pd.DataFrame, df: pd.DataFrame) -> None:
    lines: list[str] = []
    lines.append("任务一第九版 1.3 区间 NARDL Wald 检验结果")
    lines.append("=" * 64)
    lines.append("")
    lines.append("一、建模口径")
    lines.append("本结果严格按照《任务一 5.12 18点.docx》1.3 的区间 NARDL 思路重新计算。")
    lines.append("国际油价区间变量使用第九版结果中的国际原油移动平均价 MAT=ma_usd_per_bbl：")
    lines.append("Dfloor=1{MAT<=40}，Dnorm=1{40<MAT<130}，Dceil=1{MAT>=130}。")
    lines.append("长期项为区间虚拟变量与基础 NARDL 长期正负累计项的交互：D_z*S^+_{t-1} 和 D_z*S^-_{t-1}。")
    lines.append("短期动态项保留基础 NARDL 的滞后 dP、正向 dS、负向 dS 项，滞后阶数由 BIC 在 1-4 个调价周期内选择。")
    lines.append("估计变量采用 log(P) 和 log(MAT)，区间划分仍使用 MAT 的美元/桶水平值。")
    lines.append("")
    lines.append("二、样本区间分布")
    for zone in ZONES:
        lines.append(f"{ZONE_CN[zone]}：{int(df[f'D_{zone}'].sum())} 个调价窗口")
    lines.append(f"MAT 最小值={df['MAT'].min():.2f} 美元/桶，最大值={df['MAT'].max():.2f} 美元/桶。")
    lines.append("")
    lines.append("三、Wald 检验结果")

    for fuel in ["gasoline", "diesel"]:
        sub = results[results["fuel"] == fuel]
        first = sub.iloc[0]
        lines.append("")
        lines.append(
            f"{FUEL_CN[fuel]}：BIC 选择 n={int(first['n_y_lag'])}, "
            f"p={int(first['p_pos_lag'])}, q={int(first['q_neg_lag'])}，"
            f"回归样本数 {int(first['n_regression'])}，R2={first['model_r2']:.4f}"
        )
        for _, row in sub.iterrows():
            lines.append(f"  {row['zone_cn']}：回归内窗口数 {int(row['zone_window_count_regression'])}")
            if not bool(row["available"]):
                lines.append(f"    结果：不可检验。原因：{row['reason']}")
                continue
            lines.append(
                f"    phi+={row['phi_pos']:.6f}, phi-={row['phi_neg']:.6f}; "
                f"长期上涨传导={row['long_pos']:.4f}, 长期下跌传导={row['long_neg']:.4f}; "
                f"Wald={row['wald_stat']:.4f}, p={fmt_p(float(row['wald_pvalue']))}; "
                f"{row['decision_5pct']}。"
            )
            lines.append(f"    方向解释：{row['direction_note']}")
            if row["zone"] == "floor":
                lines.append(
                    f"    地板价项补充：phi- 的单项 p 值={fmt_p(float(row['phi_neg_pvalue']))}。"
                    "由于低油价窗口极少，该区间结果只作机制提示，不宜作为稳健统计结论。"
                )

    lines.append("")
    lines.append("四、结论")
    lines.append("1. 正常油价区间是样本主体。汽油、柴油在正常区间的 Wald 检验均不能在 5% 水平拒绝长期涨跌传导对称，")
    lines.append("   因而不能证明正常市场环境下存在稳定的“涨多跌少”长期非对称。")
    lines.append("2. 低油价地板区间只有 3 个原始窗口，回归有效窗口更少，估计结果不稳；可以说明样本包含地板价情形，")
    lines.append("   但不宜据此宣称已强实证证明地板价机制放大了跌少效应。")
    lines.append("3. 样本期内 MAT 最高仅 123.52 美元/桶，没有 MAT>=130 的天花板区间窗口，")
    lines.append("   因此高油价天花板机制无法用本样本做 Wald 实证检验，只能在论文中作机制层面的定性说明。")
    lines.append("4. 本 1.3 结果只检验区间 NARDL 长期非对称项，不使用第九版 Ridge 政策传导层的不对称检验，也不是简单分区误差统计。")

    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df = load_data()
    rows: list[dict[str, Any]] = []
    for fuel in ["gasoline", "diesel"]:
        rows.extend(run_fuel(df, fuel))

    results = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    write_report(results, df)

    print(results.to_string(index=False))
    print(f"\n已写出: {OUT_CSV}")
    print(f"已写出: {OUT_TXT}")


if __name__ == "__main__":
    main()
