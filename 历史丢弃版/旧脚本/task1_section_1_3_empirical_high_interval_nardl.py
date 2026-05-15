"""Task 1.3 empirical high-price interval NARDL.

Why this script exists:
    The strict docx threshold MAT>=130 has zero observations in the ninth-version
    sample, so the ceiling interval cannot be estimated. This script keeps the
    policy floor threshold MAT<=40, but defines an empirical high-price pressure
    interval using MAT>=100 for estimation, and reports sensitivity for several
    nearby high-price cutoffs.

This is not a replacement for the legal ceiling-price mechanism. It is an
empirical supplement for describing pricing behavior when oil prices are high
inside the observed sample.
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
OUT_CSV = ROOT / "outputs" / "task1_serial_fusion_ridge" / "section_1_3_empirical_high_interval_nardl.csv"
OUT_TXT = ROOT / "任务一第九版_1.3经验高油价区间NARDL结果.txt"

MAX_LAG = 4
MAIN_HIGH_CUTOFF = 115.0
SENSITIVITY_CUTOFFS = [90.0, 95.0, 100.0, 105.0, 110.0, 115.0]
ZONES = ["floor", "norm", "high"]
FUEL_CN = {"gasoline": "汽油", "diesel": "柴油"}


@dataclass(frozen=True)
class LagChoice:
    n_y: int
    p_pos: int
    q_neg: int
    bic: float


def load_base() -> pd.DataFrame:
    df = pd.read_csv(INPUT_CSV, parse_dates=["adjust_date"])
    df = df.sort_values("adjust_date").reset_index(drop=True)
    df["MAT"] = pd.to_numeric(df["ma_usd_per_bbl"], errors="coerce")
    return df


def add_zone_dummies(df: pd.DataFrame, high_cutoff: float) -> pd.DataFrame:
    d = df.copy()
    d["D_floor"] = (d["MAT"] <= 40).astype(int)
    d["D_high"] = (d["MAT"] >= high_cutoff).astype(int)
    d["D_norm"] = ((d["MAT"] > 40) & (d["MAT"] < high_cutoff)).astype(int)
    return d


def add_nardl_terms(df: pd.DataFrame, fuel: str) -> pd.DataFrame:
    d = df.copy()
    d["P"] = np.log(pd.to_numeric(d[f"{fuel}_price"], errors="coerce"))
    d["S"] = np.log(pd.to_numeric(d["MAT"], errors="coerce"))
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
        raise RuntimeError("No valid lag choice found.")
    return best


def zone_label(zone: str, high_cutoff: float) -> str:
    if zone == "floor":
        return "低油价区间（MAT<=40）"
    if zone == "norm":
        return f"中常油价区间（40<MAT<{high_cutoff:g}）"
    return f"经验高油价压力区间（MAT>={high_cutoff:g}）"


def run_one(df: pd.DataFrame, fuel: str, high_cutoff: float) -> list[dict[str, Any]]:
    d = add_nardl_terms(add_zone_dummies(df, high_cutoff), fuel)
    lag = choose_lags(d.copy())
    y, x, reg_index = build_frame(d.copy(), lag.n_y, lag.p_pos, lag.q_neg)
    model = fit_ols(y, x)
    rho = float(model.params["P_l1"])
    rows: list[dict[str, Any]] = []

    for zone in ZONES:
        pos_col = f"{zone}_pos_l1"
        neg_col = f"{zone}_neg_l1"
        row: dict[str, Any] = {
            "high_cutoff": high_cutoff,
            "fuel": fuel,
            "fuel_cn": FUEL_CN[fuel],
            "zone": zone,
            "zone_cn": zone_label(zone, high_cutoff),
            "zone_window_count_full": int(d[f"D_{zone}"].sum()),
            "zone_window_count_regression": int(d.loc[reg_index, f"D_{zone}"].sum()),
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
                    "phi_pos": np.nan,
                    "phi_neg": np.nan,
                    "long_pos": np.nan,
                    "long_neg": np.nan,
                    "wald_stat": np.nan,
                    "wald_pvalue": np.nan,
                    "decision_5pct": "不可检验",
                    "direction_note": "样本不足或变量不可识别。",
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
            note = "拒绝对称，且上涨传导更强。"
        elif p_value < 0.05 and long_pos < long_neg:
            note = "拒绝对称，且下跌传导更强。"
        else:
            note = "不能拒绝对称。"
        row.update(
            {
                "available": True,
                "phi_pos": phi_pos,
                "phi_neg": phi_neg,
                "long_pos": long_pos,
                "long_neg": long_neg,
                "wald_stat": float(wald.statistic),
                "wald_pvalue": p_value,
                "decision_5pct": "拒绝对称" if p_value < 0.05 else "不能拒绝对称",
                "direction_note": note,
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
    lines.append("任务一第九版 1.3 经验高油价压力区间 NARDL 结果")
    lines.append("=" * 66)
    lines.append("")
    lines.append("一、为什么调整 130 美元阈值")
    lines.append("严格制度口径下，高油价天花板区间为 MAT>=130。但第九版样本中 MAT 最大值为 123.52，")
    lines.append("因此 Dceil 全为 0，无法估计天花板区间的 NARDL 交互项。")
    lines.append("为分析样本内高油价阶段的定价表现，本文新增“经验高油价压力区间”，主口径按队友建议取 MAT>=115。")
    lines.append("该阈值不是替代政策天花板，而是样本内高油价压力的经验划分：")
    lines.append("115 美元/桶更接近样本最高油价和 130 美元制度天花板，但样本内仅有 3 个窗口，因此结果应作为补充性区间检验谨慎解释。")
    lines.append("")
    lines.append("不同高油价阈值下样本量：")
    for cutoff in SENSITIVITY_CUTOFFS:
        lines.append(f"MAT>={cutoff:g}：{int((df['MAT'] >= cutoff).sum())} 个窗口")
    lines.append("")
    lines.append("二、主口径：MAT>=115 的区间 NARDL")
    main = results[results["high_cutoff"] == MAIN_HIGH_CUTOFF]
    for fuel in ["gasoline", "diesel"]:
        sub = main[main["fuel"] == fuel]
        first = sub.iloc[0]
        lines.append("")
        lines.append(
            f"{FUEL_CN[fuel]}：BIC 选择 n={int(first['n_y_lag'])}, p={int(first['p_pos_lag'])}, "
            f"q={int(first['q_neg_lag'])}，回归样本数 {int(first['n_regression'])}，R2={first['model_r2']:.4f}"
        )
        for _, row in sub.iterrows():
            lines.append(f"  {row['zone_cn']}：窗口数 {int(row['zone_window_count_full'])}")
            if not bool(row["available"]):
                lines.append("    不可检验。")
                continue
            lines.append(
                f"    长期上涨传导={row['long_pos']:.4f}，长期下跌传导={row['long_neg']:.4f}，"
                f"Wald={row['wald_stat']:.4f}，p={fmt_p(float(row['wald_pvalue']))}，{row['decision_5pct']}。"
            )
            lines.append(f"    方向：{row['direction_note']}")

    lines.append("")
    lines.append("三、敏感性：高油价压力阈值变化")
    for cutoff in SENSITIVITY_CUTOFFS:
        sub_cut = results[(results["high_cutoff"] == cutoff) & (results["zone"] == "high")]
        lines.append(f"")
        lines.append(f"高油价阈值 MAT>={cutoff:g}：")
        for _, row in sub_cut.iterrows():
            lines.append(
                f"  {row['fuel_cn']}：窗口数 {int(row['zone_window_count_full'])}，"
                f"长期上涨={row['long_pos']:.4f}，长期下跌={row['long_neg']:.4f}，"
                f"p={fmt_p(float(row['wald_pvalue']))}，{row['decision_5pct']}。"
            )

    lines.append("")
    lines.append("四、结论写法")
    lines.append("1. 法定天花板区间 MAT>=130 在样本中仍然没有观测，不能宣称已经实证检验天花板价机制。")
    lines.append("2. 将高油价压力区间调整为 MAT>=115 后，可以观察最接近天花板价的样本内高油价窗口，")
    lines.append("   但该区间仅有 3 个窗口，统计结论不能过度外推。")
    lines.append("3. 主口径下，汽油和柴油在经验高油价压力区间的 Wald 结果仅能作为补充证据，")
    lines.append("   更适合与 MAT>=100、105、110 等阈值敏感性结果共同说明。")
    lines.append("5. 论文中应把该部分命名为“经验高油价压力区间分析”或“邻近天花板区间的补充检验”，")
    lines.append("   与法定 MAT>=130 天花板机制区分开。")
    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df = load_base()
    rows: list[dict[str, Any]] = []
    for cutoff, fuel in product(SENSITIVITY_CUTOFFS, ["gasoline", "diesel"]):
        rows.extend(run_one(df, fuel, cutoff))
    results = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    write_report(results, df)
    print(results.to_string(index=False))
    print(f"\n已写出: {OUT_CSV}")
    print(f"已写出: {OUT_TXT}")


if __name__ == "__main__":
    main()
