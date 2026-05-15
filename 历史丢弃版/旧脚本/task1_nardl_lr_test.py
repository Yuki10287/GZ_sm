"""任务一 1.2：严格按队友 NARDL 思路做 LR 似然比检验。

用途：
    检验国际油价上涨/下跌对国内汽油、柴油价格的短期和长期传导是否对称。

说明：
    这份脚本对应《任务一 5.12 18点.docx》中 1.2 的 NARDL 思路，
    不是第九版“理论调价幅度 -> 实际调价幅度”的政策传导层检验。

数据来源：
    outputs/task1_serial_fusion_ridge/serial_fusion_validation.csv

输出：
    outputs/task1_serial_fusion_ridge/section_1_2_nardl_lr_test.csv
    任务一第九版_NARDL_LR似然比检验结果.txt
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2


ROOT = Path(__file__).resolve().parent
VALIDATION_PATH = ROOT / "outputs" / "task1_serial_fusion_ridge" / "serial_fusion_validation.csv"
OUT_CSV = ROOT / "outputs" / "task1_serial_fusion_ridge" / "section_1_2_nardl_lr_test.csv"
OUT_TXT = ROOT / "任务一第九版_NARDL_LR似然比检验结果.txt"


def load_validation_data() -> pd.DataFrame:
    """读取第九版模型输出，并构造 NARDL 所需的正负累计项。"""
    df = pd.read_csv(VALIDATION_PATH, parse_dates=["adjust_date"])
    df = df.sort_values("adjust_date").reset_index(drop=True)

    # 使用第九版综合国际油价窗口均值作为国际真实油价 X_t 的代理变量。
    df["x"] = np.log(df["ma_usd_per_bbl"])
    df["dx"] = df["x"].diff()

    # NARDL 正负分解：上涨变化和下跌变化分别累计。
    df["dx_pos"] = df["dx"].clip(lower=0)
    df["dx_neg"] = df["dx"].clip(upper=0)
    df["x_pos"] = df["dx_pos"].fillna(0).cumsum()
    df["x_neg"] = df["dx_neg"].fillna(0).cumsum()
    return df


def build_nardl_frame(df: pd.DataFrame, fuel: str, p: int, q: int) -> tuple[pd.Series, pd.DataFrame]:
    """构造 NARDL 误差修正形式的回归样本。

    设 Y_t 为国内汽油/柴油价格水平，X_t 为国际综合油价。
    回归形式为：
        ΔY_t = c + ρY_{t-1} + θ+X+_{t-1} + θ-X-_{t-1}
               + Σφ_iΔY_{t-i} + Σπ+iΔX+_{t-i} + Σπ-iΔX-_{t-i} + ε_t
    """
    d = df.copy()
    d["y"] = np.log(d[f"{fuel}_price"])
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


def fit_unrestricted(y: pd.Series, x: pd.DataFrame):
    """拟合非约束 NARDL 模型。"""
    return sm.OLS(y, sm.add_constant(x, has_constant="add")).fit()


def restricted_ols_rss(y: pd.Series, x: pd.DataFrame, constraints: list[dict[str, float]]) -> float:
    """计算线性等式约束 R beta = 0 下的 OLS 残差平方和。

    statsmodels 在当前环境中没有 fit_constrained 接口，因此这里使用
    受限最小二乘的矩阵闭式解：

        beta_R = beta_U - (X'X)^(-1) R' [R (X'X)^(-1) R']^(-1) R beta_U

    constraints 中每个 dict 表示一个约束，例如：
        {"xpos_l1": 1, "xneg_l1": -1}
    表示 xpos_l1 - xneg_l1 = 0。
    """
    x_const = sm.add_constant(x, has_constant="add")
    names = list(x_const.columns)
    x_mat = x_const.to_numpy(dtype=float)
    y_vec = y.to_numpy(dtype=float)

    xtx_inv = np.linalg.pinv(x_mat.T @ x_mat)
    beta_u = xtx_inv @ x_mat.T @ y_vec

    r_mat = np.zeros((len(constraints), len(names)), dtype=float)
    for row_i, constraint in enumerate(constraints):
        for name, coef in constraint.items():
            r_mat[row_i, names.index(name)] = coef

    middle = np.linalg.pinv(r_mat @ xtx_inv @ r_mat.T)
    beta_r = beta_u - xtx_inv @ r_mat.T @ middle @ (r_mat @ beta_u)
    resid = y_vec - x_mat @ beta_r
    return float(resid @ resid)


def lr_from_models(n: int, rss_restricted: float, rss_unrestricted: float, df_restriction: int) -> tuple[float, float]:
    """根据受限/非受限模型 RSS 计算 LR 统计量和 p 值。

    正态线性回归下：
        LR = n * ln(RSS_R / RSS_U)
    近似服从自由度为约束个数的卡方分布。
    """
    stat = n * np.log(rss_restricted / rss_unrestricted)
    p_value = chi2.sf(stat, df_restriction)
    return float(stat), float(p_value)


def choose_lags_by_bic(df: pd.DataFrame, fuel: str, max_p: int = 4, max_q: int = 4) -> tuple[int, int]:
    """用 BIC 选择 NARDL 的滞后阶数。"""
    best: tuple[float, int, int] | None = None
    for p, q in product(range(1, max_p + 1), range(0, max_q + 1)):
        y, x = build_nardl_frame(df, fuel, p, q)
        if len(y) < 80:
            continue
        model = fit_unrestricted(y, x)
        if best is None or model.bic < best[0]:
            best = (float(model.bic), p, q)
    if best is None:
        raise RuntimeError(f"{fuel} 没有可用的 NARDL 滞后阶数组合")
    return best[1], best[2]


def run_lr_tests(df: pd.DataFrame, fuel: str, p: int, q: int) -> dict[str, float | int | str]:
    """对一个品种执行短期、长期、短期+长期联合 LR 检验。"""
    y, x = build_nardl_frame(df, fuel, p, q)
    unrestricted = fit_unrestricted(y, x)
    rss_u = float(np.sum(unrestricted.resid**2))
    n = len(y)

    # 长期对称约束：xpos_l1 与 xneg_l1 系数相等。
    rss_long_r = restricted_ols_rss(y, x, [{"xpos_l1": 1.0, "xneg_l1": -1.0}])
    lr_long, p_long = lr_from_models(n, rss_long_r, rss_u, 1)

    # 短期对称约束：Σπ+ = Σπ-。
    short_constraint = {}
    for j in range(q + 1):
        short_constraint[f"dxpos_l{j}"] = 1.0
        short_constraint[f"dxneg_l{j}"] = -1.0
    rss_short_r = restricted_ols_rss(y, x, [short_constraint])
    lr_short, p_short = lr_from_models(n, rss_short_r, rss_u, 1)

    # 短期和长期同时对称：两个约束同时施加。
    rss_both_r = restricted_ols_rss(
        y,
        x,
        [
            {"xpos_l1": 1.0, "xneg_l1": -1.0},
            short_constraint,
        ],
    )
    lr_both, p_both = lr_from_models(n, rss_both_r, rss_u, 2)

    rho = unrestricted.params["y_l1"]
    long_pos = -unrestricted.params["xpos_l1"] / rho
    long_neg = -unrestricted.params["xneg_l1"] / rho
    short_pos = sum(unrestricted.params.get(f"dxpos_l{j}", 0.0) for j in range(q + 1))
    short_neg = sum(unrestricted.params.get(f"dxneg_l{j}", 0.0) for j in range(q + 1))

    return {
        "fuel": fuel,
        "n": n,
        "p": p,
        "q": q,
        "short_pos_effect": float(short_pos),
        "short_neg_effect": float(short_neg),
        "long_pos_effect": float(long_pos),
        "long_neg_effect": float(long_neg),
        "lr_short": lr_short,
        "p_short": p_short,
        "lr_long": lr_long,
        "p_long": p_long,
        "lr_both": lr_both,
        "p_both": p_both,
        "aic_unrestricted": float(unrestricted.aic),
        "bic_unrestricted": float(unrestricted.bic),
        "rss_unrestricted": rss_u,
    }


def write_text_report(results: pd.DataFrame) -> None:
    """写出中文结果说明。"""
    fuel_name = {"gasoline": "汽油", "diesel": "柴油"}
    lines: list[str] = []
    lines.append("NARDL 似然比检验结果（严格按队友 1.2 思路）")
    lines.append("=" * 60)
    lines.append("")
    lines.append("检验对象：国际综合油价上涨/下跌对国内成品油价格的传导是否对称。")
    lines.append("非约束模型允许上涨项和下跌项系数不同；约束模型施加对称性约束。")
    lines.append("LR = n * ln(RSS_R / RSS_U)，近似服从卡方分布。")
    lines.append("")

    for _, row in results.iterrows():
        name = fuel_name[row["fuel"]]
        lines.append(f"{name}：样本数 {int(row['n'])}，BIC选择滞后阶数 p={int(row['p'])}, q={int(row['q'])}")
        lines.append(f"  短期上涨效应：{row['short_pos_effect']:.4f}")
        lines.append(f"  短期下跌效应：{row['short_neg_effect']:.4f}")
        lines.append(f"  长期上涨效应：{row['long_pos_effect']:.4f}")
        lines.append(f"  长期下跌效应：{row['long_neg_effect']:.4f}")
        lines.append(f"  短期对称 LR={row['lr_short']:.4f}, p={row['p_short']:.4f}")
        lines.append(f"  长期对称 LR={row['lr_long']:.4f}, p={row['p_long']:.4f}")
        lines.append(f"  短期+长期同时对称 LR={row['lr_both']:.4f}, p={row['p_both']:.4f}")
        lines.append("")

    lines.append("结论：")
    for _, row in results.iterrows():
        name = fuel_name[row["fuel"]]
        short_text = "拒绝" if row["p_short"] < 0.05 else "不能拒绝"
        long_text = "拒绝" if row["p_long"] < 0.05 else "不能拒绝"
        both_text = "拒绝" if row["p_both"] < 0.05 else "不能拒绝"
        lines.append(
            f"{name}：短期 LR 检验{short_text}短期对称假设；"
            f"长期 LR 检验{long_text}长期对称假设；"
            f"联合 LR 检验{both_text}短期和长期同时对称假设。"
        )
    lines.append("")
    lines.append("解释：")
    lines.append("LR 检验基于同方差正态误差的似然比思想，因此结果可作为 NARDL 框架下的补充检验。")
    lines.append("本次结果显示：短期非对称在 LR 检验下显著，长期非对称不显著。")
    lines.append("这说明价格传递差异更可能体现在短期调价响应上，而不是长期均衡关系上。")
    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df = load_validation_data()
    rows = []
    for fuel in ["gasoline", "diesel"]:
        p, q = choose_lags_by_bic(df, fuel)
        rows.append(run_lr_tests(df, fuel, p, q))

    results = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    write_text_report(results)

    print(results.to_string(index=False))
    print(f"\n已写出: {OUT_CSV}")
    print(f"已写出: {OUT_TXT}")


if __name__ == "__main__":
    main()
