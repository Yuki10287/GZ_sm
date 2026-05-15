"""任务一 1.2：NARDL 非对称检验改进实验。

本脚本用于回应队友提出的三类改进建议：

1. 扩大 NARDL 最大滞后阶数：
   对 max_lag = 4、8、12 分别用 BIC 重新选择 (p, q)，观察长期非对称是否显现。

2. 检查卡尔曼滤波是否过度平滑：
   对比三类国际油价代理变量：
   - kalman_index：第九版状态空间综合油价窗口均值；
   - raw_equal_spot：Brent、WTI、Dubai 原始现货价格等权窗口均值；
   - raw_equal_spot_ma3：raw_equal_spot 的 3 个调价窗口轻度平滑值。

3. Threshold NARDL：
   将正负冲击划分门限从 0 扩展为 1%、2%、3%。
   即只有当 Δlog(X_t) > threshold 时计入正向冲击，
   当 Δlog(X_t) < -threshold 时计入负向冲击，微小波动记为 0。

输出：
    outputs/task1_serial_fusion_ridge/nardl_robustness_experiments.csv
    任务一第九版_NARDL稳健性改进实验结果.txt
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
OUT_CSV = ROOT / "outputs" / "task1_serial_fusion_ridge" / "nardl_robustness_experiments.csv"
OUT_TXT = ROOT / "任务一第九版_NARDL稳健性改进实验结果.txt"


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


def load_base_data() -> pd.DataFrame:
    """读取第九版验证表，并补充原始现货油价窗口均值。"""
    validation = pd.read_csv(VALIDATION_PATH, parse_dates=["adjust_date", "window_start", "window_end"])
    validation = validation.sort_values("adjust_date").reset_index(drop=True)

    state = pd.read_csv(STATE_PATH, parse_dates=["date"]).sort_values("date")
    for col in ["brent", "wti", "dubai"]:
        state[col] = pd.to_numeric(state[col], errors="coerce")

    # 原始现货价格不做卡尔曼滤波。WTI 2020-04-20 负值不能进入对数，
    # 因此在构造原始等权价格时仅把非正数当作缺失，使用当日其他有效油种均值。
    raw_prices = state[["brent", "wti", "dubai"]].mask(lambda x: x <= 0)
    state["raw_equal_spot_daily"] = raw_prices.mean(axis=1, skipna=True)

    raw_window_values = []
    for _, row in validation.iterrows():
        mask = (state["date"] >= row["window_start"]) & (state["date"] <= row["window_end"])
        raw_window_values.append(state.loc[mask, "raw_equal_spot_daily"].mean())

    validation["raw_equal_spot"] = raw_window_values
    validation["raw_equal_spot"] = validation["raw_equal_spot"].interpolate(limit_direction="both").ffill().bfill()
    validation["raw_equal_spot_ma3"] = validation["raw_equal_spot"].rolling(window=3, min_periods=1).mean()

    # 第九版原有的状态空间综合油价窗口均值。
    validation["kalman_index"] = validation["ma_usd_per_bbl"]
    return validation


def add_nardl_decomposition(df: pd.DataFrame, oil_proxy: str, threshold: float) -> pd.DataFrame:
    """构造 Threshold NARDL 的正负累计冲击项。"""
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
    """构造 NARDL 误差修正形式的回归数据。"""
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
    """计算 R beta = 0 线性约束下的 OLS RSS。"""
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

    middle = np.linalg.pinv(r_mat @ xtx_inv @ r_mat.T)
    beta_r = beta_u - xtx_inv @ r_mat.T @ middle @ (r_mat @ beta_u)
    resid = y_vec - x_mat @ beta_r
    return float(resid @ resid)


def lr_test(n: int, rss_r: float, rss_u: float, df_restriction: int) -> tuple[float, float]:
    stat = n * np.log(rss_r / rss_u)
    return float(stat), float(chi2.sf(stat, df_restriction))


def fit_one_setting(df: pd.DataFrame, fuel: str, oil_proxy: str, threshold: float, max_lag: int) -> NardlResult | None:
    """在一个油价代理、门限和最大滞后设定下选择 BIC 最优 NARDL 并做 LR 检验。"""
    d = add_nardl_decomposition(df, oil_proxy, threshold)
    best = None
    for p, q in product(range(1, max_lag + 1), range(0, max_lag + 1)):
        y, x = build_nardl_frame(d, fuel, p, q)
        min_required = max(60, 8 + len(x.columns) * 3)
        if len(y) < min_required:
            continue
        try:
            model = fit_ols(y, x)
        except np.linalg.LinAlgError:
            continue
        if best is None or model.bic < best["bic"]:
            best = {"p": p, "q": q, "y": y, "x": x, "model": model, "bic": float(model.bic)}

    if best is None:
        return None

    y = best["y"]
    x = best["x"]
    model = best["model"]
    p = best["p"]
    q = best["q"]
    n = len(y)
    rss_u = float(np.sum(model.resid**2))

    short_constraint = {}
    for j in range(q + 1):
        short_constraint[f"dxpos_l{j}"] = 1.0
        short_constraint[f"dxneg_l{j}"] = -1.0
    long_constraint = {"xpos_l1": 1.0, "xneg_l1": -1.0}

    lr_short, p_short = lr_test(n, restricted_rss(y, x, [short_constraint]), rss_u, 1)
    lr_long, p_long = lr_test(n, restricted_rss(y, x, [long_constraint]), rss_u, 1)
    lr_both, p_both = lr_test(n, restricted_rss(y, x, [short_constraint, long_constraint]), rss_u, 2)

    rho = model.params["y_l1"]
    long_pos = -model.params["xpos_l1"] / rho
    long_neg = -model.params["xneg_l1"] / rho
    short_pos = sum(model.params.get(f"dxpos_l{j}", 0.0) for j in range(q + 1))
    short_neg = sum(model.params.get(f"dxneg_l{j}", 0.0) for j in range(q + 1))

    return NardlResult(
        oil_proxy=oil_proxy,
        fuel=fuel,
        threshold=threshold,
        max_lag=max_lag,
        p=p,
        q=q,
        n=n,
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


def summarize(results: pd.DataFrame) -> str:
    """生成中文总结。"""
    lines = []
    lines.append("任务一 1.2 NARDL 稳健性改进实验结果")
    lines.append("=" * 60)
    lines.append("")
    lines.append("实验设置：")
    lines.append("1. 最大滞后阶数分别设为 4、8、12，并用 BIC 自动选择最优 (p,q)。")
    lines.append("2. 对比卡尔曼综合油价、原始三油种等权现货价、原始现货价3期轻度平滑。")
    lines.append("3. 对比传统0门限与 1%、2%、3% 门限 Threshold NARDL。")
    lines.append("")

    for fuel in ["gasoline", "diesel"]:
        cn = "汽油" if fuel == "gasoline" else "柴油"
        sub = results[results["fuel"] == fuel].copy()
        sig_short = sub[sub["p_short"] < 0.05].sort_values("p_short")
        sig_long = sub[sub["p_long"] < 0.05].sort_values("p_long")
        lines.append(f"{cn}：")
        lines.append(f"  共尝试 {len(sub)} 个设定。")
        lines.append(f"  短期非对称显著设定数：{len(sig_short)}。")
        lines.append(f"  长期非对称显著设定数：{len(sig_long)}。")
        if not sig_short.empty:
            r = sig_short.iloc[0]
            lines.append(
                "  短期最显著设定："
                f"oil_proxy={r['oil_proxy']}, threshold={r['threshold']:.2%}, max_lag={int(r['max_lag'])}, "
                f"(p,q)=({int(r['p'])},{int(r['q'])}), p_short={r['p_short']:.4g}, "
                f"short_pos={r['short_pos']:.3f}, short_neg={r['short_neg']:.3f}。"
            )
        if not sig_long.empty:
            r = sig_long.iloc[0]
            lines.append(
                "  长期最显著设定："
                f"oil_proxy={r['oil_proxy']}, threshold={r['threshold']:.2%}, max_lag={int(r['max_lag'])}, "
                f"(p,q)=({int(r['p'])},{int(r['q'])}), p_long={r['p_long']:.4g}, "
                f"long_pos={r['long_pos']:.3f}, long_neg={r['long_neg']:.3f}。"
            )
        else:
            lines.append("  没有发现 5% 水平下显著的长期非对称设定。")
        lines.append("")

    lines.append("完整结果见 CSV，可按 p_short、p_long 或 p_both 排序挑选论文表格。")
    return "\n".join(lines)


def main() -> None:
    df = load_base_data()
    oil_proxies = ["kalman_index", "raw_equal_spot", "raw_equal_spot_ma3"]
    thresholds = [0.0, 0.01, 0.02, 0.03]
    max_lags = [4, 8, 12]

    rows: list[dict[str, float | int | str]] = []
    for oil_proxy, threshold, max_lag, fuel in product(oil_proxies, thresholds, max_lags, ["gasoline", "diesel"]):
        result = fit_one_setting(df, fuel, oil_proxy, threshold, max_lag)
        if result is not None:
            rows.append(result.__dict__)

    results = pd.DataFrame(rows).sort_values(["fuel", "p_long", "p_short", "oil_proxy", "threshold", "max_lag"])
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    OUT_TXT.write_text(summarize(results), encoding="utf-8")

    print(results.to_string(index=False))
    print(f"\n已写出: {OUT_CSV}")
    print(f"已写出: {OUT_TXT}")


if __name__ == "__main__":
    main()
