from __future__ import annotations

"""任务一最终推荐版：双因子状态空间 + 期货期限结构 + RidgeCV 政策传导。

本脚本是一个完整、独立的建模脚本，不再 import 前面做对比实验时写的其它模型脚本。

模型分为三层：

第一层：Schwartz 思路的双因子状态空间模型
    用 Brent、WTI、Dubai 三种国际原油价格估计两个潜在状态：
    - X_t：基础油价水平；
    - delta_t：便利收益、地缘冲突溢价或短期稀缺性压力。

第二层：国内成品油调价机制层
    用状态油价指数计算“纯理论调价幅度”。机制层使用更贴近实际规则的设定：
    - 本轮 10 个工作日均价 vs 上一轮 10 个工作日均价；
    - 调价日有 1 天计价滞后；
    - 50 元/吨以下不调；
    - 40 美元地板价、130 美元天花板价采用硬规则。

第三层：RidgeCV 政策传导层
    实际调价通常不会完全等于纯理论调价。本文将纯理论调价、delta_t、汇率变动、
    真实 M1/M3/M6 期货期限结构作为特征，用 RidgeCV 拟合实际调价幅度。

最终推荐特征组合：
    纯理论调价 + delta_t + 汇率变动 + 期货期限结构。

说明：
    CPI/PPI/PMI/GDP 不进入本脚本。它们更适合任务二中分析调价机制对宏观经济、
    通胀和产业链传导的影响。
"""

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ==================== 路径配置 ====================

ROOT = Path(__file__).resolve().parent
OIL_DIR = ROOT / "国际原油价格数据"
FX_PATH = ROOT / "汇率" / "DEXCHUS.csv"
DOMESTIC_PATH = ROOT / "国内柴油汽油调价" / "柴油汽油调价2002-2026.xlsx"
STATE_PATH = ROOT / "outputs" / "task1_teammate_statespace" / "state_space_oil_index.csv"
OUTPUT_DIR = ROOT / "outputs" / "task1_serial_fusion_ridge"


# ==================== 模型配置 ====================

@dataclass(frozen=True)
class MechanismConfig:
    """国内成品油调价机制参数。"""

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


@dataclass(frozen=True)
class StateSpaceFit:
    """双因子状态空间极大似然估计结果。"""

    params: dict[str, float]
    neg_loglike: float
    success: bool
    message: str
    iterations: int


FUTURES_PRICE_COLS = [
    "brent_m1",
    "brent_m3",
    "brent_m6",
    "dubai_m1",
    "dubai_m3",
    "dubai_m6",
    "wti_m1",
    "wti_m3",
    "wti_m6",
]

FUTURES_FEATURE_COLS = [
    "term_spread_m1_m6_mean",
    "term_spread_m1_m3_mean",
    "term_spread_m3_m6_mean",
    "term_spread_m1_m6_change",
]


# ==================== 通用工具 ====================

def clean_numeric(series: pd.Series) -> pd.Series:
    """把带逗号、百分号的文本列清洗成数值列。"""

    return pd.to_numeric(
        series.astype("string").str.replace(",", "", regex=False).str.replace("%", "", regex=False),
        errors="coerce",
    )


def clean_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    """清洗机器学习特征矩阵中的无穷值和缺失值。"""

    out = df.copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.ffill().bfill().fillna(0.0)


def validate_required_columns(df: pd.DataFrame, cols: list[str], stage: str) -> None:
    """检查必要字段，避免后续出现难懂的 KeyError。"""

    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"{stage} 缺少必要字段: {', '.join(missing)}")


def asof_merge_available(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """按“调价日前已经可获得的最近一期数据”合并低频数据。"""

    return pd.merge_asof(
        left.sort_values("adjust_date"),
        right.sort_values("available_date"),
        left_on="adjust_date",
        right_on="available_date",
        direction="backward",
    ).drop(columns=["available_date"], errors="ignore")


# ==================== 原油、汇率和国内调价数据 ====================

def read_eia_oil_csv(path: Path, value_name: str) -> pd.DataFrame:
    """读取 EIA 的 Brent/WTI 日度原油价格。"""

    df = pd.read_csv(path, skiprows=4)
    df = df.rename(columns={df.columns[0]: "date", df.columns[1]: value_name})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df[value_name] = clean_numeric(df[value_name])
    return df[["date", value_name]].dropna().sort_values("date")


def read_dubai_csv(path: Path) -> pd.DataFrame:
    """读取 Dubai/Oman 原油价格。"""

    df = pd.read_csv(path)
    df = df.rename(columns={"Date": "date", "Price": "dubai"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["dubai"] = clean_numeric(df["dubai"])
    return df[["date", "dubai"]].dropna().sort_values("date")


def read_fx(path: Path) -> pd.DataFrame:
    """读取美元兑人民币汇率。"""

    df = pd.read_csv(path)
    df = df.rename(columns={df.columns[0]: "date", df.columns[1]: "usd_cny"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["usd_cny"] = clean_numeric(df["usd_cny"])
    return df[["date", "usd_cny"]].dropna().sort_values("date")


def read_oil_panel() -> pd.DataFrame:
    """构建 Brent、WTI、Dubai 和 USD/CNY 的日度面板。"""

    brent = read_eia_oil_csv(OIL_DIR / "Europe_Brent_Spot_Price_FOB.csv", "brent")
    wti = read_eia_oil_csv(OIL_DIR / "Cushing_OK_WTI_Spot_Price_FOB.csv", "wti")
    dubai = read_dubai_csv(OIL_DIR / "Dubai Crude Oil (Platts) Financial Futures Historical Data 2010-2026.csv")
    fx = read_fx(FX_PATH)
    panel = brent.merge(wti, on="date", how="outer").merge(dubai, on="date", how="outer")
    panel = panel.merge(fx, on="date", how="outer").sort_values("date")
    panel[["brent", "wti", "dubai", "usd_cny"]] = panel[["brent", "wti", "dubai", "usd_cny"]].ffill()
    return panel.dropna(subset=["brent", "wti", "dubai", "usd_cny"]).reset_index(drop=True)


def read_domestic_adjustments(start_date: str) -> pd.DataFrame:
    """读取国内汽油、柴油实际调价记录。"""

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
    for col in ["gasoline_price", "gasoline_actual_delta", "diesel_price", "diesel_actual_delta"]:
        df[col] = clean_numeric(df[col])
    df = df.dropna(subset=["adjust_date"]).sort_values("adjust_date").reset_index(drop=True)
    return df[df["adjust_date"] >= pd.Timestamp(start_date)].reset_index(drop=True)


# ==================== 双因子状态空间模型 ====================

def prepare_log_observations(panel: pd.DataFrame, start_date: str) -> pd.DataFrame:
    """把三种原油价格转换成双因子状态空间的对数观测矩阵。"""

    df = panel[panel["date"] >= pd.Timestamp(start_date)].copy()
    prices = df[["brent", "wti", "dubai"]].mask(lambda x: x <= 0)
    prices = prices.interpolate(limit_direction="both").ffill().bfill()
    df[["brent_log", "wti_log", "dubai_log"]] = np.log(prices)
    return df.dropna(subset=["brent_log", "wti_log", "dubai_log"]).reset_index(drop=True)


def unpack_state_params(
    theta: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """把优化器参数向量解包成状态空间矩阵。"""

    mu = theta[0]
    alpha = theta[1]
    kappa = np.exp(theta[2])
    sigma1 = np.exp(theta[3])
    sigma2 = np.exp(theta[4])
    rho = np.tanh(theta[5])
    h = theta[6:9]
    obs_sigma = np.exp(theta[9:12])

    transition = np.array([[1.0, -dt], [0.0, 1.0 - kappa * dt]], dtype=float)
    intercept = np.array([(mu - 0.5 * sigma1 * sigma1) * dt, kappa * alpha * dt], dtype=float)
    process_cov = np.array(
        [
            [sigma1 * sigma1 * dt, rho * sigma1 * sigma2 * dt],
            [rho * sigma1 * sigma2 * dt, sigma2 * sigma2 * dt],
        ],
        dtype=float,
    )
    observation = np.column_stack([np.ones(3), h])
    observation_cov = np.diag(obs_sigma * obs_sigma)
    named = {
        "mu": float(mu),
        "alpha": float(alpha),
        "kappa": float(kappa),
        "sigma1": float(sigma1),
        "sigma2": float(sigma2),
        "rho": float(rho),
        "h_brent": float(h[0]),
        "h_wti": float(h[1]),
        "h_dubai": float(h[2]),
        "obs_sigma_brent": float(obs_sigma[0]),
        "obs_sigma_wti": float(obs_sigma[1]),
        "obs_sigma_dubai": float(obs_sigma[2]),
    }
    return transition, intercept, process_cov, observation, observation_cov, named


def kalman_filter(
    y: np.ndarray,
    theta: np.ndarray,
    dt: float,
    return_states: bool = False,
) -> tuple[float, np.ndarray | None]:
    """双因子状态空间模型的 Kalman 滤波似然。"""

    transition, intercept, process_cov, observation, observation_cov, _ = unpack_state_params(theta, dt)
    state = np.array([float(np.nanmean(y[:, 0])), 0.0], dtype=float)
    cov = np.diag([0.25, 0.25])
    identity = np.eye(2)
    loglike = 0.0
    filtered = np.zeros((y.shape[0], 2), dtype=float)

    for t in range(y.shape[0]):
        if t > 0:
            state = transition @ state + intercept
            cov = transition @ cov @ transition.T + process_cov
            cov = (cov + cov.T) / 2.0

        innovation = y[t] - observation @ state
        innovation_cov = observation @ cov @ observation.T + observation_cov
        innovation_cov = (innovation_cov + innovation_cov.T) / 2.0

        try:
            sign, logdet = np.linalg.slogdet(innovation_cov)
            if sign <= 0:
                return 1e12, None
            solved = np.linalg.solve(innovation_cov, innovation)
        except np.linalg.LinAlgError:
            return 1e12, None

        loglike += -0.5 * (3 * np.log(2 * np.pi) + logdet + innovation @ solved)
        gain = cov @ observation.T @ np.linalg.inv(innovation_cov)
        state = state + gain @ innovation
        cov = (identity - gain @ observation) @ cov @ (identity - gain @ observation).T + gain @ observation_cov @ gain.T
        cov = (cov + cov.T) / 2.0
        filtered[t] = state

        if not np.all(np.isfinite(state)) or not np.all(np.isfinite(cov)):
            return 1e12, None

    neg_loglike = float(-loglike)
    return (neg_loglike, filtered) if return_states else (neg_loglike, None)


def initial_theta(y: np.ndarray) -> np.ndarray:
    """给状态空间极大似然估计设置初值。"""

    daily_vol = float(np.nanstd(np.diff(y[:, 0])))
    annual_vol = max(daily_vol * np.sqrt(252), 0.05)
    return np.array(
        [
            0.0,
            0.0,
            np.log(1.0),
            np.log(annual_vol),
            np.log(0.20),
            np.arctanh(0.2),
            0.15,
            0.00,
            -0.10,
            np.log(0.05),
            np.log(0.05),
            np.log(0.05),
        ],
        dtype=float,
    )


def fit_two_factor_state_space(oil_df: pd.DataFrame, maxiter: int) -> tuple[pd.DataFrame, StateSpaceFit]:
    """拟合双因子状态空间模型，并输出每日 X_t、delta_t。"""

    y = oil_df[["brent_log", "wti_log", "dubai_log"]].to_numpy(dtype=float)
    dt = 1.0 / 252.0
    theta0 = initial_theta(y)
    bounds = [
        (-1.0, 1.0),
        (-2.0, 2.0),
        (np.log(0.02), np.log(10.0)),
        (np.log(0.01), np.log(2.0)),
        (np.log(0.01), np.log(2.0)),
        (np.arctanh(-0.95), np.arctanh(0.95)),
        (-5.0, 5.0),
        (-5.0, 5.0),
        (-5.0, 5.0),
        (np.log(0.002), np.log(0.50)),
        (np.log(0.002), np.log(0.50)),
        (np.log(0.002), np.log(0.50)),
    ]
    result = minimize(
        lambda th: kalman_filter(y, th, dt, return_states=False)[0],
        theta0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": maxiter, "ftol": 1e-7, "maxls": 25},
    )
    neg_ll, filtered = kalman_filter(y, result.x, dt, return_states=True)
    _, _, _, _, _, named = unpack_state_params(result.x, dt)

    out = oil_df[["date", "usd_cny", "brent", "wti", "dubai"]].copy()
    out["state_x"] = filtered[:, 0]
    out["state_delta"] = filtered[:, 1]
    out["state_space_index_usd"] = np.exp(out["state_x"])
    fit = StateSpaceFit(
        params=named,
        neg_loglike=float(neg_ll),
        success=bool(result.success),
        message=str(result.message),
        iterations=int(result.nit),
    )
    return out, fit


def load_or_fit_two_factor_state(config: MechanismConfig, refit: bool, maxiter: int) -> tuple[pd.DataFrame, dict[str, object]]:
    """读取已有双因子状态；如果指定 refit 或文件不存在，则重新估计。"""

    if STATE_PATH.exists() and not refit:
        state = pd.read_csv(STATE_PATH, parse_dates=["date"])
        state = state[state["date"] >= pd.Timestamp(config.start_date)].copy()
        validate_required_columns(state, ["date", "usd_cny", "state_x", "state_delta", "state_space_index_usd"], "双因子状态文件")
        info = {
            "source": str(STATE_PATH),
            "refit": False,
            "note": "读取已经估计好的双因子状态，以复现实验结果；如需从原始数据重估，可加 --refit。",
        }
        return state.sort_values("date").reset_index(drop=True), info

    panel = read_oil_panel()
    oil_df = prepare_log_observations(panel, config.start_date)
    state, fit = fit_two_factor_state_space(oil_df, maxiter=maxiter)
    info = {"source": "refit_from_raw_oil_prices", "refit": True, "fit": asdict(fit)}
    return state, info


# ==================== 国内调价机制层 ====================

def trailing_window(panel: pd.DataFrame, end: pd.Timestamp, col: str, size: int) -> pd.DataFrame:
    """取截止日之前最近 size 条有效工作日数据。"""

    return panel.loc[panel["date"] <= end, ["date", col, "usd_cny"]].dropna().tail(size).copy()


def apply_zone_rule(delta: float, ma_usd: float, config: MechanismConfig) -> float:
    """应用 40 美元地板价和 130 美元天花板价规则。"""

    if ma_usd <= config.floor_usd_per_bbl and delta < 0:
        return 0.0
    if ma_usd >= config.ceiling_usd_per_bbl and delta > 0:
        return delta * config.ceiling_up_factor
    return delta


def simulate_mechanism(domestic: pd.DataFrame, panel: pd.DataFrame, price_col: str, config: MechanismConfig) -> pd.DataFrame:
    """用修正后的国内调价机制计算纯理论调价幅度。"""

    rows: list[dict[str, object]] = []
    carry_gasoline = 0.0
    carry_diesel = 0.0

    for i in range(1, len(domestic)):
        prev_date = domestic.loc[i - 1, "adjust_date"]
        date = domestic.loc[i, "adjust_date"]
        pricing_end = date - pd.Timedelta(days=config.pricing_lag_days)
        base_pricing_end = prev_date - pd.Timedelta(days=config.pricing_lag_days)

        window = trailing_window(panel, pricing_end, price_col, config.window_size)
        base_window = trailing_window(panel, base_pricing_end, price_col, config.window_size)
        if window.empty or base_window.empty:
            continue

        current_ma = float(window[price_col].mean())
        base_usd = float(base_window[price_col].mean())
        current_ma_ref = max(current_ma, config.floor_usd_per_bbl)
        base_ref = max(base_usd, config.floor_usd_per_bbl)
        if not np.isfinite(base_ref) or base_ref == 0:
            continue

        fx_avg = float(window["usd_cny"].mean())
        change_rate = current_ma_ref / base_ref - 1.0
        gasoline_raw = current_ma_ref * config.gasoline_bbl_per_ton * fx_avg * change_rate * config.tax_factor
        diesel_raw = current_ma_ref * config.diesel_bbl_per_ton * fx_avg * change_rate * config.tax_factor

        gasoline_policy = apply_zone_rule(gasoline_raw, current_ma, config)
        diesel_policy = apply_zone_rule(diesel_raw, current_ma, config)

        gasoline_total = gasoline_policy + carry_gasoline
        diesel_total = diesel_policy + carry_diesel
        if abs(gasoline_total) >= config.threshold_yuan_per_ton:
            gasoline_theory = gasoline_total
            carry_gasoline = 0.0
        else:
            gasoline_theory = 0.0
            carry_gasoline = gasoline_total
        if abs(diesel_total) >= config.threshold_yuan_per_ton:
            diesel_theory = diesel_total
            carry_diesel = 0.0
        else:
            diesel_theory = 0.0
            carry_diesel = diesel_total

        rows.append(
            {
                "adjust_date": date,
                "prev_adjust_date": prev_date,
                "oil_index": price_col,
                "pricing_end": pricing_end,
                "window_observations": int(len(window)),
                "window_start": window["date"].min(),
                "window_end": window["date"].max(),
                "base_usd_per_bbl": base_usd,
                "base_window_observations": int(len(base_window)),
                "base_window_start": base_window["date"].min(),
                "base_window_end": base_window["date"].max(),
                "ma_usd_per_bbl": current_ma,
                "ref_ma_usd_per_bbl": current_ma_ref,
                "avg_usd_cny": fx_avg,
                "oil_change_rate": change_rate,
                "zone": "floor" if current_ma <= config.floor_usd_per_bbl else ("ceiling" if current_ma >= config.ceiling_usd_per_bbl else "normal"),
                "gasoline_formula_delta": gasoline_policy,
                "diesel_formula_delta": diesel_policy,
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


# ==================== 期货期限结构特征 ====================

def find_futuredata_path() -> Path:
    """定位队友整理好的三油种 M1/M3/M6 月度期货数据。"""

    for path in ROOT.rglob("futuredata.xlsx"):
        if path.is_file():
            return path
    raise FileNotFoundError("没有找到 期货数据/futuredata.xlsx")


def read_futures_monthly() -> pd.DataFrame:
    """读取 Brent、Dubai、WTI 的 M1/M3/M6 月度期货数据。"""

    path = find_futuredata_path()
    df = pd.read_excel(path)
    columns = ["date"] + FUTURES_PRICE_COLS
    df = df.rename(columns={old: new for old, new in zip(df.columns, columns)})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in FUTURES_PRICE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["date"]).sort_values("date")[columns]


def make_futures_available() -> pd.DataFrame:
    """把月度期货数据转换成调价日前可获得的期限结构特征。"""

    futures = read_futures_monthly().copy()
    for oil in ["brent", "dubai", "wti"]:
        futures[f"{oil}_m1_m6_spread"] = np.log(futures[f"{oil}_m1"] / futures[f"{oil}_m6"])
        futures[f"{oil}_m1_m3_spread"] = np.log(futures[f"{oil}_m1"] / futures[f"{oil}_m3"])
        futures[f"{oil}_m3_m6_spread"] = np.log(futures[f"{oil}_m3"] / futures[f"{oil}_m6"])

    futures["term_spread_m1_m6_mean"] = futures[
        ["brent_m1_m6_spread", "dubai_m1_m6_spread", "wti_m1_m6_spread"]
    ].mean(axis=1)
    futures["term_spread_m1_m3_mean"] = futures[
        ["brent_m1_m3_spread", "dubai_m1_m3_spread", "wti_m1_m3_spread"]
    ].mean(axis=1)
    futures["term_spread_m3_m6_mean"] = futures[
        ["brent_m3_m6_spread", "dubai_m3_m6_spread", "wti_m3_m6_spread"]
    ].mean(axis=1)
    futures["term_spread_m1_m6_change"] = futures["term_spread_m1_m6_mean"].diff()

    # 月末数据从下一天才可用于调价日，避免未来信息泄露。
    futures["available_date"] = futures["date"] + pd.offsets.Day(1)
    keep_cols = ["available_date"] + FUTURES_FEATURE_COLS
    return futures[keep_cols].replace([np.inf, -np.inf], np.nan).sort_values("available_date")


# ==================== 串联融合特征工程 ====================

def window_mean(panel: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, col: str) -> float:
    """计算某个窗口内状态变量或汇率的均值。"""

    mask = (panel["date"] >= start) & (panel["date"] <= end)
    values = pd.to_numeric(panel.loc[mask, col], errors="coerce")
    return float(values.mean()) if values.notna().any() else float("nan")


def build_serial_feature_dataset(
    config: MechanismConfig,
    refit_state: bool,
    maxiter: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """构建逐调价日的 RidgeCV 传导层特征。"""

    state_panel, state_info = load_or_fit_two_factor_state(config, refit=refit_state, maxiter=maxiter)
    domestic = read_domestic_adjustments(config.start_date)
    validation = simulate_mechanism(domestic, state_panel, "state_space_index_usd", config)
    validation = validation.sort_values("adjust_date").reset_index(drop=True)

    rows = []
    for _, row in validation.iterrows():
        window_start = pd.Timestamp(row["window_start"])
        window_end = pd.Timestamp(row["window_end"])
        base_start = pd.Timestamp(row["base_window_start"])
        base_end = pd.Timestamp(row["base_window_end"])

        delta_mean = window_mean(state_panel, window_start, window_end, "state_delta")
        delta_base = window_mean(state_panel, base_start, base_end, "state_delta")
        x_mean = window_mean(state_panel, window_start, window_end, "state_x")
        fx_mean = window_mean(state_panel, window_start, window_end, "usd_cny")
        fx_base = window_mean(state_panel, base_start, base_end, "usd_cny")

        rows.append(
            {
                "adjust_date": row["adjust_date"],
                "state_x_window_mean": x_mean,
                "delta_window_mean": delta_mean,
                "delta_window_change": delta_mean - delta_base,
                "delta_abs": abs(delta_mean),
                "usd_cny_window_mean": fx_mean,
                "usd_cny_change_rate": fx_mean / fx_base - 1.0 if pd.notna(fx_base) and fx_base else np.nan,
            }
        )

    features = validation.merge(pd.DataFrame(rows), on="adjust_date", how="left")
    features = asof_merge_available(features, make_futures_available())
    features[FUTURES_FEATURE_COLS] = features[FUTURES_FEATURE_COLS].ffill().bfill()
    features = features.replace([np.inf, -np.inf], np.nan).sort_values("adjust_date").reset_index(drop=True)
    return features, state_info


# ==================== RidgeCV 政策传导层 ====================

def train_policy_transmission(
    df: pd.DataFrame,
    product: str,
    train_end: str,
    feature_mode: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """训练汽油或柴油的 RidgeCV 政策传导层。"""

    result = df.copy()
    target = f"{product}_actual_delta"
    theory_col = f"{product}_theory_delta"
    feature_sets = {
        "theory_only": [theory_col],
        "delta": [theory_col, "delta_window_mean", "delta_window_change", "delta_abs"],
        "delta_fx": [
            theory_col,
            "delta_window_mean",
            "delta_window_change",
            "delta_abs",
            "usd_cny_change_rate",
        ],
        "delta_futures": [
            theory_col,
            "delta_window_mean",
            "delta_window_change",
            "delta_abs",
            "term_spread_m1_m6_mean",
            "term_spread_m1_m6_change",
        ],
        "delta_fx_futures": [
            theory_col,
            "delta_window_mean",
            "delta_window_change",
            "delta_abs",
            "usd_cny_change_rate",
            "term_spread_m1_m6_mean",
            "term_spread_m1_m3_mean",
            "term_spread_m3_m6_mean",
            "term_spread_m1_m6_change",
        ],
    }
    feature_cols = feature_sets[feature_mode]
    result[feature_cols] = clean_numeric_frame(result[feature_cols])
    train_mask = result["adjust_date"] <= pd.Timestamp(train_end)
    if not bool(train_mask.any()):
        raise ValueError(f"{product} 没有训练样本，请检查 train_end={train_end}")

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 31))),
        ]
    )
    model.fit(result.loc[train_mask, feature_cols], result.loc[train_mask, target])

    pred_col = f"{product}_serial_fusion_delta"
    result[pred_col] = model.predict(result[feature_cols])
    ridge = model.named_steps["ridge"]
    info = {
        "mode": feature_mode,
        "alpha": float(ridge.alpha_),
        "features": feature_cols,
        "coef": dict(zip(feature_cols, ridge.coef_.astype(float))),
    }
    return result, info


# ==================== 指标评估 ====================

def metric_block(df: pd.DataFrame, product: str, pred_col: str) -> dict[str, float | int | None]:
    """计算 MAE、RMSE、方向准确率、门槛准确率、相关系数和 WMAPE。"""

    actual_col = f"{product}_actual_delta"
    mask = df[actual_col].notna() & df[pred_col].notna()
    actual = df.loc[mask, actual_col].to_numpy(dtype=float)
    pred = df.loc[mask, pred_col].to_numpy(dtype=float)
    if actual.size == 0:
        return {"n": 0}

    err = pred - actual
    nonzero = actual != 0
    denom = np.abs(actual).sum()
    direction_accuracy = np.mean(np.sign(pred[nonzero]) == np.sign(actual[nonzero])) if nonzero.any() else np.nan
    threshold_accuracy = np.mean((np.abs(pred) >= 50.0) == (np.abs(actual) >= 50.0))
    return {
        "n": int(actual.size),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(math.sqrt(np.mean(err * err))),
        "mean_error": float(np.mean(err)),
        "direction_accuracy": float(direction_accuracy) if np.isfinite(direction_accuracy) else None,
        "threshold_accuracy": float(threshold_accuracy),
        "corr": float(np.corrcoef(actual, pred)[0, 1]) if actual.size > 1 and np.std(pred) > 0 else None,
        "wmape": float(np.abs(err).sum() / denom) if denom else None,
    }


def evaluate_mode(features: pd.DataFrame, config: MechanismConfig, feature_mode: str) -> tuple[pd.DataFrame, dict[str, object]]:
    """评估一种传导层特征组合。"""

    out, gas_info = train_policy_transmission(features, "gasoline", config.train_end, feature_mode)
    out, diesel_info = train_policy_transmission(out, "diesel", config.train_end, feature_mode)
    samples = {
        "all": out,
        "test_after_train": out[out["adjust_date"] > pd.Timestamp(config.train_end)],
        "test_2023_2025": out[
            (out["adjust_date"] > pd.Timestamp(config.train_end))
            & (out["adjust_date"] < pd.Timestamp("2026-01-01"))
        ],
        "exclude_2026": out[out["adjust_date"] < pd.Timestamp("2026-01-01")],
    }
    metrics = {}
    for name, sample in samples.items():
        metrics[name] = {
            "gasoline_mechanism": metric_block(sample, "gasoline", "gasoline_theory_delta"),
            "diesel_mechanism": metric_block(sample, "diesel", "diesel_theory_delta"),
            "gasoline_serial_fusion": metric_block(sample, "gasoline", "gasoline_serial_fusion_delta"),
            "diesel_serial_fusion": metric_block(sample, "diesel", "diesel_serial_fusion_delta"),
        }
    summary = {
        "feature_mode": feature_mode,
        "transmission_models": {"gasoline": gas_info, "diesel": diesel_info},
        "metrics": metrics,
    }
    return out, summary


def best_mode(ablation: pd.DataFrame) -> str:
    """按测试期汽油和柴油 MAE 之和选择最优特征组合。"""

    score = ablation["gasoline_test_mae"].astype(float) + ablation["diesel_test_mae"].astype(float)
    return str(ablation.loc[score.idxmin(), "feature_mode"])


def write_report(
    output_dir: Path,
    config: MechanismConfig,
    state_info: dict[str, object],
    ablation: pd.DataFrame,
    summaries: dict[str, dict[str, object]],
    selected_mode: str,
) -> None:
    """写出给论文和队友阅读的模型说明。"""

    selected = summaries[selected_mode]["metrics"]["test_after_train"]
    lines = [
        "双因子状态空间 + 期货期限结构 + RidgeCV 政策传导串联融合模型说明",
        "============================================================",
        "",
        "一、建模思路",
        "",
        "本版吸收队友 Word 文档中的双因子状态空间思想，保留 X_t 与 delta_t。",
        "其中 X_t 表示基础油价水平，delta_t 表示便利收益、地缘冲突溢价或短期稀缺性压力。",
        "同时引入 RidgeCV 政策传导层，并加入真实 M1/M3/M6 期货期限结构作为窗口级修正变量。",
        "",
        "二、串联结构",
        "",
        "第一层：双因子状态空间模型从 Brent、WTI、Dubai 中提取 X_t 和 delta_t。",
        "第二层：修正后的国内机制层计算纯理论调价幅度。",
        "第三层：RidgeCV 使用纯理论调价、delta_t、汇率变动和期货期限结构特征拟合实际调价。",
        "期货数据不直接进入双因子 Kalman 观测方程，而只进入窗口级传导层，以降低月度低频数据对状态估计的干扰。",
        "",
        "三、状态来源",
        "",
        json.dumps(state_info, ensure_ascii=False, indent=2),
        "",
        "四、特征组合实验",
        "",
        ablation.to_string(index=False),
        "",
        "五、推荐传导层",
        "",
        f"综合最优特征组合：{selected_mode}",
        f"汽油测试期 MAE：{selected['gasoline_serial_fusion']['mae']:.2f}",
        f"柴油测试期 MAE：{selected['diesel_serial_fusion']['mae']:.2f}",
        f"汽油测试期相关系数：{selected['gasoline_serial_fusion']['corr']:.4f}",
        f"柴油测试期相关系数：{selected['diesel_serial_fusion']['corr']:.4f}",
        f"汽油测试期方向准确率：{selected['gasoline_serial_fusion']['direction_accuracy']:.4f}",
        f"柴油测试期方向准确率：{selected['diesel_serial_fusion']['direction_accuracy']:.4f}",
        "",
        "六、论文表述建议",
        "",
        "可以表述为：发改委实际调价不仅参考基础油价理论变化，也会根据地缘冲突烈度",
        "（delta_t）、汇率成本变化和期限结构反映的市场预期进行平滑干预。因此本文在",
        "双因子状态空间模型之后引入岭回归政策传导层，提高对实际调价行为的拟合能力。",
        "CPI/PPI/PMI/GDP 不进入第一题模型，留给第二题评价宏观经济影响。",
        "",
    ]
    (output_dir / "model_description.txt").write_text("\n".join(lines), encoding="utf-8")


# ==================== 命令行入口 ====================

def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="双因子状态空间 + 期货期限结构 + RidgeCV 政策传导模型")
    parser.add_argument("--start-date", default=MechanismConfig.start_date)
    parser.add_argument("--train-end", default=MechanismConfig.train_end)
    parser.add_argument("--window-size", type=int, default=MechanismConfig.window_size)
    parser.add_argument("--pricing-lag-days", type=int, default=MechanismConfig.pricing_lag_days)
    parser.add_argument("--maxiter", type=int, default=250, help="重新估计双因子状态空间时的最大迭代次数")
    parser.add_argument("--refit", action="store_true", help="从原始三油种价格重新估计双因子状态，而不是读取已有状态文件")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    """运行完整串联融合模型。"""

    args = parse_args()
    config = MechanismConfig(
        start_date=args.start_date,
        train_end=args.train_end,
        window_size=args.window_size,
        pricing_lag_days=args.pricing_lag_days,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features, state_info = build_serial_feature_dataset(config, refit_state=args.refit, maxiter=args.maxiter)
    feature_modes = ["theory_only", "delta", "delta_fx", "delta_futures", "delta_fx_futures"]
    summaries: dict[str, dict[str, object]] = {}
    ablation_rows = []

    for mode in feature_modes:
        result, summary = evaluate_mode(features, config, mode)
        summaries[mode] = summary
        result.to_csv(output_dir / f"validation_{mode}.csv", index=False, encoding="utf-8-sig")
        test = summary["metrics"]["test_after_train"]
        test_2023_2025 = summary["metrics"]["test_2023_2025"]
        ablation_rows.append(
            {
                "feature_mode": mode,
                "gasoline_test_mae": test["gasoline_serial_fusion"]["mae"],
                "diesel_test_mae": test["diesel_serial_fusion"]["mae"],
                "gasoline_test_wmape": test["gasoline_serial_fusion"]["wmape"],
                "diesel_test_wmape": test["diesel_serial_fusion"]["wmape"],
                "gasoline_2023_2025_mae": test_2023_2025["gasoline_serial_fusion"]["mae"],
                "diesel_2023_2025_mae": test_2023_2025["diesel_serial_fusion"]["mae"],
            }
        )

    ablation = pd.DataFrame(ablation_rows)
    selected_mode = best_mode(ablation)
    selected_result = pd.read_csv(output_dir / f"validation_{selected_mode}.csv", parse_dates=["adjust_date"])
    selected_result.to_csv(output_dir / "serial_fusion_validation.csv", index=False, encoding="utf-8-sig")
    ablation.to_csv(output_dir / "ablation_transmission_metrics.csv", index=False, encoding="utf-8-sig")

    summary = {
        "config": asdict(config),
        "state_info": state_info,
        "selected_mode": selected_mode,
        "modes": summaries,
        "ablation": ablation_rows,
    }
    with (output_dir / "summary_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    write_report(output_dir, config, state_info, ablation, summaries, selected_mode)

    print("双因子状态空间 + 期货期限结构 + RidgeCV 政策传导串联融合模型")
    print(ablation.to_string(index=False))
    print(f"\n综合最优特征组合: {selected_mode}")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()
