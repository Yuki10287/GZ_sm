from __future__ import annotations

"""真实 Brent/Dubai/WTI M1/M3/M6 期限结构增强版双因子状态空间模型。

这版是在 Gemini 建议方向上的进一步实验：
1. 保留队友双因子状态变量 X_t 和 delta_t；
2. 使用真实 Brent、Dubai、WTI 的 M1/M3/M6 期货数据；
3. 不把月度期货数据线性插值到未来，而是按“已公布的最近一期”向后填充；
4. 观测方程区分价格水平和期限价差：
   - 现货价格水平主要锚定基础油价状态 X_t；
   - log(M1/M3)、log(M1/M6)、log(M3/M6) 主要帮助识别 delta_t；
5. 调价机制层沿用修正版：上一周期均价基准、1 天计价滞后、50 元硬门槛、40/130 美元硬边界。

注意：这不是严格复刻 Pizzinga 文献，而是面向本题数据条件的
“真实期限结构增强双因子状态空间模型”。
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import task1_price_mechanism as base


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs" / "task1_real_m136_gemini"


@dataclass(frozen=True)
class ObservationSpec:
    """单个观测变量在观测矩阵中的类型。"""

    name: str
    kind: str  # "level" 表示 H=[1,h]；"spread" 表示 H=[0,h]


@dataclass(frozen=True)
class RealM136Fit:
    """状态空间极大似然估计结果。"""

    params: dict[str, float]
    neg_loglike: float
    success: bool
    message: str
    iterations: int
    obs_specs: list[dict[str, str]]


def find_futuredata_path() -> Path:
    """定位队友提供的真实三油种 M1/M3/M6 数据文件。"""

    for path in ROOT.rglob("futuredata.xlsx"):
        if path.is_file():
            return path
    raise FileNotFoundError("没有找到 futuredata.xlsx，请确认期货数据目录存在。")


def read_real_m136_futures() -> pd.DataFrame:
    """读取真实 M1/M3/M6 月度数据，并保留原始月末日期。"""

    path = find_futuredata_path()
    df = pd.read_excel(path)
    expected = [
        "date",
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
    df = df.rename(columns={old: new for old, new in zip(df.columns, expected)})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in expected[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    return df[expected]


def expand_monthly_futures_without_leakage(futures: pd.DataFrame) -> pd.DataFrame:
    """把月度期货数据扩展到日度。

    这里不用线性插值，因为线性插值会提前用到下一月末数据，存在未来信息泄露。
    保守处理为：某个月末数据从该日期起向后可用，直到下一期月末数据出现。
    """

    futures = futures.set_index("date").sort_index()
    daily_index = pd.date_range(futures.index.min(), futures.index.max(), freq="D")
    daily = futures.reindex(daily_index).ffill()
    daily.index.name = "date"
    return daily.reset_index()


def build_observation_panel(start_date: str) -> tuple[pd.DataFrame, list[ObservationSpec]]:
    """构建现货价格水平 + 真实期限价差观测面板。"""

    spot = base.read_oil_panel()
    futures = expand_monthly_futures_without_leakage(read_real_m136_futures())
    df = spot.merge(futures, on="date", how="inner")
    df = df[df["date"] >= pd.Timestamp(start_date)].copy()

    price_cols = [
        "brent",
        "wti",
        "dubai",
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
    for col in price_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").mask(lambda s: s <= 0)

    level_cols = ["brent", "wti", "dubai"]
    for col in level_cols:
        df[f"log_{col}"] = np.log(df[col])

    spread_specs: list[tuple[str, str, str]] = []
    for oil in ["brent", "dubai", "wti"]:
        spread_specs.extend(
            [
                (f"{oil}_m1_m3_spread", f"{oil}_m1", f"{oil}_m3"),
                (f"{oil}_m1_m6_spread", f"{oil}_m1", f"{oil}_m6"),
                (f"{oil}_m3_m6_spread", f"{oil}_m3", f"{oil}_m6"),
            ]
        )
    for out_col, near_col, far_col in spread_specs:
        df[out_col] = np.log(df[near_col] / df[far_col])

    obs_specs = [
        ObservationSpec("log_brent", "level"),
        ObservationSpec("log_wti", "level"),
        ObservationSpec("log_dubai", "level"),
    ]
    obs_specs.extend(ObservationSpec(name, "spread") for name, _, _ in spread_specs)

    obs_cols = [spec.name for spec in obs_specs]
    df = df.dropna(subset=obs_cols).reset_index(drop=True)
    return df, obs_specs


def unpack(
    theta: np.ndarray,
    obs_specs: list[ObservationSpec],
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """把优化器参数映射为状态转移矩阵和观测矩阵。"""

    n_obs = len(obs_specs)
    mu = theta[0]
    alpha = theta[1]
    kappa = np.exp(theta[2])
    sigma1 = np.exp(theta[3])
    sigma2 = np.exp(theta[4])
    h = theta[5 : 5 + n_obs]
    obs_sigma = np.exp(theta[5 + n_obs : 5 + 2 * n_obs])

    a = np.array([[1.0, -dt], [0.0, 1.0 - kappa * dt]], dtype=float)
    c = np.array([(mu - 0.5 * sigma1 * sigma1) * dt, kappa * alpha * dt], dtype=float)
    q = np.array([[sigma1 * sigma1 * dt, 0.0], [0.0, sigma2 * sigma2 * dt]], dtype=float)

    hmat = np.zeros((n_obs, 2), dtype=float)
    for i, spec in enumerate(obs_specs):
        hmat[i, 0] = 1.0 if spec.kind == "level" else 0.0
        hmat[i, 1] = h[i]

    r = np.diag(obs_sigma * obs_sigma)
    named = {
        "mu": float(mu),
        "alpha": float(alpha),
        "kappa": float(kappa),
        "sigma1": float(sigma1),
        "sigma2": float(sigma2),
        "rho_fixed": 0.0,
    }
    return a, c, q, hmat, r, named


def kalman_filter(
    y: np.ndarray,
    theta: np.ndarray,
    obs_specs: list[ObservationSpec],
    dt: float,
    return_states: bool = False,
    penalty_weight: float = 0.0,
) -> tuple[float, np.ndarray | None]:
    """带 delta_t L2 惩罚的双因子 Kalman 滤波似然。"""

    n_obs = y.shape[1]
    a, c, q, hmat, r, _ = unpack(theta, obs_specs, dt)
    state = np.array([float(np.nanmean(y[:, 0])), 0.0], dtype=float)
    cov = np.diag([0.25, 0.10])
    identity = np.eye(2)
    loglike = 0.0
    penalty = 0.0
    filtered = np.zeros((y.shape[0], 2), dtype=float)

    for t in range(y.shape[0]):
        if t > 0:
            state = a @ state + c
            cov = a @ cov @ a.T + q
            cov = (cov + cov.T) / 2.0

        innovation = y[t] - hmat @ state
        innovation_cov = hmat @ cov @ hmat.T + r
        innovation_cov = (innovation_cov + innovation_cov.T) / 2.0

        try:
            sign, logdet = np.linalg.slogdet(innovation_cov)
            if sign <= 0:
                return 1e12, None
            solved = np.linalg.solve(innovation_cov, innovation)
        except np.linalg.LinAlgError:
            return 1e12, None

        loglike += -0.5 * (n_obs * np.log(2 * np.pi) + logdet + innovation @ solved)
        gain = cov @ hmat.T @ np.linalg.inv(innovation_cov)
        state = state + gain @ innovation
        cov = (identity - gain @ hmat) @ cov @ (identity - gain @ hmat).T + gain @ r @ gain.T
        cov = (cov + cov.T) / 2.0
        filtered[t] = state
        penalty += state[1] * state[1]

        if not np.all(np.isfinite(state)) or not np.all(np.isfinite(cov)):
            return 1e12, None

    neg_loglike = float(-loglike + penalty_weight * penalty / y.shape[0])
    return (neg_loglike, filtered) if return_states else (neg_loglike, None)


def initial_theta(y: np.ndarray, obs_specs: list[ObservationSpec]) -> np.ndarray:
    """给极大似然优化器设置较稳健的初始值。"""

    n_obs = y.shape[1]
    daily_vol = float(np.nanstd(np.diff(y[:, 0])))
    annual_vol = max(daily_vol * np.sqrt(252), 0.05)
    h0 = np.array([0.15 if spec.kind == "level" else 1.0 for spec in obs_specs], dtype=float)
    obs_sigma0 = np.array([0.04 if spec.kind == "level" else 0.02 for spec in obs_specs], dtype=float)
    return np.r_[0.0, 0.0, np.log(1.0), np.log(annual_vol), np.log(0.15), h0, np.log(obs_sigma0)]


def fit_state_space(
    obs_panel: pd.DataFrame,
    obs_specs: list[ObservationSpec],
    maxiter: int,
    penalty_weight: float,
) -> tuple[pd.DataFrame, RealM136Fit]:
    """拟合真实 M1/M3/M6 期限结构增强双因子状态空间模型。"""

    obs_cols = [spec.name for spec in obs_specs]
    y = obs_panel[obs_cols].to_numpy(dtype=float)
    dt = 1.0 / 252.0
    theta0 = initial_theta(y, obs_specs)
    n_obs = y.shape[1]
    bounds = [
        (-1.0, 1.0),
        (-2.0, 2.0),
        (np.log(0.05), np.log(20.0)),
        (np.log(0.01), np.log(2.0)),
        (np.log(0.01), np.log(2.0)),
    ]
    bounds += [(-5.0, 5.0)] * n_obs
    bounds += [(np.log(0.002), np.log(0.80))] * n_obs

    result = minimize(
        lambda th: kalman_filter(y, th, obs_specs, dt, return_states=False, penalty_weight=penalty_weight)[0],
        theta0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": maxiter, "ftol": 1e-7, "maxls": 25},
    )

    neg_ll, filtered = kalman_filter(y, result.x, obs_specs, dt, return_states=True, penalty_weight=penalty_weight)
    _, _, _, _, _, named = unpack(result.x, obs_specs, dt)
    h = result.x[5 : 5 + n_obs]
    obs_sigma = np.exp(result.x[5 + n_obs : 5 + 2 * n_obs])
    for spec, value in zip(obs_specs, h):
        named[f"h_{spec.name}"] = float(value)
    for spec, value in zip(obs_specs, obs_sigma):
        named[f"obs_sigma_{spec.name}"] = float(value)

    keep_cols = [
        "date",
        "usd_cny",
        "brent",
        "wti",
        "dubai",
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
    out = obs_panel[keep_cols].copy()
    out["state_x"] = filtered[:, 0]
    out["state_delta"] = filtered[:, 1]
    out["real_m136_index_usd"] = np.exp(out["state_x"])

    fit = RealM136Fit(
        params=named,
        neg_loglike=float(neg_ll),
        success=bool(result.success),
        message=str(result.message),
        iterations=int(result.nit),
        obs_specs=[asdict(spec) for spec in obs_specs],
    )
    return out, fit


def run_validation(config: base.MechanismConfig, maxiter: int, penalty_weight: float) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """拟合状态空间模型，并用修正机制层回测国内调价。"""

    obs_panel, obs_specs = build_observation_panel(config.start_date)
    state_panel, fit = fit_state_space(obs_panel, obs_specs, maxiter=maxiter, penalty_weight=penalty_weight)

    domestic = base.read_domestic_adjustments(config.start_date)
    validation = base.simulate_mechanism(domestic, state_panel, "real_m136_index_usd", config)
    max_date = state_panel["date"].max()
    validation = validation[pd.to_datetime(validation["pricing_end"]) <= max_date].reset_index(drop=True)

    summary = {
        "config": asdict(config),
        "data_note": {
            "futuredata_path": str(find_futuredata_path()),
            "futures_data_start": str(read_real_m136_futures()["date"].min().date()),
            "futures_data_end": str(read_real_m136_futures()["date"].max().date()),
            "leakage_control": "Monthly futures are forward-filled only after each month-end observation; no linear interpolation is used.",
            "observation_design": "Spot log levels identify X_t; log term spreads identify delta_t.",
        },
        "state_space_fit": asdict(fit),
        "metrics": {
            "gasoline": base.metric_block(validation, "gasoline", "gasoline_theory_delta"),
            "diesel": base.metric_block(validation, "diesel", "diesel_theory_delta"),
            "test_after_train": {
                "gasoline": base.metric_block(
                    validation[validation["adjust_date"] > pd.Timestamp(config.train_end)],
                    "gasoline",
                    "gasoline_theory_delta",
                ),
                "diesel": base.metric_block(
                    validation[validation["adjust_date"] > pd.Timestamp(config.train_end)],
                    "diesel",
                    "diesel_theory_delta",
                ),
            },
        },
    }
    return state_panel, validation, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real Brent/Dubai/WTI M1/M3/M6 Gemini-style two-factor state-space model.")
    parser.add_argument("--start-date", default=base.MechanismConfig.start_date)
    parser.add_argument("--train-end", default=base.MechanismConfig.train_end)
    parser.add_argument("--window-size", type=int, default=base.MechanismConfig.window_size)
    parser.add_argument("--pricing-lag-days", type=int, default=1)
    parser.add_argument("--maxiter", type=int, default=250)
    parser.add_argument("--penalty-weight", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = base.MechanismConfig(
        start_date=args.start_date,
        train_end=args.train_end,
        window_size=args.window_size,
        pricing_lag_days=args.pricing_lag_days,
    )
    state_panel, validation, summary = run_validation(config, maxiter=args.maxiter, penalty_weight=args.penalty_weight)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    state_panel.to_csv(OUTPUT_DIR / "real_m136_state_index.csv", index=False, encoding="utf-8-sig")
    validation.to_csv(OUTPUT_DIR / "real_m136_validation.csv", index=False, encoding="utf-8-sig")
    with (OUTPUT_DIR / "summary_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("真实 M1/M3/M6 期限结构增强双因子状态空间模型")
    print("Fit:", summary["state_space_fit"])
    print("Gasoline:", summary["metrics"]["gasoline"])
    print("Diesel:", summary["metrics"]["diesel"])
    print("Test after train gasoline:", summary["metrics"]["test_after_train"]["gasoline"])
    print("Test after train diesel:", summary["metrics"]["test_after_train"]["diesel"])
    print(f"\nOutputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
