from __future__ import annotations

"""严格版队友双因子状态空间模型。

本脚本实现队友文档中最理论化的一版：潜在状态
z_t = [X_t, delta_t]^T，由 Brent/WTI/Dubai 的对数价格观测。
模型使用 Kalman 似然和极大似然估计参数，再通过
``task1_teammate_model`` 接入队友原始调价控制公式。

该脚本主要作为诊断基准：用于观察仅凭三条近似现货油价序列强行识别
类似便利收益的 delta_t 因子会出现什么效果。
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import task1_price_mechanism as base
import task1_teammate_model as teammate


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "task1_teammate_statespace"


@dataclass(frozen=True)
class StateSpaceFit:
    """极大似然估计结果的简洁记录。"""

    params: dict[str, float]
    neg_loglike: float
    success: bool
    message: str
    iterations: int


def prepare_log_observations(panel: pd.DataFrame, start_date: str) -> pd.DataFrame:
    """为状态空间模型准备 Brent、WTI、Dubai 的对数观测值。"""

    df = panel[panel["date"] >= pd.Timestamp(start_date)].copy()
    prices = df[["brent", "wti", "dubai"]].copy()
    prices = prices.mask(prices <= 0)
    prices = prices.interpolate(limit_direction="both").ffill().bfill()
    df[["brent_log", "wti_log", "dubai_log"]] = np.log(prices)
    return df.dropna(subset=["brent_log", "wti_log", "dubai_log"]).reset_index(drop=True)


def unpack(theta: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """把优化器参数转换为状态空间矩阵。

    优化器使用变换后的参数，以保证方差为正、相关系数在合理范围内。
    本函数将这些参数映射成 A、c、Q、H、R。
    """

    mu = theta[0]
    alpha = theta[1]
    kappa = np.exp(theta[2])
    sigma1 = np.exp(theta[3])
    sigma2 = np.exp(theta[4])
    rho = np.tanh(theta[5])
    h = theta[6:9]
    obs_sigma = np.exp(theta[9:12])

    a = np.array([[1.0, -dt], [0.0, 1.0 - kappa * dt]], dtype=float)
    c = np.array([(mu - 0.5 * sigma1 * sigma1) * dt, kappa * alpha * dt], dtype=float)
    q = np.array(
        [
            [sigma1 * sigma1 * dt, rho * sigma1 * sigma2 * dt],
            [rho * sigma1 * sigma2 * dt, sigma2 * sigma2 * dt],
        ],
        dtype=float,
    )
    hmat = np.column_stack([np.ones(3), h])
    r = np.diag(obs_sigma * obs_sigma)
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
    return a, c, q, hmat, r, named


def kalman_filter(
    y: np.ndarray,
    theta: np.ndarray,
    dt: float,
    return_states: bool = False,
) -> tuple[float, np.ndarray | None]:
    """计算 Kalman 滤波对数似然，并可选返回滤波状态。"""

    a, c, q, hmat, r, _ = unpack(theta, dt)
    n = y.shape[0]
    state = np.array([float(np.nanmean(y[:, 0])), 0.0], dtype=float)
    cov = np.diag([0.25, 0.25])
    identity = np.eye(2)
    loglike = 0.0
    filtered = np.zeros((n, 2), dtype=float)

    for t in range(n):
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

        loglike += -0.5 * (3 * np.log(2 * np.pi) + logdet + innovation @ solved)
        gain = cov @ hmat.T @ np.linalg.inv(innovation_cov)
        state = state + gain @ innovation
        cov = (identity - gain @ hmat) @ cov @ (identity - gain @ hmat).T + gain @ r @ gain.T
        cov = (cov + cov.T) / 2.0
        filtered[t] = state

        if not np.all(np.isfinite(state)) or not np.all(np.isfinite(cov)):
            return 1e12, None

    neg_loglike = float(-loglike)
    return (neg_loglike, filtered) if return_states else (neg_loglike, None)


def initial_theta(y: np.ndarray) -> np.ndarray:
    """构造数值上较稳定的极大似然初始值。"""

    log_returns = np.diff(y[:, 0])
    daily_vol = float(np.nanstd(log_returns))
    daily_vol = daily_vol if np.isfinite(daily_vol) and daily_vol > 1e-5 else 0.02
    annual_vol = daily_vol * np.sqrt(252)
    return np.array(
        [
            0.0,
            0.0,
            np.log(1.0),
            np.log(max(annual_vol, 0.05)),
            np.log(0.20),
            0.0,
            0.10,
            0.10,
            0.10,
            np.log(0.03),
            np.log(0.03),
            np.log(0.03),
        ],
        dtype=float,
    )


def fit_teammate_state_space(oil_df: pd.DataFrame, maxiter: int) -> tuple[pd.DataFrame, StateSpaceFit]:
    """用极大似然估计严格双因子模型。"""

    y = oil_df[["brent_log", "wti_log", "dubai_log"]].to_numpy(dtype=float)
    dt = 1.0 / 252.0
    theta0 = initial_theta(y)
    bounds = [
        (-1.0, 1.0),
        (-2.0, 2.0),
        (np.log(0.05), np.log(20.0)),
        (np.log(0.01), np.log(2.0)),
        (np.log(0.01), np.log(2.0)),
        (-3.0, 3.0),
        (-3.0, 3.0),
        (-3.0, 3.0),
        (-3.0, 3.0),
        (np.log(0.002), np.log(0.50)),
        (np.log(0.002), np.log(0.50)),
        (np.log(0.002), np.log(0.50)),
    ]

    result = minimize(
        lambda th: kalman_filter(y, th, dt, return_states=False)[0],
        theta0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": maxiter, "ftol": 1e-7, "maxls": 30},
    )
    neg_ll, filtered = kalman_filter(y, result.x, dt, return_states=True)
    _, _, _, _, _, named = unpack(result.x, dt)

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


def run_validation(config: base.MechanismConfig, maxiter: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """估计潜在油价状态，并用队友原公式验证调价效果。"""

    panel = base.read_oil_panel()
    oil_df = prepare_log_observations(panel, config.start_date)
    state_panel, fit = fit_teammate_state_space(oil_df, maxiter=maxiter)
    domestic = base.read_domestic_adjustments(config.start_date)
    validation = teammate.simulate_teammate_control(domestic, state_panel, "state_space_index_usd", config)
    summary = {
        "state_space_fit": asdict(fit),
        "metrics": teammate.build_summary(validation),
        "nardl_like": {
            "gasoline": base.run_nardl_like_test(validation, "gasoline"),
            "diesel": base.run_nardl_like_test(validation, "diesel"),
        },
    }
    return state_panel, validation, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict teammate two-factor state-space validation.")
    parser.add_argument("--start-date", default=base.MechanismConfig.start_date)
    parser.add_argument("--window-size", type=int, default=base.MechanismConfig.window_size)
    parser.add_argument("--pricing-lag-days", type=int, default=0)
    parser.add_argument("--maxiter", type=int, default=250)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = base.MechanismConfig(
        start_date=args.start_date,
        window_size=args.window_size,
        pricing_lag_days=args.pricing_lag_days,
    )
    state_panel, validation, summary = run_validation(config, maxiter=args.maxiter)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    state_panel.to_csv(OUTPUT_DIR / "state_space_oil_index.csv", index=False, encoding="utf-8-sig")
    validation.to_csv(OUTPUT_DIR / "state_space_teammate_validation.csv", index=False, encoding="utf-8-sig")
    with (OUTPUT_DIR / "summary_metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"config": asdict(config), "summary": summary}, f, ensure_ascii=False, indent=2)

    print("Strict teammate two-factor state-space validation")
    print("Fit:", summary["state_space_fit"])
    print("Gasoline:", summary["metrics"]["gasoline"])
    print("Diesel:", summary["metrics"]["diesel"])
    print("NARDL-like gasoline:", summary["nardl_like"]["gasoline"])
    print("NARDL-like diesel:", summary["nardl_like"]["diesel"])
    print(f"\nOutputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
