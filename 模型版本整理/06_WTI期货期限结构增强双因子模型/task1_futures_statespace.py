from __future__ import annotations

"""加入 WTI 期货期限结构的双因子状态空间模型。

本脚本沿着 Gemini 的改进方向：
1. 保留队友双因子状态向量 [X_t, delta_t]；
2. 将 WTI 期货 M1-M4 加入观测方程，使 delta_t 能利用期限结构信息，
   而不是只靠近似现货价格识别；
3. 使用优化机制模型中的修正调价控制层。

EIA 期货工作簿当前只到 2024-04-05，因此验证仅限于期货数据覆盖的
调价窗口。
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
FUTURES_PATH = ROOT / "PET_PRI_FUT_S1_D.xls"
OUTPUT_DIR = ROOT / "outputs" / "task1_futures_statespace"


@dataclass(frozen=True)
class FuturesFit:
    """期货增强版极大似然估计结果的简洁记录。"""

    params: dict[str, float]
    neg_loglike: float
    success: bool
    message: str
    iterations: int
    obs_cols: list[str]


def read_wti_futures() -> pd.DataFrame:
    """从 PET_PRI_FUT_S1_D.xls 读取 EIA 的 WTI 期货 1-4 号合约。"""

    df = pd.read_excel(FUTURES_PATH, sheet_name="Data 1", header=2)
    df = df.rename(
        columns={
            df.columns[0]: "date",
            df.columns[1]: "wti_m1",
            df.columns[2]: "wti_m2",
            df.columns[3]: "wti_m3",
            df.columns[4]: "wti_m4",
        }
    )
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["wti_m1", "wti_m2", "wti_m3", "wti_m4"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["date"]).sort_values("date")


def build_observation_panel(start_date: str) -> tuple[pd.DataFrame, list[str]]:
    """合并原油现货价格和 WTI 期货期限结构，并转为对数观测。"""

    panel = base.read_oil_panel()
    futures = read_wti_futures()
    df = panel.merge(futures, on="date", how="inner")
    df = df[df["date"] >= pd.Timestamp(start_date)].copy()
    obs_cols = ["brent", "wti", "dubai", "wti_m1", "wti_m2", "wti_m3", "wti_m4"]
    df[obs_cols] = df[obs_cols].mask(df[obs_cols] <= 0)
    df[obs_cols] = df[obs_cols].interpolate(limit_direction="both").ffill().bfill()
    for col in obs_cols:
        df[f"log_{col}"] = np.log(df[col])
    return df.dropna(subset=[f"log_{col}" for col in obs_cols]).reset_index(drop=True), [f"log_{c}" for c in obs_cols]


def unpack(theta: np.ndarray, n_obs: int, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """把优化器参数映射为状态空间矩阵。"""

    mu = theta[0]
    alpha = theta[1]
    kappa = np.exp(theta[2])
    sigma1 = np.exp(theta[3])
    sigma2 = np.exp(theta[4])
    h = theta[5 : 5 + n_obs]
    obs_sigma = np.exp(theta[5 + n_obs : 5 + 2 * n_obs])

    a = np.array([[1.0, -dt], [0.0, 1.0 - kappa * dt]], dtype=float)
    c = np.array([(mu - 0.5 * sigma1 * sigma1) * dt, kappa * alpha * dt], dtype=float)
    # 固定 rho = 0，用来降低弱识别和局部最优问题。
    q = np.array([[sigma1 * sigma1 * dt, 0.0], [0.0, sigma2 * sigma2 * dt]], dtype=float)
    hmat = np.column_stack([np.ones(n_obs), h])
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
    dt: float,
    return_states: bool = False,
    penalty_weight: float = 0.0,
) -> tuple[float, np.ndarray | None]:
    """带可选 delta 惩罚项的双因子 Kalman 似然。"""

    n_obs = y.shape[1]
    a, c, q, hmat, r, _ = unpack(theta, n_obs, dt)
    n = y.shape[0]
    state = np.array([float(np.nanmean(y[:, 0])), 0.0], dtype=float)
    cov = np.diag([0.25, 0.25])
    identity = np.eye(2)
    loglike = 0.0
    penalty = 0.0
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

        loglike += -0.5 * (n_obs * np.log(2 * np.pi) + logdet + innovation @ solved)
        gain = cov @ hmat.T @ np.linalg.inv(innovation_cov)
        state = state + gain @ innovation
        cov = (identity - gain @ hmat) @ cov @ (identity - gain @ hmat).T + gain @ r @ gain.T
        cov = (cov + cov.T) / 2.0
        filtered[t] = state
        penalty += state[1] * state[1]

        if not np.all(np.isfinite(state)) or not np.all(np.isfinite(cov)):
            return 1e12, None

    neg_loglike = float(-loglike + penalty_weight * penalty / n)
    return (neg_loglike, filtered) if return_states else (neg_loglike, None)


def initial_theta(y: np.ndarray) -> np.ndarray:
    """极大似然优化器的初始值。"""

    n_obs = y.shape[1]
    daily_vol = float(np.nanstd(np.diff(y[:, 0])))
    annual_vol = max(daily_vol * np.sqrt(252), 0.05)
    h0 = np.linspace(0.4, -0.4, n_obs)
    obs_sigma0 = np.full(n_obs, 0.03)
    return np.r_[
        0.0,
        0.0,
        np.log(1.0),
        np.log(annual_vol),
        np.log(0.20),
        h0,
        np.log(obs_sigma0),
    ].astype(float)


def fit_futures_state_space(
    obs_panel: pd.DataFrame,
    obs_cols: list[str],
    maxiter: int,
    penalty_weight: float,
) -> tuple[pd.DataFrame, FuturesFit]:
    """用现货和 WTI 期货观测估计双因子模型。

    WTI 期货 M1-M4 提供期限结构信息，这是尝试识别短期 delta_t 状态
    更合适的数据类型。
    """

    y = obs_panel[obs_cols].to_numpy(dtype=float)
    dt = 1.0 / 252.0
    theta0 = initial_theta(y)
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
        lambda th: kalman_filter(y, th, dt, return_states=False, penalty_weight=penalty_weight)[0],
        theta0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": maxiter, "ftol": 1e-7, "maxls": 25},
    )
    neg_ll, filtered = kalman_filter(y, result.x, dt, return_states=True, penalty_weight=penalty_weight)
    _, _, _, _, _, named = unpack(result.x, n_obs, dt)
    h = result.x[5 : 5 + n_obs]
    obs_sigma = np.exp(result.x[5 + n_obs : 5 + 2 * n_obs])
    for col, value in zip(obs_cols, h):
        named[f"h_{col}"] = float(value)
    for col, value in zip(obs_cols, obs_sigma):
        named[f"obs_sigma_{col}"] = float(value)

    out = obs_panel[["date", "usd_cny", "brent", "wti", "dubai", "wti_m1", "wti_m2", "wti_m3", "wti_m4"]].copy()
    out["state_x"] = filtered[:, 0]
    out["state_delta"] = filtered[:, 1]
    out["state_space_futures_index_usd"] = np.exp(out["state_x"])
    fit = FuturesFit(
        params=named,
        neg_loglike=float(neg_ll),
        success=bool(result.success),
        message=str(result.message),
        iterations=int(result.nit),
        obs_cols=obs_cols,
    )
    return out, fit


def run_validation(config: base.MechanismConfig, maxiter: int, penalty_weight: float) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """拟合期货增强状态模型，并用修正机制层验证调价结果。

    验证仅限于期货数据覆盖的窗口；EIA 工作簿截至 2024-04-05。
    """

    obs_panel, obs_cols = build_observation_panel(config.start_date)
    state_panel, fit = fit_futures_state_space(obs_panel, obs_cols, maxiter=maxiter, penalty_weight=penalty_weight)
    domestic = base.read_domestic_adjustments(config.start_date)
    validation = base.simulate_mechanism(domestic, state_panel, "state_space_futures_index_usd", config)
    # 只保留计价截止日被期货数据覆盖的调价窗口。否则 trailing_window
    # 会把 2024 年最后一期期货数据反复用于 2025-2026 年窗口。
    max_futures_date = state_panel["date"].max()
    validation = validation[pd.to_datetime(validation["pricing_end"]) <= max_futures_date].reset_index(drop=True)

    summary = {
        "config": asdict(config),
        "state_space_fit": asdict(fit),
        "futures_data_end": str(state_panel["date"].max().date()),
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
    parser = argparse.ArgumentParser(description="WTI futures term-structure two-factor state-space validation.")
    parser.add_argument("--start-date", default=base.MechanismConfig.start_date)
    parser.add_argument("--train-end", default=base.MechanismConfig.train_end)
    parser.add_argument("--window-size", type=int, default=base.MechanismConfig.window_size)
    parser.add_argument("--pricing-lag-days", type=int, default=1)
    parser.add_argument("--maxiter", type=int, default=250)
    parser.add_argument("--penalty-weight", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = base.MechanismConfig(
        start_date=args.start_date,
        train_end=args.train_end,
        window_size=args.window_size,
        pricing_lag_days=args.pricing_lag_days,
    )
    state_panel, validation, summary = run_validation(
        config,
        maxiter=args.maxiter,
        penalty_weight=args.penalty_weight,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    state_panel.to_csv(OUTPUT_DIR / "futures_state_space_oil_index.csv", index=False, encoding="utf-8-sig")
    validation.to_csv(OUTPUT_DIR / "futures_state_space_validation.csv", index=False, encoding="utf-8-sig")
    with (OUTPUT_DIR / "summary_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("WTI futures-enhanced two-factor state-space validation")
    print("Fit:", summary["state_space_fit"])
    print("Futures data end:", summary["futures_data_end"])
    print("Gasoline:", summary["metrics"]["gasoline"])
    print("Diesel:", summary["metrics"]["diesel"])
    print("Test after train gasoline:", summary["metrics"]["test_after_train"]["gasoline"])
    print("Test after train diesel:", summary["metrics"]["test_after_train"]["diesel"])
    print(f"\nOutputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
