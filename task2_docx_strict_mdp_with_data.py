"""Strict docx-style Task 2 MDP baseline with cleaned data.

This script follows ``任务二(1).docx`` more literally than the rolling-horizon
main model:

1. five-part social welfare loss;
2. Markov state transition equations;
3. Bellman equation solved by approximate value iteration on discretized states.

Some parts of the document are not directly observable in the public data or are
not Markov with the four listed states. These are implemented as transparent
numerical approximations and written to the report.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import task2_dynamic_pricing_dp as data_model


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs" / "task2_dynamic_pricing" / "docx_strict_mdp"


@dataclass(frozen=True)
class StrictCalibration:
    c0: float
    rho_c: float
    sigma_c: float
    alpha0: float
    alpha1: float
    theta_cpi: float
    epsilon_demand: float
    target_margin: float
    beta: float
    tau_security: float
    weights: dict[str, float]
    transport_share_mean: float
    fx_median: float
    refining_cost: float
    action_grid: list[float]


@dataclass
class StrictDPResult:
    fuel: str
    oil_grid: np.ndarray
    price_grid: np.ndarray
    inv_grid: np.ndarray
    cpi_grid: np.ndarray
    prev_action_grid: np.ndarray
    value: np.ndarray
    policy: np.ndarray
    calibration: StrictCalibration


def fit_cpi_transition_docx(df: pd.DataFrame) -> tuple[float, float, float]:
    pi = df["cpi_yoy"].to_numpy(dtype=float)
    price = df["domestic_price"].to_numpy(dtype=float)
    delta = df["actual_delta"].to_numpy(dtype=float)
    prev_price = price - delta
    rate = np.divide(delta[1:], prev_price[1:], out=np.zeros(len(delta) - 1), where=prev_price[1:] != 0)
    y = pi[1:]
    x = np.column_stack([np.ones(len(y)), pi[:-1], rate])
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    return float(beta[0]), float(beta[1]), float(beta[2])


def calibrate_docx(df: pd.DataFrame, fuel: str) -> StrictCalibration:
    c0, rho_c, sigma_c = data_model.fit_ar1_log_oil(df)
    alpha0, alpha1, theta_cpi = fit_cpi_transition_docx(df)
    transport_share_mean = float(df["transport_share"].dropna().mean())
    raw_weights = {
        "consumer": transport_share_mean,
        "profit": 0.055,
        "cpi": 0.22,
        "expectation": 0.18,
        "security": 0.46,
    }
    weight_sum = sum(raw_weights.values())
    weights = {k: v / weight_sum for k, v in raw_weights.items()}

    target_margin = 0.06
    fx_median = float(df["exchange_rate"].median())
    median_price = float(df["domestic_price"].median())
    median_import_cost = float(df["import_cost_rmb_ton"].median())
    refining_cost = max(300.0, median_price * (1 - target_margin) - median_import_cost)

    return StrictCalibration(
        c0=c0,
        rho_c=rho_c,
        sigma_c=max(sigma_c, 1e-4),
        alpha0=alpha0,
        alpha1=alpha1,
        theta_cpi=theta_cpi,
        epsilon_demand=-0.6,
        target_margin=target_margin,
        beta=0.98,
        tau_security=2.0,
        weights=weights,
        transport_share_mean=transport_share_mean,
        fx_median=fx_median,
        refining_cost=refining_cost,
        action_grid=[-400, -200, -100, 0, 100, 200, 400],
    )


def make_grid(series: pd.Series, n: int, pad: float = 0.08) -> np.ndarray:
    lo = float(series.quantile(0.03))
    hi = float(series.quantile(0.97))
    spread = max(hi - lo, 1.0)
    return np.linspace(lo - pad * spread, hi + pad * spread, n)


def nearest_idx(grid: np.ndarray, value: float) -> int:
    return int(np.abs(grid - value).argmin())


def strict_single_period_loss(
    price_prev: float,
    action: float,
    action_prev: float,
    inventory_cost: float,
    cpi_prev: float,
    quantity: float,
    calib: StrictCalibration,
) -> tuple[float, dict[str, float], float]:
    price = max(price_prev + action, 1.0)
    equilibrium_price = inventory_cost / max(1.0 - calib.target_margin, 1e-6)
    margin = (price - inventory_cost) / price
    cpi_now = calib.alpha0 + calib.alpha1 * cpi_prev + calib.theta_cpi * (action / max(price_prev, 1.0))

    loss_cs_raw = data_model.consumer_loss(price, equilibrium_price, quantity, calib.epsilon_demand)
    loss_consumer = loss_cs_raw / max(quantity * price_prev, 1.0)
    loss_profit = (margin - calib.target_margin) ** 2
    loss_cpi = (cpi_now - 0.02) ** 2
    loss_expectation = ((action - action_prev) / 500.0) ** 2

    supply_ratio = max((price - inventory_cost) / max(inventory_cost, 1.0), -1.0)
    demand = quantity * (price / max(price_prev, 1.0)) ** calib.epsilon_demand
    supply = quantity * max(0.0, 1.0 + 2.0 * supply_ratio)
    gap_ratio = max(0.0, demand - supply) / max(quantity, 1.0)
    loss_security = math.exp(calib.tau_security * gap_ratio) - 1.0

    parts = {
        "consumer": loss_consumer,
        "profit": loss_profit,
        "cpi": loss_cpi,
        "expectation": loss_expectation,
        "security": loss_security,
    }
    total = sum(calib.weights[k] * parts[k] for k in parts)
    return float(total), parts, float(cpi_now)


def transition_expectation(
    oil: float,
    price: float,
    inv: float,
    cpi: float,
    action: float,
    calib: StrictCalibration,
) -> list[tuple[float, float, float, float, float]]:
    shocks = [(-calib.sigma_c, 0.25), (0.0, 0.5), (calib.sigma_c, 0.25)]
    out = []
    for shock, prob in shocks:
        log_oil_next = calib.c0 + calib.rho_c * math.log(max(oil, 1.0)) + shock
        oil_next = math.exp(log_oil_next)
        price_next = max(price + action, 1.0)
        crude_cost_next = oil_next * calib.fx_median * 7.33 + calib.refining_cost
        # Approximation of the document's three-lag inventory equation without
        # expanding the state to include all lagged procurement costs.
        inv_next = 0.65 * inv + 0.35 * crude_cost_next
        cpi_next = calib.alpha0 + calib.alpha1 * cpi + calib.theta_cpi * (action / max(price, 1.0))
        out.append((prob, oil_next, price_next, inv_next, cpi_next))
    return out


def solve_docx_value_iteration(df: pd.DataFrame, fuel: str, max_iter: int = 35, tol: float = 5e-5) -> StrictDPResult:
    calib = calibrate_docx(df, fuel)
    oil_grid = make_grid(df["oil_price_usd_bbl"], 4)
    price_grid = make_grid(df["domestic_price"], 5)
    inv_grid = make_grid(df["inventory_cost"], 4)
    cpi_grid = make_grid(df["cpi_yoy"], 4)
    prev_action_grid = np.array(calib.action_grid, dtype=float)
    shape = (len(oil_grid), len(price_grid), len(inv_grid), len(cpi_grid), len(prev_action_grid))
    value = np.zeros(shape, dtype=float)
    policy = np.zeros(shape, dtype=float)
    quantity_ref = float(df["quantity_tonnes"].median())

    for _ in range(max_iter):
        old = value.copy()
        max_diff = 0.0
        for idx in np.ndindex(shape):
            io, ip, ii, ic, ia_prev = idx
            oil = float(oil_grid[io])
            price = float(price_grid[ip])
            inv = float(inv_grid[ii])
            cpi = float(cpi_grid[ic])
            action_prev = float(prev_action_grid[ia_prev])

            best_val = math.inf
            best_action = 0.0
            for action in calib.action_grid:
                loss, _, _ = strict_single_period_loss(
                    price_prev=price,
                    action=float(action),
                    action_prev=action_prev,
                    inventory_cost=inv,
                    cpi_prev=cpi,
                    quantity=quantity_ref,
                    calib=calib,
                )
                expected_value = 0.0
                for prob, oil_next, price_next, inv_next, cpi_next in transition_expectation(
                    oil, price, inv, cpi, float(action), calib
                ):
                    next_idx = (
                        nearest_idx(oil_grid, oil_next),
                        nearest_idx(price_grid, price_next),
                        nearest_idx(inv_grid, inv_next),
                        nearest_idx(cpi_grid, cpi_next),
                        nearest_idx(prev_action_grid, float(action)),
                    )
                    expected_value += prob * old[next_idx]
                candidate = loss + calib.beta * expected_value
                if candidate < best_val:
                    best_val = candidate
                    best_action = float(action)

            value[idx] = best_val
            policy[idx] = best_action
            max_diff = max(max_diff, abs(best_val - old[idx]))
        if max_diff < tol:
            break

    return StrictDPResult(
        fuel=fuel,
        oil_grid=oil_grid,
        price_grid=price_grid,
        inv_grid=inv_grid,
        cpi_grid=cpi_grid,
        prev_action_grid=prev_action_grid,
        value=value,
        policy=policy,
        calibration=calib,
    )


def policy_action(dp: StrictDPResult, oil: float, price: float, inv: float, cpi: float, action_prev: float) -> float:
    idx = (
        nearest_idx(dp.oil_grid, oil),
        nearest_idx(dp.price_grid, price),
        nearest_idx(dp.inv_grid, inv),
        nearest_idx(dp.cpi_grid, cpi),
        nearest_idx(dp.prev_action_grid, action_prev),
    )
    return float(dp.policy[idx])


def simulate_docx_policy(df: pd.DataFrame, dp: StrictDPResult) -> pd.DataFrame:
    rows = []
    price_opt = float(df.loc[0, "domestic_price"])
    action_prev = 0.0
    actual_action_prev = 0.0
    mechanism_action_prev = 0.0
    cpi_prev = float(df.loc[0, "cpi_yoy"])

    for _, row in df.iterrows():
        current_prev_price = float(row["domestic_price"] - row["actual_delta"])
        actual_loss, actual_parts, cpi_actual = strict_single_period_loss(
            current_prev_price,
            float(row["actual_delta"]),
            actual_action_prev,
            float(row["inventory_cost"]),
            cpi_prev,
            float(row["quantity_tonnes"]),
            dp.calibration,
        )
        mechanism_delta = float(row["current_mechanism_delta"])
        mechanism_loss, mechanism_parts, cpi_mechanism = strict_single_period_loss(
            current_prev_price,
            mechanism_delta,
            mechanism_action_prev,
            float(row["inventory_cost"]),
            cpi_prev,
            float(row["quantity_tonnes"]),
            dp.calibration,
        )

        action = policy_action(
            dp,
            float(row["oil_price_usd_bbl"]),
            price_opt,
            float(row["inventory_cost"]),
            cpi_prev,
            action_prev,
        )
        optimal_loss, optimal_parts, cpi_opt = strict_single_period_loss(
            price_opt,
            action,
            action_prev,
            float(row["inventory_cost"]),
            cpi_prev,
            float(row["quantity_tonnes"]),
            dp.calibration,
        )
        price_opt = max(price_opt + action, 1.0)
        action_prev = action
        actual_action_prev = float(row["actual_delta"])
        mechanism_action_prev = mechanism_delta
        cpi_prev = cpi_opt

        rows.append(
            {
                "adjust_date": row["adjust_date"],
                "fuel": dp.fuel,
                "scenario": "docx_strict_mdp_value_iteration",
                "oil_price_usd_bbl": row["oil_price_usd_bbl"],
                "inventory_cost": row["inventory_cost"],
                "quantity_tonnes": row["quantity_tonnes"],
                "transport_share": row["transport_share"],
                "actual_delta": row["actual_delta"],
                "mechanism_delta": mechanism_delta,
                "optimal_delta": action,
                "carry_over_next": 0.0,
                "actual_price": row["domestic_price"],
                "optimal_price": price_opt,
                "actual_loss": actual_loss,
                "mechanism_loss": mechanism_loss,
                "optimal_loss": optimal_loss,
                "actual_cpi_pred": cpi_actual,
                "mechanism_cpi_pred": cpi_mechanism,
                "optimal_cpi_pred": cpi_opt,
                **{f"actual_loss_{k}": v for k, v in actual_parts.items()},
                **{f"mechanism_loss_{k}": v for k, v in mechanism_parts.items()},
                **{f"optimal_loss_{k}": v for k, v in optimal_parts.items()},
            }
        )
    return pd.DataFrame(rows)


def write_report(summary: pd.DataFrame) -> None:
    lines = [
        "任务二严格docx版MDP对照结果",
        "=" * 50,
        "",
        "已严格落实的部分：",
        "- 五维社会福利损失：消费者、利润、CPI、预期平滑、能源安全。",
        "- 状态变量：国际油价、国内价格、库存成本、CPI，并额外加入上一期动作以保证二阶差分损失满足马尔可夫性。",
        "- 转移方程：国际油价AR(1)、国内价格P'=P+a、库存成本动态更新、CPI传导方程。",
        "- 求解方法：离散状态空间上的贝尔曼方程近似值迭代。",
        "",
        "无法完全按原文无损落实的部分及原因：",
        "- docx列出的4维状态不足以计算预期平滑损失中的价格二阶差分，因此代码加入上一期动作作为辅助状态。",
        "- 库存成本原文需要3期原油采购成本和库存周转权重；公开数据没有炼厂存货周转天数，且若严格加入3期滞后成本会显著扩展状态维度，因此用当前库存成本与下一期原油采购成本的加权更新近似。",
        "- 原文贝尔曼方程是连续高维状态和随机期望；代码采用离散网格、三点油价冲击近似期望。",
        "- WACC、炼油单位加工成本、实际供需缺口没有直接公开月度数据，仍用前面已整理数据和校准参数近似。",
        "",
        "summary：",
        summary.to_string(index=False),
    ]
    (OUT_DIR / "task2_docx_strict_mdp_report.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []
    sims = []
    for fuel in ["gasoline", "diesel"]:
        df = data_model.make_model_data(fuel)  # type: ignore[arg-type]
        dp = solve_docx_value_iteration(df, fuel)
        sim = simulate_docx_policy(df, dp)
        sim.to_csv(OUT_DIR / f"task2_docx_strict_mdp_{fuel}_strategy.csv", index=False, encoding="utf-8-sig")
        sims.append(sim)
        summary = data_model.summarize(sim)
        summary["scenario"] = "docx_strict_mdp_value_iteration"
        summary["fuel"] = fuel
        summaries.append(summary)
        (OUT_DIR / f"task2_docx_strict_mdp_{fuel}_calibration.json").write_text(
            json.dumps(
                {
                    "weights": dp.calibration.weights,
                    "action_grid": dp.calibration.action_grid,
                    "beta": dp.calibration.beta,
                    "tau_security": dp.calibration.tau_security,
                    "target_margin": dp.calibration.target_margin,
                    "transport_share_mean": dp.calibration.transport_share_mean,
                    "grid_shape": list(dp.value.shape),
                    "state_note": "oil, price, inventory_cost, cpi, previous_action",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OUT_DIR / "task2_docx_strict_mdp_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(sims, ignore_index=True).to_csv(
        OUT_DIR / "task2_docx_strict_mdp_all_fuels_strategy.csv", index=False, encoding="utf-8-sig"
    )
    write_report(summary_df)
    print(summary_df.to_string(index=False))
    print(f"\n已输出目录: {OUT_DIR}")


if __name__ == "__main__":
    main()
