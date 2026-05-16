"""Task 2: dynamic optimal pricing strategy.

Implementation follows ``任务二(1).docx`` as closely as available data allow:

1. Build a five-part social welfare loss:
   consumer surplus loss, refinery profit protection loss, CPI loss,
   expectation/smoothing loss, and energy security loss.
2. Define state variables:
   international crude price, previous domestic retail price,
   refinery inventory cost, previous CPI inflation.
3. Define control:
   actual adjustment amount of domestic refined-oil price.
4. Estimate crude oil AR(1) transition and CPI pass-through from historical data.
5. Solve a finite rolling-horizon dynamic optimization problem.
6. Simulate the optimized strategy on historical task-1 windows and compare it with
   the current mechanism.

The script creates outputs under outputs/task2_dynamic_pricing/.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import json
import math

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
TASK1_CSV = ROOT / "outputs" / "task1_serial_fusion_ridge" / "serial_fusion_validation.csv"
CPI_XLSX = ROOT / "国内的一些数据" / "2026-2008_cpi.xlsx"
IMPORT_CSV = ROOT / "海关进出口数量数据" / "进口原油数量和金额_合并总表.csv"
OUT_DIR = ROOT / "outputs" / "task2_dynamic_pricing"
APPARENT_CSV = OUT_DIR / "task2_apparent_consumption_monthly.csv"
TRANSPORT_EXP_CSV = ROOT / "data" / "task2" / "resident_transport_communication_expenditure.csv"
TOTAL_EXP_CSV = ROOT / "data" / "task2" / "resident_consumption_expenditure_per_capita.csv"

Fuel = Literal["gasoline", "diesel"]


@dataclass(frozen=True)
class Calibration:
    c0: float
    rho_c: float
    sigma_c: float
    alpha0: float
    alpha1: float
    theta0_cpi: float
    theta1_cpi: float
    epsilon_demand: float
    target_margin: float
    beta: float
    tau_security: float
    weights: dict[str, float]
    ratio_grid: list[float]
    transport_share_mean: float
    security_floor_ratio: float
    security_penalty_mild: float
    security_penalty_severe: float
    horizon: int
    scenario: str


@dataclass
class DPResult:
    fuel: Fuel
    oil_grid: np.ndarray
    price_grid: np.ndarray
    inv_grid: np.ndarray
    cpi_grid: np.ndarray
    value: np.ndarray
    policy: np.ndarray
    calibration: Calibration


def load_task1_data() -> pd.DataFrame:
    df = pd.read_csv(TASK1_CSV, parse_dates=["adjust_date", "window_start", "window_end"])
    df = df.sort_values("adjust_date").reset_index(drop=True)
    df["month"] = df["adjust_date"].dt.to_period("M").astype(str)
    return df


def load_monthly_cpi() -> pd.DataFrame:
    cpi = pd.read_excel(CPI_XLSX)
    cpi = cpi.rename(columns={"月份": "month", "全国_同比增长": "cpi_yoy", "全国_环比增长": "cpi_mom"})
    cpi["month"] = cpi["month"].astype(str)
    return cpi[["month", "cpi_yoy", "cpi_mom"]]


def load_import_cost() -> pd.DataFrame:
    imp = pd.read_csv(IMPORT_CSV)
    imp["month"] = imp["年"].astype(str) + "-" + imp["月"].astype(int).astype(str).str.zfill(2)
    imp["import_tonnes"] = pd.to_numeric(imp["第一数量"], errors="coerce") / 1000.0
    imp["import_cost_rmb_ton"] = pd.to_numeric(imp["每吨人民币"], errors="coerce")
    return imp[["month", "import_tonnes", "import_cost_rmb_ton"]]


def load_apparent_consumption() -> pd.DataFrame:
    if not APPARENT_CSV.exists():
        return pd.DataFrame(columns=["month", "gasoline_quantity_tonnes", "diesel_quantity_tonnes"])
    df = pd.read_csv(APPARENT_CSV)
    out = pd.DataFrame({"month": df["month"].astype(str)})
    for fuel in ["gasoline", "diesel"]:
        col = f"{fuel}_apparent_consumption"
        out[f"{fuel}_quantity_tonnes"] = pd.to_numeric(df[col], errors="coerce") * 10_000.0
    return out


def load_resident_expenditure() -> pd.DataFrame:
    transport = pd.read_csv(TRANSPORT_EXP_CSV)
    total = pd.read_csv(TOTAL_EXP_CSV)
    df = transport.merge(total, on="year", how="outer").sort_values("year")
    df["transport_communication_expenditure_per_capita"] = pd.to_numeric(
        df["transport_communication_expenditure_per_capita"], errors="coerce"
    )
    df["resident_consumption_expenditure_per_capita"] = pd.to_numeric(
        df["resident_consumption_expenditure_per_capita"], errors="coerce"
    )
    df[[
        "transport_communication_expenditure_per_capita",
        "resident_consumption_expenditure_per_capita",
    ]] = df[[
        "transport_communication_expenditure_per_capita",
        "resident_consumption_expenditure_per_capita",
    ]].interpolate(limit_direction="both").ffill().bfill()
    df["transport_share"] = (
        df["transport_communication_expenditure_per_capita"]
        / df["resident_consumption_expenditure_per_capita"]
    )
    return df[[
        "year",
        "transport_communication_expenditure_per_capita",
        "resident_consumption_expenditure_per_capita",
        "transport_share",
    ]]


def make_model_data(fuel: Fuel) -> pd.DataFrame:
    df = load_task1_data()
    cpi = load_monthly_cpi()
    imp = load_import_cost()
    apparent = load_apparent_consumption()
    expenditure = load_resident_expenditure()
    df["year"] = df["adjust_date"].dt.year
    df = (
        df.merge(cpi, on="month", how="left")
        .merge(imp, on="month", how="left")
        .merge(apparent, on="month", how="left")
        .merge(expenditure, on="year", how="left")
    )

    df["cpi_yoy"] = df["cpi_yoy"].interpolate(limit_direction="both").ffill().bfill()
    df["cpi_mom"] = df["cpi_mom"].interpolate(limit_direction="both").ffill().bfill()
    df["import_tonnes"] = df["import_tonnes"].interpolate(limit_direction="both").ffill().bfill()
    df["import_cost_rmb_ton"] = df["import_cost_rmb_ton"].interpolate(limit_direction="both").ffill().bfill()
    df["transport_share"] = df["transport_share"].interpolate(limit_direction="both").ffill().bfill()

    df["oil_price_usd_bbl"] = pd.to_numeric(df["ma_usd_per_bbl"], errors="coerce")
    df["exchange_rate"] = pd.to_numeric(df["avg_usd_cny"], errors="coerce")
    df["domestic_price"] = pd.to_numeric(df[f"{fuel}_price"], errors="coerce")
    df["actual_delta"] = pd.to_numeric(df[f"{fuel}_actual_delta"], errors="coerce")
    df["current_mechanism_delta"] = pd.to_numeric(df[f"{fuel}_serial_fusion_delta"], errors="coerce")

    # Import cost already includes crude procurement cost in RMB/ton. The document
    # adds refining cost separately; we calibrate it from historical domestic price
    # and import cost so the median gross margin is close to the target margin.
    target_margin = 0.06
    median_price = float(df["domestic_price"].median())
    median_import_cost = float(df["import_cost_rmb_ton"].median())
    refining_cost = max(300.0, median_price * (1 - target_margin) - median_import_cost)
    df["inventory_cost"] = (
        0.5 * df["import_cost_rmb_ton"]
        + 0.3 * df["import_cost_rmb_ton"].shift(1)
        + 0.2 * df["import_cost_rmb_ton"].shift(2)
    ).bfill() + refining_cost

    # Use gasoline/diesel apparent consumption as the refined-oil demand scale.
    # If the constructed table is unavailable, fall back to monthly crude import.
    windows_per_month = df.groupby("month")["adjust_date"].transform("count").clip(lower=1)
    fuel_quantity_col = f"{fuel}_quantity_tonnes"
    if fuel_quantity_col in df.columns:
        df[fuel_quantity_col] = pd.to_numeric(df[fuel_quantity_col], errors="coerce")
        df[fuel_quantity_col] = df[fuel_quantity_col].interpolate(limit_direction="both").ffill().bfill()
        df["quantity_tonnes"] = df[fuel_quantity_col] / windows_per_month
    else:
        df["quantity_tonnes"] = df["import_tonnes"] / windows_per_month
    return df.dropna(subset=["oil_price_usd_bbl", "domestic_price", "inventory_cost", "cpi_yoy"]).reset_index(drop=True)


def fit_ar1_log_oil(df: pd.DataFrame) -> tuple[float, float, float]:
    x = np.log(df["oil_price_usd_bbl"].to_numpy(dtype=float))
    y = x[1:]
    X = np.column_stack([np.ones(len(x) - 1), x[:-1]])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    return float(beta[0]), float(beta[1]), float(np.std(resid, ddof=2))


def fit_cpi_transition(df: pd.DataFrame, fuel: Fuel) -> tuple[float, float, float, float]:
    pi = df["cpi_yoy"].to_numpy(dtype=float)
    price = df["domestic_price"].to_numpy(dtype=float)
    delta = df["actual_delta"].to_numpy(dtype=float)
    prev_price = price - delta
    price_change_rate = np.divide(delta, prev_price, out=np.zeros(len(delta)), where=prev_price != 0)
    y = pi[1:]
    X = np.column_stack([np.ones(len(y)), pi[:-1], price_change_rate[1:], price_change_rate[:-1]])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(beta[0]), float(beta[1]), float(beta[2]), float(beta[3])


def nearest_index(grid: np.ndarray, value: float) -> int:
    return int(np.abs(grid - value).argmin())


def interpolate_grid(values: pd.Series, n: int, pad: float = 0.05) -> np.ndarray:
    lo = float(values.quantile(0.02))
    hi = float(values.quantile(0.98))
    spread = hi - lo
    return np.linspace(lo - pad * spread, hi + pad * spread, n)


def scenario_raw_weights(scenario: str, transport_share_mean: float) -> dict[str, float]:
    if scenario == "baseline":
        return {
            "consumer": transport_share_mean,
            "profit": 0.055,
            "cpi": 0.22,
            "expectation": 0.18,
            "security": 0.46,
        }
    if scenario == "livelihood_priority":
        return {
            "consumer": transport_share_mean * 1.60,
            "profit": 0.045,
            "cpi": 0.20,
            "expectation": 0.17,
            "security": 0.36,
        }
    if scenario == "stability_priority":
        return {
            "consumer": transport_share_mean,
            "profit": 0.045,
            "cpi": 0.30,
            "expectation": 0.28,
            "security": 0.33,
        }
    if scenario == "security_priority":
        return {
            "consumer": transport_share_mean * 0.85,
            "profit": 0.08,
            "cpi": 0.18,
            "expectation": 0.14,
            "security": 0.60,
        }
    raise ValueError(f"unknown policy scenario: {scenario}")


def calibrate(
    df: pd.DataFrame,
    fuel: Fuel,
    *,
    scenario: str = "baseline",
    raw_weights: dict[str, float] | None = None,
    security_floor_ratio: float = 0.97,
    epsilon_demand: float = -0.6,
    security_penalty_mild: float = 8.0,
    security_penalty_severe: float = 35.0,
    horizon: int = 3,
) -> Calibration:
    c0, rho_c, sigma_c = fit_ar1_log_oil(df)
    alpha0, alpha1, theta0_cpi, theta1_cpi = fit_cpi_transition(df, fuel)
    ratio_grid = [0.0, 0.25, 0.5, 0.75, 1.0]

    # Weight calibration follows the document's idea: expenditure/industry shares
    # plus larger macro/security preference coefficients. Values are normalized.
    transport_share_mean = float(df["transport_share"].dropna().mean()) if "transport_share" in df else 0.085
    if raw_weights is None:
        raw_weights = scenario_raw_weights(scenario, transport_share_mean)
    s = sum(raw_weights.values())
    weights = {k: v / s for k, v in raw_weights.items()}

    return Calibration(
        c0=c0,
        rho_c=rho_c,
        sigma_c=max(sigma_c, 1e-4),
        alpha0=alpha0,
        alpha1=alpha1,
        theta0_cpi=theta0_cpi,
        theta1_cpi=theta1_cpi,
        epsilon_demand=epsilon_demand,
        target_margin=0.06,
        beta=0.98,
        tau_security=2.0,
        weights=weights,
        ratio_grid=ratio_grid,
        transport_share_mean=transport_share_mean,
        security_floor_ratio=security_floor_ratio,
        security_penalty_mild=security_penalty_mild,
        security_penalty_severe=security_penalty_severe,
        horizon=horizon,
        scenario=scenario,
    )


def consumer_loss(price: float, equilibrium_price: float, quantity: float, epsilon: float) -> float:
    price = max(price, 1.0)
    equilibrium_price = max(equilibrium_price, 1.0)
    if abs(price - equilibrium_price) < 1e-9:
        return 0.0
    a_scale = quantity / (price ** epsilon)
    # Integral of demand curve between regulated and equilibrium price.
    area = a_scale / (1 + epsilon) * abs(equilibrium_price ** (1 + epsilon) - price ** (1 + epsilon))
    rectangle = abs(equilibrium_price - price) * min(quantity, a_scale * equilibrium_price**epsilon)
    return max(area - rectangle, 0.0)


def normalized_single_period_loss(
    price_prev: float,
    action: float,
    action_prev: float,
    price_prev2: float,
    oil_price: float,
    inventory_cost: float,
    cpi_prev: float,
    quantity: float,
    calib: Calibration,
) -> tuple[float, dict[str, float], float]:
    price = max(price_prev + action, 1.0)
    # No-intervention theoretical equilibrium price P*: the retail price that
    # covers dynamic marginal cost and the target reasonable profit margin.
    equilibrium_price = inventory_cost / max(1.0 - calib.target_margin, 1e-6)
    margin = (price - inventory_cost) / price
    cpi_now = (
        calib.alpha0
        + calib.alpha1 * cpi_prev
        + calib.theta0_cpi * (action / max(price_prev, 1.0))
        + calib.theta1_cpi * (action_prev / max(price_prev2, 1.0))
    )

    # Use dimensionless normalized losses so one very large monetary component
    # does not dominate by unit choice.
    loss_cs_raw = consumer_loss(price, equilibrium_price, quantity, calib.epsilon_demand)
    loss_cs = loss_cs_raw / max(quantity * price_prev, 1.0)
    loss_profit = (margin - calib.target_margin) ** 2
    loss_cpi = (cpi_now - 0.02) ** 2
    # This equals the second difference of the domestic price path:
    # P_t - 2P_{t-1} + P_{t-2} = action_t - action_{t-1}.
    loss_exp = ((action - action_prev) / 500.0) ** 2

    target_cost_price = inventory_cost * (1.0 + calib.target_margin)
    if price >= target_cost_price:
        loss_sec = 0.0
    elif price >= inventory_cost:
        gap = (target_cost_price - price) / max(inventory_cost, 1.0)
        loss_sec = calib.security_penalty_mild * gap**2
    else:
        mild_gap = (target_cost_price - inventory_cost) / max(inventory_cost, 1.0)
        severe_gap = (inventory_cost - price) / max(inventory_cost, 1.0)
        loss_sec = calib.security_penalty_mild * mild_gap**2 + calib.security_penalty_severe * severe_gap**2

    parts = {
        "consumer": loss_cs,
        "profit": loss_profit,
        "cpi": loss_cpi,
        "expectation": loss_exp,
        "security": loss_sec,
    }
    total = sum(calib.weights[k] * parts[k] for k in parts)
    return float(total), parts, float(cpi_now)


def initialize_rolling_horizon_model(
    df: pd.DataFrame,
    fuel: Fuel,
    *,
    scenario: str = "baseline",
    raw_weights: dict[str, float] | None = None,
    security_floor_ratio: float = 0.97,
    epsilon_demand: float = -0.6,
    security_penalty_mild: float = 8.0,
    security_penalty_severe: float = 35.0,
    horizon: int = 3,
) -> DPResult:
    calib = calibrate(
        df,
        fuel,
        scenario=scenario,
        raw_weights=raw_weights,
        security_floor_ratio=security_floor_ratio,
        epsilon_demand=epsilon_demand,
        security_penalty_mild=security_penalty_mild,
        security_penalty_severe=security_penalty_severe,
        horizon=horizon,
    )
    # The policy is solved by finite rolling-horizon optimization over observed
    # pricing windows. These one-point arrays are retained only as compact
    # calibration containers for downstream output compatibility.
    oil_grid = np.array([float(df["oil_price_usd_bbl"].median())])
    price_grid = np.array([float(df["domestic_price"].median())])
    inv_grid = np.array([float(df["inventory_cost"].median())])
    cpi_grid = np.array([float(df["cpi_yoy"].median())])
    shape = (len(oil_grid), len(price_grid), len(inv_grid), len(cpi_grid))
    value = np.zeros(shape, dtype=float)
    policy = np.zeros(shape, dtype=float)

    return DPResult(
        fuel=fuel,
        oil_grid=oil_grid,
        price_grid=price_grid,
        inv_grid=inv_grid,
        cpi_grid=cpi_grid,
        value=value,
        policy=policy,
        calibration=calib,
    )


def policy_action(dp: DPResult, oil: float, price: float, inv: float, cpi: float) -> float:
    io = nearest_index(dp.oil_grid, oil)
    ip = nearest_index(dp.price_grid, price)
    ii = nearest_index(dp.inv_grid, inv)
    ic = nearest_index(dp.cpi_grid, cpi)
    return float(dp.policy[io, ip, ii, ic])


def action_from_ratio(mechanism_signal: float, ratio: float) -> tuple[float, float, float]:
    raw_action = ratio * mechanism_signal
    if abs(raw_action) < 50:
        return 0.0, mechanism_signal, raw_action
    return raw_action, mechanism_signal - raw_action, raw_action


def evaluate_action_sequence(
    df: pd.DataFrame,
    start_idx: int,
    ratios: tuple[float, ...],
    price: float,
    price_prev2: float,
    action_prev: float,
    cpi_prev: float,
    carry_over: float,
    calib: Calibration,
) -> tuple[float, dict[str, object]]:
    total_loss = 0.0
    state_price = price
    state_price_prev2 = price_prev2
    state_action_prev = action_prev
    state_cpi = cpi_prev
    state_carry = carry_over
    first_step: dict[str, object] | None = None

    for step, ratio in enumerate(ratios):
        row = df.iloc[start_idx + step]
        mechanism_signal = float(row["current_mechanism_delta"]) + state_carry
        action, next_carry, raw_action = action_from_ratio(mechanism_signal, ratio)
        loss, parts, cpi_next = normalized_single_period_loss(
            price_prev=state_price,
            action=action,
            action_prev=state_action_prev,
            price_prev2=state_price_prev2,
            oil_price=float(row["oil_price_usd_bbl"]),
            inventory_cost=float(row["inventory_cost"]),
            cpi_prev=state_cpi,
            quantity=float(row["quantity_tonnes"]),
            calib=calib,
        )
        loss += 0.20 * (next_carry / 500.0) ** 2
        total_loss += (calib.beta**step) * loss

        next_price = max(state_price + action, 1.0)
        if first_step is None:
            first_step = {
                "ratio": ratio,
                "raw_action": raw_action,
                "action": action,
                "carry": next_carry,
                "price": next_price,
                "loss": loss,
                "parts": parts,
                "cpi": cpi_next,
                "mechanism_signal": mechanism_signal,
            }

        state_price_prev2 = state_price
        state_price = next_price
        state_action_prev = action
        state_cpi = cpi_next
        state_carry = next_carry

    return total_loss, first_step or {}


def rolling_horizon_optimization(
    df: pd.DataFrame,
    start_idx: int,
    price: float,
    price_prev2: float,
    action_prev: float,
    cpi_prev: float,
    carry_over: float,
    calib: Calibration,
) -> dict[str, object]:
    horizon = min(calib.horizon, len(df) - start_idx)
    best_loss = math.inf
    best_step: dict[str, object] | None = None
    for ratios in itertools.product(calib.ratio_grid, repeat=horizon):
        total_loss, first_step = evaluate_action_sequence(
            df=df,
            start_idx=start_idx,
            ratios=ratios,
            price=price,
            price_prev2=price_prev2,
            action_prev=action_prev,
            cpi_prev=cpi_prev,
            carry_over=carry_over,
            calib=calib,
        )
        if total_loss < best_loss:
            best_loss = total_loss
            best_step = first_step
    if best_step is None:
        raise RuntimeError("rolling horizon optimization failed to find an action")
    best_step["discounted_horizon_loss"] = best_loss
    return best_step


def simulate_strategy(df: pd.DataFrame, dp: DPResult, fuel: Fuel, scenario: str = "baseline") -> pd.DataFrame:
    rows = []
    price_opt = float(df.loc[0, "domestic_price"])
    price_opt_prev2 = price_opt
    opt_action_prev = 0.0
    actual_action_prev = 0.0
    actual_price_prev2 = float(df.loc[0, "domestic_price"] - df.loc[0, "actual_delta"])
    mechanism_action_prev = 0.0
    mechanism_price_prev2 = actual_price_prev2
    cpi_prev = float(df.loc[0, "cpi_yoy"])
    carry_over = 0.0

    for idx, row in df.iterrows():
        current_prev_price = float(row["domestic_price"] - row["actual_delta"])
        cur_loss, cur_parts, cpi_cur = normalized_single_period_loss(
            price_prev=current_prev_price,
            action=float(row["actual_delta"]),
            action_prev=actual_action_prev,
            price_prev2=actual_price_prev2,
            oil_price=float(row["oil_price_usd_bbl"]),
            inventory_cost=float(row["inventory_cost"]),
            cpi_prev=cpi_prev,
            quantity=float(row["quantity_tonnes"]),
            calib=dp.calibration,
        )

        mechanism_delta = float(row["current_mechanism_delta"])
        mech_loss, mech_parts, cpi_mech = normalized_single_period_loss(
            price_prev=current_prev_price,
            action=mechanism_delta,
            action_prev=mechanism_action_prev,
            price_prev2=mechanism_price_prev2,
            oil_price=float(row["oil_price_usd_bbl"]),
            inventory_cost=float(row["inventory_cost"]),
            cpi_prev=cpi_prev,
            quantity=float(row["quantity_tonnes"]),
            calib=dp.calibration,
        )

        best = rolling_horizon_optimization(
            df=df,
            start_idx=int(idx),
            price=price_opt,
            price_prev2=price_opt_prev2,
            action_prev=opt_action_prev,
            cpi_prev=cpi_prev,
            carry_over=carry_over,
            calib=dp.calibration,
        )

        action = float(best["action"])
        opt_loss = float(best["loss"])
        opt_parts = best["parts"]
        cpi_opt = float(best["cpi"])
        chosen_ratio = float(best["ratio"])
        raw_optimal_delta = float(best["raw_action"])
        carry_over = float(best["carry"])
        mechanism_signal = float(best["mechanism_signal"])

        price_before_action = price_opt
        price_opt = max(price_opt + action, 1.0)
        price_opt_prev2 = price_before_action
        opt_action_prev = action
        actual_price_prev2 = current_prev_price
        actual_action_prev = float(row["actual_delta"])
        mechanism_price_prev2 = current_prev_price
        mechanism_action_prev = mechanism_delta
        cpi_prev = cpi_opt

        rows.append(
            {
                "adjust_date": row["adjust_date"],
                "fuel": fuel,
                "oil_price_usd_bbl": row["oil_price_usd_bbl"],
                "inventory_cost": row["inventory_cost"],
                "quantity_tonnes": row["quantity_tonnes"],
                "transport_share": row["transport_share"],
                "actual_delta": row["actual_delta"],
                "mechanism_delta": mechanism_delta,
                "mechanism_signal_with_carry": mechanism_signal,
                "optimal_ratio": chosen_ratio,
                "raw_optimal_delta": raw_optimal_delta,
                "optimal_delta": action,
                "carry_over_next": carry_over,
                "discounted_horizon_loss": float(best["discounted_horizon_loss"]),
                "scenario": scenario,
                "actual_price": row["domestic_price"],
                "optimal_price": price_opt,
                "actual_loss": cur_loss,
                "mechanism_loss": mech_loss,
                "optimal_loss": opt_loss,
                "actual_cpi_pred": cpi_cur,
                "mechanism_cpi_pred": cpi_mech,
                "optimal_cpi_pred": cpi_opt,
                **{f"actual_loss_{k}": v for k, v in cur_parts.items()},
                **{f"mechanism_loss_{k}": v for k, v in mech_parts.items()},
                **{f"optimal_loss_{k}": v for k, v in opt_parts.items()},
            }
        )
    return pd.DataFrame(rows)


def summarize(sim: pd.DataFrame) -> dict[str, float]:
    def mean_col(col: str) -> float:
        return float(sim[col].mean())

    mechanism_sign = np.sign(sim["mechanism_delta"].to_numpy(dtype=float))
    actual_sign = np.sign(sim["actual_delta"].to_numpy(dtype=float))
    optimal_sign = np.sign(sim["optimal_delta"].to_numpy(dtype=float))
    nonzero_mechanism = sim["mechanism_delta"].abs() > 1e-9
    intervention_ratio = np.divide(
        sim["optimal_delta"],
        sim["mechanism_delta"],
        out=np.full(len(sim), np.nan, dtype=float),
        where=nonzero_mechanism.to_numpy(),
    )

    summary = {
        "n_windows": float(len(sim)),
        "actual_total_loss_mean": mean_col("actual_loss"),
        "mechanism_total_loss_mean": mean_col("mechanism_loss"),
        "optimal_total_loss_mean": mean_col("optimal_loss"),
        "loss_reduction_vs_actual_pct": float(
            (sim["actual_loss"].mean() - sim["optimal_loss"].mean()) / max(sim["actual_loss"].mean(), 1e-12) * 100
        ),
        "loss_reduction_vs_mechanism_pct": float(
            (sim["mechanism_loss"].mean() - sim["optimal_loss"].mean()) / max(sim["mechanism_loss"].mean(), 1e-12) * 100
        ),
        "actual_delta_abs_mean": float(sim["actual_delta"].abs().mean()),
        "mechanism_delta_abs_mean": float(sim["mechanism_delta"].abs().mean()),
        "optimal_delta_abs_mean": float(sim["optimal_delta"].abs().mean()),
        "actual_delta_std": float(sim["actual_delta"].std()),
        "mechanism_delta_std": float(sim["mechanism_delta"].std()),
        "optimal_delta_std": float(sim["optimal_delta"].std()),
        "large_adjustment_count_abs_ge_500": float((sim["optimal_delta"].abs() >= 500).sum()),
        "price_below_inventory_cost_count": float((sim["optimal_price"] < sim["inventory_cost"]).sum()),
        "price_below_cost_plus_margin_count": float(
            (sim["optimal_price"] < sim["inventory_cost"] * (1.0 + 0.06)).sum()
        ),
        "carry_over_abs_mean": float(sim["carry_over_next"].abs().mean()),
        "direction_match_vs_mechanism_rate": float((optimal_sign == mechanism_sign).mean()),
        "direction_match_vs_actual_rate": float((optimal_sign == actual_sign).mean()),
        "intervention_ratio_mean": float(pd.Series(intervention_ratio).dropna().mean()),
    }
    for k in ["consumer", "profit", "cpi", "expectation", "security"]:
        summary[f"actual_{k}_loss_mean"] = mean_col(f"actual_loss_{k}")
        summary[f"mechanism_{k}_loss_mean"] = mean_col(f"mechanism_loss_{k}")
        summary[f"optimal_{k}_loss_mean"] = mean_col(f"optimal_loss_{k}")
    return summary


def write_report(all_summary: dict[str, dict[str, float]], all_calib: dict[str, Calibration]) -> None:
    lines = []
    lines.append("任务二 动态最优调价策略计算结果")
    lines.append("=" * 60)
    lines.append("")
    lines.append("模型口径：按照《任务二(1).docx》构建五维社会总福利损失函数，并将调价问题写成逐期动态决策问题。")
    lines.append("状态变量包括国际原油价格、上一期国内零售价、炼厂库存成本、上一期CPI；控制变量为本期实际调价幅度。")
    lines.append("数据修正：需求量使用汽油/柴油产量+进口-出口构造的表观消费量；消费者权重由交通通信支出占居民总消费支出比例校准。")
    lines.append("求解方法：由于连续高维状态空间下严格贝尔曼方程求解复杂，本文采用有限期滚动优化方法进行近似求解。")
    lines.append("每期以任务一得到的理论调幅为基准，在离散比例动作集中搜索未来若干期贴现社会福利损失最小的执行幅度，")
    lines.append("每次仅执行第一个动作，并在下一调价窗口重新滚动优化。")
    lines.append("")
    for key, summary in all_summary.items():
        fuel = str(summary.get("fuel", key))
        scenario = str(summary.get("scenario", "baseline"))
        name = "汽油" if fuel == "gasoline" else "柴油"
        calib = all_calib[key]
        lines.append(f"情景：{scenario}；品种：{name}")
        lines.append(f"{name}：")
        lines.append(f"  交通通信消费支出占居民总消费支出均值：{calib.transport_share_mean:.4f}")
        lines.append(f"  滚动优化预测期：H={calib.horizon}")
        lines.append(f"  归一化权重：{calib.weights}")
        lines.append(
            f"  AR(1)油价转移：ln P_c(t+1) = {calib.c0:.4f} + {calib.rho_c:.4f} ln P_c(t)，"
            f"sigma={calib.sigma_c:.4f}"
        )
        lines.append(
            f"  平均总损失：现行实际={summary['actual_total_loss_mean']:.6f}，"
            f"机制预测={summary['mechanism_total_loss_mean']:.6f}，"
            f"动态最优={summary['optimal_total_loss_mean']:.6f}"
        )
        lines.append(
            f"  最优策略相对实际调价平均损失下降 {summary['loss_reduction_vs_actual_pct']:.2f}%，"
            f"相对机制预测下降 {summary['loss_reduction_vs_mechanism_pct']:.2f}%"
        )
        lines.append(
            f"  平均绝对调价幅度：实际={summary['actual_delta_abs_mean']:.2f}，"
            f"机制预测={summary['mechanism_delta_abs_mean']:.2f}，"
            f"动态最优={summary['optimal_delta_abs_mean']:.2f} 元/吨"
        )
        lines.append(
            f"  调价幅度标准差：实际={summary['actual_delta_std']:.2f}，"
            f"机制预测={summary['mechanism_delta_std']:.2f}，"
            f"动态最优={summary['optimal_delta_std']:.2f}"
        )
        lines.append("  五类损失均值（实际 / 机制预测 / 动态最优）：")
        for key, label in [
            ("consumer", "消费者福利"),
            ("profit", "炼油企业利润"),
            ("cpi", "CPI通胀"),
            ("expectation", "预期平滑"),
            ("security", "能源安全"),
        ]:
            lines.append(
                f"    {label}：{summary[f'actual_{key}_loss_mean']:.6f} / "
                f"{summary[f'mechanism_{key}_loss_mean']:.6f} / "
                f"{summary[f'optimal_{key}_loss_mean']:.6f}"
            )
        lines.append("")
    lines.append("说明：需求量、WACC目标利润率、能源安全权重等难以从题目直接观测的变量，按文档给出的经验方式校准。")
    lines.append("报告未声称严格值迭代或全局最优；结果为有限期滚动优化下的近似动态调价策略。")
    (OUT_DIR / "task2_dynamic_pricing_report.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_summary: dict[str, dict[str, float]] = {}
    all_calib: dict[str, Calibration] = {}
    all_sim = []
    baseline_sim = []
    scenarios = ["baseline", "livelihood_priority", "stability_priority", "security_priority"]

    for scenario in scenarios:
      for fuel in ["gasoline", "diesel"]:
        df = make_model_data(fuel)  # type: ignore[arg-type]
        dp = initialize_rolling_horizon_model(df, fuel, scenario=scenario, horizon=3)  # type: ignore[arg-type]
        sim = simulate_strategy(df, dp, fuel, scenario=scenario)  # type: ignore[arg-type]
        sim.to_csv(OUT_DIR / f"task2_{scenario}_{fuel}_optimal_strategy.csv", index=False, encoding="utf-8-sig")
        if scenario == "baseline":
            sim.to_csv(OUT_DIR / f"task2_{fuel}_optimal_strategy.csv", index=False, encoding="utf-8-sig")
            baseline_sim.append(sim)
        all_sim.append(sim)
        summary = summarize(sim)
        summary["scenario"] = scenario
        summary["fuel"] = fuel
        key = f"{scenario}_{fuel}"
        all_summary[key] = summary
        all_calib[key] = dp.calibration

        calib_payload = {
            "scenario": dp.calibration.scenario,
            "c0": dp.calibration.c0,
            "rho_c": dp.calibration.rho_c,
            "sigma_c": dp.calibration.sigma_c,
            "alpha0": dp.calibration.alpha0,
            "alpha1": dp.calibration.alpha1,
            "theta0_cpi": dp.calibration.theta0_cpi,
            "theta1_cpi": dp.calibration.theta1_cpi,
            "epsilon_demand": dp.calibration.epsilon_demand,
            "target_margin": dp.calibration.target_margin,
            "beta": dp.calibration.beta,
            "tau_security": dp.calibration.tau_security,
            "weights": dp.calibration.weights,
            "ratio_grid": dp.calibration.ratio_grid,
            "transport_share_mean": dp.calibration.transport_share_mean,
            "security_floor_ratio": dp.calibration.security_floor_ratio,
            "security_penalty_mild": dp.calibration.security_penalty_mild,
            "security_penalty_severe": dp.calibration.security_penalty_severe,
            "horizon": dp.calibration.horizon,
        }
        (OUT_DIR / f"task2_{scenario}_{fuel}_calibration.json").write_text(
            json.dumps(calib_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if scenario == "baseline":
            (OUT_DIR / f"task2_{fuel}_calibration.json").write_text(
                json.dumps(calib_payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    summary_df = pd.DataFrame(
        [summary for summary in all_summary.values()]
    )
    summary_df.to_csv(
        OUT_DIR / "task2_strategy_comparison_summary_by_scenario.csv", index=False, encoding="utf-8-sig"
    )
    try:
        summary_df.query("scenario == 'baseline'").to_csv(
            OUT_DIR / "task2_strategy_comparison_summary.csv", index=False, encoding="utf-8-sig"
        )
    except PermissionError:
        summary_df.query("scenario == 'baseline'").to_csv(
            OUT_DIR / "task2_strategy_comparison_summary_latest.csv", index=False, encoding="utf-8-sig"
        )
    pd.concat(all_sim, ignore_index=True).to_csv(
        OUT_DIR / "task2_all_fuels_optimal_strategy_by_scenario.csv", index=False, encoding="utf-8-sig"
    )
    try:
        pd.concat(baseline_sim, ignore_index=True).to_csv(
            OUT_DIR / "task2_all_fuels_optimal_strategy.csv", index=False, encoding="utf-8-sig"
        )
    except PermissionError:
        pd.concat(baseline_sim, ignore_index=True).to_csv(
            OUT_DIR / "task2_all_fuels_optimal_strategy_latest.csv", index=False, encoding="utf-8-sig"
        )
    window_cols = [
        "adjust_date",
        "scenario",
        "fuel",
        "actual_delta",
        "mechanism_delta",
        "mechanism_signal_with_carry",
        "optimal_ratio",
        "raw_optimal_delta",
        "optimal_delta",
        "carry_over_next",
        "discounted_horizon_loss",
        "actual_loss",
        "mechanism_loss",
        "optimal_loss",
        "optimal_loss_consumer",
        "optimal_loss_profit",
        "optimal_loss_cpi",
        "optimal_loss_expectation",
        "optimal_loss_security",
        "optimal_price",
        "inventory_cost",
        "quantity_tonnes",
        "transport_share",
    ]
    pd.concat(all_sim, ignore_index=True)[window_cols].to_csv(
        OUT_DIR / "task2_window_loss_breakdown.csv", index=False, encoding="utf-8-sig"
    )
    write_report(all_summary, all_calib)

    print(summary_df.to_string(index=False))
    print(f"\n已写出目录: {OUT_DIR}")


if __name__ == "__main__":
    main()
