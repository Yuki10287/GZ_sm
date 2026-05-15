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
    theta_cpi: float
    epsilon_demand: float
    target_margin: float
    beta: float
    tau_security: float
    weights: dict[str, float]
    ratio_grid: list[float]
    transport_share_mean: float
    security_floor_ratio: float


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


def fit_cpi_transition(df: pd.DataFrame, fuel: Fuel) -> tuple[float, float, float]:
    pi = df["cpi_yoy"].to_numpy(dtype=float)
    price = df["domestic_price"].to_numpy(dtype=float)
    delta = df["actual_delta"].to_numpy(dtype=float)
    oil_return = np.divide(delta[1:], price[:-1], out=np.zeros(len(delta) - 1), where=price[:-1] != 0)
    y = pi[1:]
    X = np.column_stack([np.ones(len(y)), pi[:-1], oil_return])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(beta[0]), float(beta[1]), float(beta[2])


def nearest_index(grid: np.ndarray, value: float) -> int:
    return int(np.abs(grid - value).argmin())


def interpolate_grid(values: pd.Series, n: int, pad: float = 0.05) -> np.ndarray:
    lo = float(values.quantile(0.02))
    hi = float(values.quantile(0.98))
    spread = hi - lo
    return np.linspace(lo - pad * spread, hi + pad * spread, n)


def calibrate(
    df: pd.DataFrame,
    fuel: Fuel,
    *,
    security_weight: float = 0.46,
    security_floor_ratio: float = 0.97,
    epsilon_demand: float = -0.6,
) -> Calibration:
    c0, rho_c, sigma_c = fit_ar1_log_oil(df)
    alpha0, alpha1, theta_cpi = fit_cpi_transition(df, fuel)
    ratio_grid = [0.0, 0.25, 0.5, 0.75, 1.0]

    # Weight calibration follows the document's idea: expenditure/industry shares
    # plus larger macro/security preference coefficients. Values are normalized.
    transport_share_mean = float(df["transport_share"].dropna().mean()) if "transport_share" in df else 0.085
    raw_weights = {
        "consumer": transport_share_mean,
        "profit": 0.055,
        "cpi": 0.22,
        "expectation": 0.18,
        "security": security_weight,
    }
    s = sum(raw_weights.values())
    weights = {k: v / s for k, v in raw_weights.items()}

    return Calibration(
        c0=c0,
        rho_c=rho_c,
        sigma_c=max(sigma_c, 1e-4),
        alpha0=alpha0,
        alpha1=alpha1,
        theta_cpi=theta_cpi,
        epsilon_demand=epsilon_demand,
        target_margin=0.06,
        beta=0.98,
        tau_security=2.0,
        weights=weights,
        ratio_grid=ratio_grid,
        transport_share_mean=transport_share_mean,
        security_floor_ratio=security_floor_ratio,
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
    cpi_now = calib.alpha0 + calib.alpha1 * cpi_prev + calib.theta_cpi * (action / max(price_prev, 1.0))

    # Use dimensionless normalized losses so one very large monetary component
    # does not dominate by unit choice.
    loss_cs_raw = consumer_loss(price, equilibrium_price, quantity, calib.epsilon_demand)
    loss_cs = loss_cs_raw / max(quantity * price_prev, 1.0)
    loss_profit = (margin - calib.target_margin) ** 2
    loss_cpi = (cpi_now - 0.02) ** 2
    loss_exp = ((action - action_prev) / 500.0) ** 2

    supply_ratio = max((price - inventory_cost) / max(inventory_cost, 1.0), -1.0)
    demand = quantity * (price / max(price_prev, 1.0)) ** calib.epsilon_demand
    supply = quantity * max(0.0, 1.0 + 2.0 * supply_ratio)
    gap_ratio = max(0.0, demand - supply) / max(quantity, 1.0)
    loss_sec = math.exp(calib.tau_security * gap_ratio) - 1.0
    security_floor = inventory_cost * calib.security_floor_ratio
    if price < security_floor:
        cost_gap = (security_floor - price) / max(inventory_cost, 1.0)
        loss_sec += 100.0 + 1000.0 * cost_gap**2

    parts = {
        "consumer": loss_cs,
        "profit": loss_profit,
        "cpi": loss_cpi,
        "expectation": loss_exp,
        "security": loss_sec,
    }
    total = sum(calib.weights[k] * parts[k] for k in parts)
    return float(total), parts, float(cpi_now)


def solve_value_iteration(
    df: pd.DataFrame,
    fuel: Fuel,
    *,
    security_weight: float = 0.46,
    security_floor_ratio: float = 0.97,
    epsilon_demand: float = -0.6,
) -> DPResult:
    calib = calibrate(
        df,
        fuel,
        security_weight=security_weight,
        security_floor_ratio=security_floor_ratio,
        epsilon_demand=epsilon_demand,
    )
    # The task-2 policy is now evaluated on each observed pricing window as a
    # ratio of the mechanism signal. We keep the DPResult container so the rest
    # of the script structure and calibration outputs stay unchanged.
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


def simulate_strategy(df: pd.DataFrame, dp: DPResult, fuel: Fuel) -> pd.DataFrame:
    rows = []
    price_opt = float(df.loc[0, "domestic_price"])
    opt_action_prev = 0.0
    actual_action_prev = 0.0
    mechanism_action_prev = 0.0
    cpi_prev = float(df.loc[0, "cpi_yoy"])
    carry_over = 0.0

    for _, row in df.iterrows():
        current_prev_price = float(row["domestic_price"] - row["actual_delta"])
        cur_loss, cur_parts, cpi_cur = normalized_single_period_loss(
            price_prev=current_prev_price,
            action=float(row["actual_delta"]),
            action_prev=actual_action_prev,
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
            oil_price=float(row["oil_price_usd_bbl"]),
            inventory_cost=float(row["inventory_cost"]),
            cpi_prev=cpi_prev,
            quantity=float(row["quantity_tonnes"]),
            calib=dp.calibration,
        )

        mechanism_signal = mechanism_delta + carry_over
        best = None
        for ratio in dp.calibration.ratio_grid:
            raw_action = ratio * mechanism_signal
            if abs(raw_action) < 50:
                candidate_action = 0.0
                candidate_carry = mechanism_signal
            else:
                candidate_action = raw_action
                candidate_carry = mechanism_signal - candidate_action

            candidate_price = price_opt + candidate_action
            feasible = candidate_price >= float(row["inventory_cost"]) * dp.calibration.security_floor_ratio
            if not feasible:
                continue
            opt_loss, opt_parts, cpi_opt = normalized_single_period_loss(
                price_prev=price_opt,
                action=candidate_action,
                action_prev=opt_action_prev,
                oil_price=float(row["oil_price_usd_bbl"]),
                inventory_cost=float(row["inventory_cost"]),
                cpi_prev=cpi_prev,
                quantity=float(row["quantity_tonnes"]),
                calib=dp.calibration,
            )
            opt_loss += 0.20 * (candidate_carry / 500.0) ** 2
            if best is None or opt_loss < best["loss"]:
                best = {
                    "ratio": ratio,
                    "raw_action": raw_action,
                    "action": candidate_action,
                    "carry": candidate_carry,
                    "price": candidate_price,
                    "loss": opt_loss,
                    "parts": opt_parts,
                    "cpi": cpi_opt,
                    "feasible": feasible,
                }

        if best is None:
            required_action = float(row["inventory_cost"]) * dp.calibration.security_floor_ratio - price_opt
            if 0 < abs(required_action) < 50:
                required_action = 50.0 if required_action > 0 else -50.0
            opt_loss, opt_parts, cpi_opt = normalized_single_period_loss(
                price_prev=price_opt,
                action=required_action,
                action_prev=opt_action_prev,
                oil_price=float(row["oil_price_usd_bbl"]),
                inventory_cost=float(row["inventory_cost"]),
                cpi_prev=cpi_prev,
                quantity=float(row["quantity_tonnes"]),
                calib=dp.calibration,
            )
            best = {
                "ratio": np.nan,
                "raw_action": required_action,
                "action": required_action,
                "carry": mechanism_signal - required_action,
                "price": price_opt + required_action,
                "loss": opt_loss,
                "parts": opt_parts,
                "cpi": cpi_opt,
                "feasible": True,
            }

        action = float(best["action"])
        opt_loss = float(best["loss"])
        opt_parts = best["parts"]
        cpi_opt = float(best["cpi"])
        chosen_ratio = float(best["ratio"])
        raw_optimal_delta = float(best["raw_action"])
        action_feasible = bool(best["feasible"])
        carry_over = float(best["carry"])

        price_opt = max(price_opt + action, 1.0)
        opt_action_prev = action
        actual_action_prev = float(row["actual_delta"])
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
                "supply_constraint_feasible": action_feasible,
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
    lines.append("求解方法：在每个历史调价窗口内，以任务一机制理论调幅为基准，搜索0%、25%、50%、75%、100%的比例执行策略，")
    lines.append("并加入50元/吨调价门槛、carry-over累计项和库存成本供应安全约束。")
    lines.append("")
    for fuel, summary in all_summary.items():
        name = "汽油" if fuel == "gasoline" else "柴油"
        calib = all_calib[fuel]
        lines.append(f"{name}：")
        lines.append(f"  交通通信消费支出占居民总消费支出均值：{calib.transport_share_mean:.4f}")
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
    lines.append("后续可进一步根据队友意见调整权重和约束强度。")
    (OUT_DIR / "task2_dynamic_pricing_report.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_summary: dict[str, dict[str, float]] = {}
    all_calib: dict[str, Calibration] = {}
    all_sim = []

    for fuel in ["gasoline", "diesel"]:
        df = make_model_data(fuel)  # type: ignore[arg-type]
        dp = solve_value_iteration(df, fuel)  # type: ignore[arg-type]
        sim = simulate_strategy(df, dp, fuel)  # type: ignore[arg-type]
        sim.to_csv(OUT_DIR / f"task2_{fuel}_optimal_strategy.csv", index=False, encoding="utf-8-sig")
        all_sim.append(sim)
        all_summary[fuel] = summarize(sim)
        all_calib[fuel] = dp.calibration

        calib_payload = {
            "c0": dp.calibration.c0,
            "rho_c": dp.calibration.rho_c,
            "sigma_c": dp.calibration.sigma_c,
            "alpha0": dp.calibration.alpha0,
            "alpha1": dp.calibration.alpha1,
            "theta_cpi": dp.calibration.theta_cpi,
            "epsilon_demand": dp.calibration.epsilon_demand,
            "target_margin": dp.calibration.target_margin,
            "beta": dp.calibration.beta,
            "tau_security": dp.calibration.tau_security,
            "weights": dp.calibration.weights,
            "ratio_grid": dp.calibration.ratio_grid,
            "transport_share_mean": dp.calibration.transport_share_mean,
            "security_floor_ratio": dp.calibration.security_floor_ratio,
        }
        (OUT_DIR / f"task2_{fuel}_calibration.json").write_text(
            json.dumps(calib_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    summary_df = pd.DataFrame(
        [{"fuel": fuel, **summary} for fuel, summary in all_summary.items()]
    )
    summary_df.to_csv(OUT_DIR / "task2_strategy_comparison_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(all_sim, ignore_index=True).to_csv(
        OUT_DIR / "task2_all_fuels_optimal_strategy.csv", index=False, encoding="utf-8-sig"
    )
    window_cols = [
        "adjust_date",
        "fuel",
        "actual_delta",
        "mechanism_delta",
        "mechanism_signal_with_carry",
        "optimal_ratio",
        "raw_optimal_delta",
        "optimal_delta",
        "carry_over_next",
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
        "supply_constraint_feasible",
    ]
    pd.concat(all_sim, ignore_index=True)[window_cols].to_csv(
        OUT_DIR / "task2_window_loss_breakdown.csv", index=False, encoding="utf-8-sig"
    )
    write_report(all_summary, all_calib)

    print(summary_df.to_string(index=False))
    print(f"\n已写出目录: {OUT_DIR}")


if __name__ == "__main__":
    main()
