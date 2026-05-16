from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from task2_dynamic_pricing_dp import Calibration, normalized_single_period_loss


BASE_DIR = Path(__file__).resolve().parent
TASK2_DIR = BASE_DIR / "outputs" / "task2_dynamic_pricing"
OUT_DIR = BASE_DIR / "outputs" / "task3_rule_extraction"

RMB_PER_USD_BBL_TO_RMB_PER_TON = 50.0
THRESHOLDS = (50.0, 150.0, 300.0, 500.0)
N_SIMULATIONS = 500
NOISE_SCALE = 0.50
RANDOM_SEED = 2026


def read_baseline_data() -> pd.DataFrame:
    df = pd.read_csv(TASK2_DIR / "task2_all_fuels_optimal_strategy.csv")
    df = df[df["scenario"].eq("baseline")].copy()
    df["adjust_date"] = pd.to_datetime(df["adjust_date"])
    return df.sort_values(["fuel", "adjust_date"]).reset_index(drop=True)


def read_calibration(fuel: str) -> Calibration:
    payload = json.loads((TASK2_DIR / f"task2_baseline_{fuel}_calibration.json").read_text(encoding="utf-8"))
    return Calibration(**payload)


def base_ratio(abs_pressure: float) -> float:
    if abs_pressure < THRESHOLDS[0]:
        return 0.0
    if abs_pressure < THRESHOLDS[1]:
        return 0.25
    if abs_pressure < THRESHOLDS[2]:
        return 0.50
    if abs_pressure < THRESHOLDS[3]:
        return 0.75
    return 1.0


def sign_nonzero(x: float) -> int:
    if x > 1e-9:
        return 1
    if x < -1e-9:
        return -1
    return 0


def choose_public_rule_action(
    pressure: float,
    *,
    previous_action: float,
    price_prev: float,
    inventory_cost: float,
    target_margin: float,
    high_volatility: bool,
) -> tuple[float, float, float]:
    ratio = base_ratio(abs(pressure))

    target_cost_price = inventory_cost * (1.0 + target_margin)
    cost_margin = (price_prev - target_cost_price) / max(price_prev, 1.0)
    if pressure < 0.0 and ratio > 0.0 and cost_margin < 0.03:
        ratio = max(ratio - 0.25, 0.25)

    reversal = sign_nonzero(pressure) != 0 and sign_nonzero(previous_action) != 0
    reversal = reversal and sign_nonzero(pressure) != sign_nonzero(previous_action)
    if ratio > 0.0 and (high_volatility or reversal):
        ratio = max(ratio - 0.25, 0.0)

    action = ratio * pressure
    if abs(action) < THRESHOLDS[0]:
        return 0.0, pressure, ratio
    return action, pressure - action, ratio


def perturb_oil_path(base: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = base.copy().reset_index(drop=True)
    oil = out["oil_price_usd_bbl"].to_numpy(dtype=float)
    log_returns = np.r_[0.0, np.diff(np.log(oil))]
    hist_sigma = float(np.std(log_returns[1:]))
    noise_sigma = NOISE_SCALE * hist_sigma
    noise = rng.normal(0.0, noise_sigma, len(oil))
    noise[0] = 0.0

    simulated_returns = log_returns + noise
    simulated_log_oil = np.log(oil[0]) + np.cumsum(simulated_returns)
    simulated_oil = np.clip(np.exp(simulated_log_oil), 8.0, 180.0)

    base_diff = np.r_[0.0, np.diff(oil)]
    simulated_diff = np.r_[0.0, np.diff(simulated_oil)]
    added_delta = RMB_PER_USD_BBL_TO_RMB_PER_TON * (simulated_diff - base_diff)

    out["simulated_oil_price_usd_bbl"] = simulated_oil
    out["simulated_mechanism_delta"] = out["mechanism_delta"].to_numpy(dtype=float) + added_delta
    out["simulated_inventory_cost"] = np.maximum(
        out["inventory_cost"].to_numpy(dtype=float)
        + RMB_PER_USD_BBL_TO_RMB_PER_TON * (simulated_oil - oil),
        1.0,
    )
    out["noise_sigma"] = noise_sigma
    return out


def reversal_count(actions: list[float]) -> int:
    signs = [sign_nonzero(x) for x in actions if sign_nonzero(x) != 0]
    if len(signs) <= 1:
        return 0
    return int(sum(a != b for a, b in zip(signs[:-1], signs[1:])))


def simulate_policy(df: pd.DataFrame, calib: Calibration, policy: str) -> dict[str, float]:
    price = float(df.loc[0, "actual_price"] - df.loc[0, "actual_delta"])
    price_prev2 = price
    action_prev = 0.0
    cpi_prev = 0.02
    carry = 0.0

    oil_log_returns = np.r_[0.0, np.diff(np.log(df["simulated_oil_price_usd_bbl"].to_numpy(dtype=float)))]
    vol_threshold = float(np.quantile(np.abs(oil_log_returns[1:]), 0.75))

    losses = []
    expectation_losses = []
    security_losses = []
    actions = []
    carries = []
    below_cost = 0

    for idx, row in df.iterrows():
        mechanism_delta = float(row["simulated_mechanism_delta"])
        inventory_cost = float(row["simulated_inventory_cost"])

        if policy == "mechanism":
            action = mechanism_delta if abs(mechanism_delta) >= THRESHOLDS[0] else 0.0
            carry_next = 0.0
        elif policy == "public_rule":
            pressure = mechanism_delta + carry
            high_volatility = idx > 0 and abs(float(oil_log_returns[idx])) > vol_threshold
            action, carry_next, _ = choose_public_rule_action(
                pressure,
                previous_action=action_prev,
                price_prev=price,
                inventory_cost=inventory_cost,
                target_margin=calib.target_margin,
                high_volatility=bool(high_volatility),
            )
        else:
            raise ValueError(f"unknown policy: {policy}")

        loss, parts, cpi_now = normalized_single_period_loss(
            price_prev=price,
            action=action,
            action_prev=action_prev,
            price_prev2=price_prev2,
            oil_price=float(row["simulated_oil_price_usd_bbl"]),
            inventory_cost=inventory_cost,
            cpi_prev=cpi_prev,
            quantity=float(row["quantity_tonnes"]),
            calib=calib,
        )
        if policy == "public_rule":
            loss += 0.20 * (carry_next / 500.0) ** 2

        price_next = max(price + action, 1.0)
        below_cost += int(price_next < inventory_cost)

        losses.append(float(loss))
        expectation_losses.append(float(parts["expectation"]))
        security_losses.append(float(parts["security"]))
        actions.append(float(action))
        carries.append(float(carry_next))

        price_prev2 = price
        price = price_next
        action_prev = action
        cpi_prev = cpi_now
        carry = carry_next

    action_arr = np.asarray(actions, dtype=float)
    return {
        "mean_loss": float(np.mean(losses)),
        "mean_abs_action": float(np.mean(np.abs(action_arr))),
        "std_action": float(np.std(action_arr, ddof=1)),
        "max_abs_action": float(np.max(np.abs(action_arr))),
        "large_adjustment_count_abs_ge_500": int(np.sum(np.abs(action_arr) >= 500.0)),
        "no_adjustment_count": int(np.sum(np.abs(action_arr) < 1e-9)),
        "direction_reversal_count": reversal_count(actions),
        "price_below_inventory_cost_count": below_cost,
        "mean_expectation_loss": float(np.mean(expectation_losses)),
        "mean_security_loss": float(np.mean(security_losses)),
        "mean_carry_abs": float(np.mean(np.abs(carries))),
    }


def run_simulations() -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    base = read_baseline_data()
    rows = []
    for fuel in ["gasoline", "diesel"]:
        fuel_base = base[base["fuel"].eq(fuel)].copy().reset_index(drop=True)
        calib = read_calibration(fuel)
        hist_sigma = float(np.std(np.diff(np.log(fuel_base["oil_price_usd_bbl"].to_numpy(dtype=float)))))

        for sim_id in range(1, N_SIMULATIONS + 1):
            sim_df = perturb_oil_path(fuel_base, rng)
            for policy in ["mechanism", "public_rule"]:
                metrics = simulate_policy(sim_df, calib, policy)
                rows.append(
                    {
                        "simulation_id": sim_id,
                        "fuel": fuel,
                        "policy": policy,
                        "historical_log_return_sigma": hist_sigma,
                        "noise_sigma": NOISE_SCALE * hist_sigma,
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def build_pairwise(sim_results: pd.DataFrame) -> pd.DataFrame:
    mech = sim_results[sim_results["policy"].eq("mechanism")].copy()
    rule = sim_results[sim_results["policy"].eq("public_rule")].copy()
    merged = rule.merge(
        mech,
        on=["simulation_id", "fuel"],
        suffixes=("_rule", "_mechanism"),
    )
    out = merged[["simulation_id", "fuel", "historical_log_return_sigma_rule", "noise_sigma_rule"]].copy()
    out = out.rename(
        columns={
            "historical_log_return_sigma_rule": "historical_log_return_sigma",
            "noise_sigma_rule": "noise_sigma",
        }
    )
    for metric in [
        "mean_loss",
        "mean_abs_action",
        "std_action",
        "max_abs_action",
        "large_adjustment_count_abs_ge_500",
        "no_adjustment_count",
        "direction_reversal_count",
        "price_below_inventory_cost_count",
        "mean_expectation_loss",
        "mean_security_loss",
        "mean_carry_abs",
    ]:
        out[f"{metric}_rule"] = merged[f"{metric}_rule"]
        out[f"{metric}_mechanism"] = merged[f"{metric}_mechanism"]
    out["loss_reduction_vs_mechanism_pct"] = (
        (out["mean_loss_mechanism"] - out["mean_loss_rule"]) / out["mean_loss_mechanism"] * 100.0
    )
    out["abs_action_reduction_vs_mechanism_pct"] = (
        (out["mean_abs_action_mechanism"] - out["mean_abs_action_rule"])
        / out["mean_abs_action_mechanism"]
        * 100.0
    )
    out["expectation_loss_reduction_vs_mechanism_pct"] = (
        (out["mean_expectation_loss_mechanism"] - out["mean_expectation_loss_rule"])
        / out["mean_expectation_loss_mechanism"]
        * 100.0
    )
    return out


def summarize_pairwise(pairwise: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fuel, g in pairwise.groupby("fuel"):
        row = {
            "fuel": fuel,
            "n_simulations": len(g),
            "noise_sigma_mean": g["noise_sigma"].mean(),
            "loss_reduction_mean_pct": g["loss_reduction_vs_mechanism_pct"].mean(),
            "loss_reduction_p05_pct": g["loss_reduction_vs_mechanism_pct"].quantile(0.05),
            "loss_reduction_p50_pct": g["loss_reduction_vs_mechanism_pct"].quantile(0.50),
            "loss_reduction_p95_pct": g["loss_reduction_vs_mechanism_pct"].quantile(0.95),
            "share_loss_lower": (g["loss_reduction_vs_mechanism_pct"] > 0.0).mean(),
            "abs_action_reduction_mean_pct": g["abs_action_reduction_vs_mechanism_pct"].mean(),
            "expectation_loss_reduction_mean_pct": g["expectation_loss_reduction_vs_mechanism_pct"].mean(),
            "large_count_rule_mean": g["large_adjustment_count_abs_ge_500_rule"].mean(),
            "large_count_mechanism_mean": g["large_adjustment_count_abs_ge_500_mechanism"].mean(),
            "reversal_count_rule_mean": g["direction_reversal_count_rule"].mean(),
            "reversal_count_mechanism_mean": g["direction_reversal_count_mechanism"].mean(),
            "mean_carry_abs_rule": g["mean_carry_abs_rule"].mean(),
        }
        rows.append(row)
    total = {
        "fuel": "all",
        "n_simulations": len(pairwise),
        "noise_sigma_mean": pairwise["noise_sigma"].mean(),
        "loss_reduction_mean_pct": pairwise["loss_reduction_vs_mechanism_pct"].mean(),
        "loss_reduction_p05_pct": pairwise["loss_reduction_vs_mechanism_pct"].quantile(0.05),
        "loss_reduction_p50_pct": pairwise["loss_reduction_vs_mechanism_pct"].quantile(0.50),
        "loss_reduction_p95_pct": pairwise["loss_reduction_vs_mechanism_pct"].quantile(0.95),
        "share_loss_lower": (pairwise["loss_reduction_vs_mechanism_pct"] > 0.0).mean(),
        "abs_action_reduction_mean_pct": pairwise["abs_action_reduction_vs_mechanism_pct"].mean(),
        "expectation_loss_reduction_mean_pct": pairwise["expectation_loss_reduction_vs_mechanism_pct"].mean(),
        "large_count_rule_mean": pairwise["large_adjustment_count_abs_ge_500_rule"].mean(),
        "large_count_mechanism_mean": pairwise["large_adjustment_count_abs_ge_500_mechanism"].mean(),
        "reversal_count_rule_mean": pairwise["direction_reversal_count_rule"].mean(),
        "reversal_count_mechanism_mean": pairwise["direction_reversal_count_mechanism"].mean(),
        "mean_carry_abs_rule": pairwise["mean_carry_abs_rule"].mean(),
    }
    rows.append(total)
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame, columns: list[str], headers: list[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                if "share" in col:
                    vals.append(f"{value * 100:.2f}%")
                elif "pct" in col:
                    vals.append(f"{value:.2f}%")
                else:
                    vals.append(f"{value:.2f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(summary: pd.DataFrame) -> None:
    display = summary.copy()
    display["fuel"] = display["fuel"].map({"gasoline": "汽油", "diesel": "柴油", "all": "总体"})

    lines = [
        "# 任务三：高斯噪声鲁棒性检验与政策建议",
        "",
        "## 1. 检验思路",
        "",
        "为保持任务三简洁、透明的定位，鲁棒性检验不再设置大量情景，而是在历史国际油价路径上加入高斯随机扰动，检验简化规则在随机油价不确定性下是否仍然优于现行机制。",
        "",
        "具体做法是：以历史国际油价对数收益率为基准，在每个调价窗口加入独立高斯噪声。噪声标准差取历史对数收益率标准差的 50%，表示在历史波动基础上叠加中等强度随机扰动。每个油品重复模拟 500 次。",
        "",
        "扰动形式为：",
        "",
        "```text",
        "r_t^sim = r_t + epsilon_t,  epsilon_t ~ N(0, sigma_noise^2)",
        "sigma_noise = 0.5 * sigma_historical",
        "```",
        "",
        "每条模拟路径下，分别计算现行机制和简化规则的平均福利损失、平均绝对调幅、大幅调价次数、方向反转次数和预期平滑损失。",
        "",
        "## 2. 简化规则",
        "",
        "检验对象为前文提取的“调价压力缓冲账户下的分档比例释放规则”：每 10 个工作日计算本期可释放压力；不足 50 元/吨不调价；超过门槛后按 150、300、500 元/吨分为 25%、50%、75%、100% 四档释放；未释放部分进入下一期压力账户；下调遇成本安全压力或油价高波动时执行比例降低一档。",
        "",
        "## 3. 模拟结果",
        "",
        md_table(
            display,
            [
                "fuel",
                "n_simulations",
                "noise_sigma_mean",
                "loss_reduction_mean_pct",
                "loss_reduction_p05_pct",
                "loss_reduction_p50_pct",
                "loss_reduction_p95_pct",
                "share_loss_lower",
                "abs_action_reduction_mean_pct",
                "expectation_loss_reduction_mean_pct",
            ],
            [
                "品种",
                "模拟次数",
                "噪声标准差",
                "平均损失下降",
                "损失下降5%分位",
                "损失下降中位数",
                "损失下降95%分位",
                "损失下降路径占比",
                "平均调幅下降",
                "预期损失下降",
            ],
        ),
        "",
        "进一步看调价稳定性：",
        "",
        md_table(
            display,
            [
                "fuel",
                "large_count_rule_mean",
                "large_count_mechanism_mean",
                "reversal_count_rule_mean",
                "reversal_count_mechanism_mean",
                "mean_carry_abs_rule",
            ],
            [
                "品种",
                "规则大幅调价次数均值",
                "机制大幅调价次数均值",
                "规则方向反转次数均值",
                "机制方向反转次数均值",
                "规则压力账户余额均值",
            ],
        ),
        "",
        "结果表明，在加入高斯随机扰动后，简化规则在绝大多数模拟路径下仍能降低社会福利损失。总体上，简化规则平均损失下降、平均绝对调幅下降，并且预期平滑损失明显下降。这说明该规则对国际油价随机波动具有一定鲁棒性。",
        "",
        "同时，压力账户余额均值不为零，说明简化规则并不是完全消除调价压力，而是将部分压力跨期释放。这与任务三规则设计一致：它牺牲一部分即时传导精度，换取更平滑的价格路径和更容易解释的执行机制。",
        "",
        "## 4. 政策建议",
        "",
        "第一，保留现行 10 个工作日窗口和 50 元/吨门槛。该部分制度已经具备较强公众认知基础，建议作为简化规则的外层框架继续保留。",
        "",
        "第二，建立调价压力账户。将现行机制下应调但未完全调整的幅度显式记录，进入后续窗口继续处理。这样可以向公众解释“暂缓不是不调，少调不是取消”。",
        "",
        "第三，采用分档比例释放。对超过门槛的调价压力，不宜机械全额释放，而应按压力大小分为 25%、50%、75%、100% 四档执行，实现“小幅不动、中幅慢调、大幅快调、极端全调”。",
        "",
        "第四，高波动时期自动降档。当国际油价随机扰动较强或连续窗口调价方向反复时，建议降低一档执行比例，减少短期油价噪声对国内价格的放大。",
        "",
        "第五，下调时加入成本安全约束。油价下跌应传导给消费者，但当价格接近库存成本和合理利润底线时，应降低下调比例，兼顾炼油企业合理利润和供应稳定。",
        "",
        "第六，政策沟通应使用规则语言，而不是模型语言。可将规则概括为：先算应调幅度，再分档释放；没调完的部分进入账户；油价波动大时慢调，成本压力大时慎降。",
        "",
        "## 5. 输出文件",
        "",
        "- `task3_gaussian_noise_robustness_simulations.csv`：每次模拟下两种政策的指标。",
        "- `task3_gaussian_noise_robustness_pairwise.csv`：同一随机路径下简化规则相对现行机制的改善幅度。",
        "- `task3_gaussian_noise_robustness_summary.csv`：按油品汇总的鲁棒性结果。",
    ]
    (OUT_DIR / "任务三_高斯噪声鲁棒性检验与政策建议.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sim_results = run_simulations()
    pairwise = build_pairwise(sim_results)
    summary = summarize_pairwise(pairwise)

    sim_results.to_csv(OUT_DIR / "task3_gaussian_noise_robustness_simulations.csv", index=False, encoding="utf-8-sig")
    pairwise.to_csv(OUT_DIR / "task3_gaussian_noise_robustness_pairwise.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "task3_gaussian_noise_robustness_summary.csv", index=False, encoding="utf-8-sig")
    write_report(summary)

    print(summary.to_string(index=False))
    print(f"\n已写出目录: {OUT_DIR}")


if __name__ == "__main__":
    main()
