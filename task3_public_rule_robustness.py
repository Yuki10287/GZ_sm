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


SCENARIOS = {
    "historical": "历史波动复现",
    "high_volatility": "高波动情景",
    "sustained_up": "单边上涨情景",
    "sustained_down": "单边下跌情景",
    "up_then_down": "先涨后跌反转情景",
    "down_then_up": "先跌后涨反转情景",
    "extreme_jump": "极端跳升冲击情景",
}


def read_baseline_data() -> pd.DataFrame:
    path = TASK2_DIR / "task2_all_fuels_optimal_strategy.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing task2 output: {path}")
    df = pd.read_csv(path)
    df = df[df["scenario"].eq("baseline")].copy()
    df["adjust_date"] = pd.to_datetime(df["adjust_date"])
    return df.sort_values(["fuel", "adjust_date"]).reset_index(drop=True)


def read_calibration(fuel: str) -> Calibration:
    path = TASK2_DIR / f"task2_baseline_{fuel}_calibration.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return Calibration(**payload)


def scenario_log_multiplier(n: int, scenario: str) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)
    if scenario == "historical":
        return np.zeros(n)
    if scenario == "high_volatility":
        return 0.12 * np.sin(2.0 * np.pi * np.arange(n) / 6.0)
    if scenario == "sustained_up":
        return 0.35 * t
    if scenario == "sustained_down":
        return -0.30 * t
    if scenario == "up_then_down":
        return 0.28 * (1.0 - np.abs(2.0 * t - 1.0))
    if scenario == "down_then_up":
        return -0.25 * (1.0 - np.abs(2.0 * t - 1.0))
    if scenario == "extreme_jump":
        m = np.zeros(n)
        jump_start = int(n * 0.45)
        jump_end = int(n * 0.65)
        m[jump_start:jump_end] = 0.45
        if jump_end < n:
            m[jump_end:] = np.linspace(0.45, 0.12, n - jump_end)
        return m
    raise ValueError(f"unknown scenario: {scenario}")


def build_scenario_data(base: pd.DataFrame, scenario: str) -> pd.DataFrame:
    out = base.copy().reset_index(drop=True)
    oil = out["oil_price_usd_bbl"].to_numpy(dtype=float)
    multiplier = scenario_log_multiplier(len(out), scenario)
    shocked_oil = oil * np.exp(multiplier)

    base_diff = np.r_[0.0, np.diff(oil)]
    shocked_diff = np.r_[0.0, np.diff(shocked_oil)]
    added_delta = RMB_PER_USD_BBL_TO_RMB_PER_TON * (shocked_diff - base_diff)

    out["robustness_scenario"] = scenario
    out["scenario_label"] = SCENARIOS[scenario]
    out["shock_log_multiplier"] = multiplier
    out["scenario_oil_price_usd_bbl"] = shocked_oil
    out["scenario_mechanism_delta"] = out["mechanism_delta"].to_numpy(dtype=float) + added_delta
    out["scenario_inventory_cost"] = np.maximum(
        out["inventory_cost"].to_numpy(dtype=float)
        + RMB_PER_USD_BBL_TO_RMB_PER_TON * (shocked_oil - oil),
        1.0,
    )
    return out


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


def signed_nonzero(x: float) -> int:
    if x > 1e-9:
        return 1
    if x < -1e-9:
        return -1
    return 0


def choose_rule_action(
    *,
    pressure: float,
    price_prev: float,
    inventory_cost: float,
    target_margin: float,
    high_volatility: bool,
    previous_action: float,
) -> tuple[float, float, float, str]:
    ratio = base_ratio(abs(pressure))
    reasons: list[str] = []

    target_cost_price = inventory_cost * (1.0 + target_margin)
    cost_margin = (price_prev - target_cost_price) / max(price_prev, 1.0)
    if pressure < 0.0 and ratio > 0.0 and cost_margin < 0.03:
        ratio = max(ratio - 0.25, 0.25)
        reasons.append("cost_safety_downshift")

    if signed_nonzero(pressure) != 0 and signed_nonzero(previous_action) != 0:
        direction_reversal = signed_nonzero(pressure) != signed_nonzero(previous_action)
    else:
        direction_reversal = False

    if ratio > 0.0 and (high_volatility or direction_reversal):
        ratio = max(ratio - 0.25, 0.0)
        if high_volatility:
            reasons.append("volatility_downshift")
        if direction_reversal:
            reasons.append("reversal_downshift")

    raw_action = ratio * pressure
    if abs(raw_action) < THRESHOLDS[0]:
        return 0.0, pressure, ratio, ",".join(reasons + ["below_threshold"])

    carry = pressure - raw_action
    return raw_action, carry, ratio, ",".join(reasons)


def simulate_policy(
    df: pd.DataFrame,
    calib: Calibration,
    *,
    fuel: str,
    scenario: str,
    policy: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    price = float(df.loc[0, "actual_price"] - df.loc[0, "actual_delta"])
    price_prev2 = price
    action_prev = 0.0
    cpi_prev = 0.02
    carry = 0.0

    log_returns = np.r_[0.0, np.diff(np.log(df["scenario_oil_price_usd_bbl"].to_numpy(dtype=float)))]
    vol_threshold = np.nanquantile(np.abs(log_returns), 0.75)

    for idx, row in df.iterrows():
        mechanism_delta = float(row["scenario_mechanism_delta"])
        inventory_cost = float(row["scenario_inventory_cost"])
        quantity = float(row["quantity_tonnes"])
        oil_price = float(row["scenario_oil_price_usd_bbl"])

        if policy == "mechanism":
            action = mechanism_delta if abs(mechanism_delta) >= THRESHOLDS[0] else 0.0
            pressure = mechanism_delta
            ratio = 1.0 if action != 0.0 else 0.0
            next_carry = 0.0
            rule_reason = ""
        elif policy == "public_rule":
            pressure = mechanism_delta + carry
            high_volatility = abs(log_returns[idx]) > vol_threshold and idx > 0
            action, next_carry, ratio, rule_reason = choose_rule_action(
                pressure=pressure,
                price_prev=price,
                inventory_cost=inventory_cost,
                target_margin=calib.target_margin,
                high_volatility=bool(high_volatility),
                previous_action=action_prev,
            )
        else:
            raise ValueError(f"unknown policy: {policy}")

        loss, parts, cpi_now = normalized_single_period_loss(
            price_prev=price,
            action=action,
            action_prev=action_prev,
            price_prev2=price_prev2,
            oil_price=oil_price,
            inventory_cost=inventory_cost,
            cpi_prev=cpi_prev,
            quantity=quantity,
            calib=calib,
        )
        if policy == "public_rule":
            loss = loss + 0.20 * (next_carry / 500.0) ** 2

        new_price = max(price + action, 1.0)
        rows.append(
            {
                "adjust_date": row["adjust_date"],
                "fuel": fuel,
                "robustness_scenario": scenario,
                "scenario_label": SCENARIOS[scenario],
                "policy": policy,
                "oil_price_usd_bbl": oil_price,
                "inventory_cost": inventory_cost,
                "mechanism_delta": mechanism_delta,
                "pressure_before_action": pressure,
                "execution_ratio": ratio,
                "action": action,
                "carry_over_next": next_carry,
                "price_before": price,
                "price_after": new_price,
                "loss": loss,
                "loss_consumer": parts["consumer"],
                "loss_profit": parts["profit"],
                "loss_cpi": parts["cpi"],
                "loss_expectation": parts["expectation"],
                "loss_security": parts["security"],
                "cpi_pred": cpi_now,
                "rule_reason": rule_reason,
            }
        )

        price_prev2 = price
        price = new_price
        action_prev = action
        cpi_prev = cpi_now
        carry = next_carry

    return pd.DataFrame(rows)


def reversal_count(actions: pd.Series) -> int:
    signs = np.sign(actions.to_numpy(dtype=float))
    signs = signs[signs != 0]
    if len(signs) <= 1:
        return 0
    return int(np.sum(signs[1:] != signs[:-1]))


def summarize(window: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, g in window.groupby(["robustness_scenario", "scenario_label", "fuel", "policy"]):
        scenario, label, fuel, policy = keys
        rows.append(
            {
                "robustness_scenario": scenario,
                "scenario_label": label,
                "fuel": fuel,
                "policy": policy,
                "n_windows": len(g),
                "mean_loss": g["loss"].mean(),
                "mean_abs_action": g["action"].abs().mean(),
                "std_action": g["action"].std(),
                "max_abs_action": g["action"].abs().max(),
                "large_adjustment_count_abs_ge_500": int((g["action"].abs() >= 500.0).sum()),
                "no_adjustment_count": int((g["action"].abs() < 1e-9).sum()),
                "direction_reversal_count": reversal_count(g["action"]),
                "price_below_inventory_cost_count": int((g["price_after"] < g["inventory_cost"]).sum()),
                "mean_expectation_loss": g["loss_expectation"].mean(),
                "mean_security_loss": g["loss_security"].mean(),
                "mean_carry_abs": g["carry_over_next"].abs().mean(),
            }
        )
    summary = pd.DataFrame(rows)

    mech = summary[summary["policy"].eq("mechanism")][
        ["robustness_scenario", "fuel", "mean_loss", "mean_abs_action", "large_adjustment_count_abs_ge_500", "direction_reversal_count"]
    ].rename(
        columns={
            "mean_loss": "mechanism_mean_loss",
            "mean_abs_action": "mechanism_mean_abs_action",
            "large_adjustment_count_abs_ge_500": "mechanism_large_count",
            "direction_reversal_count": "mechanism_reversal_count",
        }
    )
    summary = summary.merge(mech, on=["robustness_scenario", "fuel"], how="left")
    summary["loss_reduction_vs_mechanism_pct"] = (
        (summary["mechanism_mean_loss"] - summary["mean_loss"])
        / summary["mechanism_mean_loss"].replace(0.0, np.nan)
        * 100.0
    )
    summary["abs_action_reduction_vs_mechanism_pct"] = (
        (summary["mechanism_mean_abs_action"] - summary["mean_abs_action"])
        / summary["mechanism_mean_abs_action"].replace(0.0, np.nan)
        * 100.0
    )
    return summary


def markdown_table(df: pd.DataFrame, columns: list[str], headers: list[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                if "pct" in col:
                    vals.append(f"{value:.2f}%")
                elif "loss" in col:
                    vals.append(f"{value:.6f}")
                else:
                    vals.append(f"{value:.2f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(summary: pd.DataFrame, scenario_meta: pd.DataFrame) -> None:
    rule = summary[summary["policy"].eq("public_rule")].copy()
    display = rule[
        [
            "scenario_label",
            "fuel",
            "mean_loss",
            "loss_reduction_vs_mechanism_pct",
            "mean_abs_action",
            "abs_action_reduction_vs_mechanism_pct",
            "large_adjustment_count_abs_ge_500",
            "direction_reversal_count",
        ]
    ].copy()
    display["fuel"] = display["fuel"].map({"gasoline": "汽油", "diesel": "柴油"})

    avg = (
        rule.groupby("scenario_label", as_index=False)
        .agg(
            mean_loss_reduction_pct=("loss_reduction_vs_mechanism_pct", "mean"),
            mean_abs_action_reduction_pct=("abs_action_reduction_vs_mechanism_pct", "mean"),
            total_large_count=("large_adjustment_count_abs_ge_500", "sum"),
            total_reversal_count=("direction_reversal_count", "sum"),
        )
        .sort_values("scenario_label")
    )

    lines = [
        "# 任务三：简化规则鲁棒性检验与政策建议",
        "",
        "## 1. 检验目的",
        "",
        "任务三后半部分检验的对象不是任务二滚动优化策略，而是前文提取出的简化规则：调价压力缓冲账户下的分档比例释放规则。检验思路是，在历史油价路径基础上构造多种国际油价不确定性情景，比较现行机制和简化规则的表现。",
        "",
        "这里的鲁棒性不是要求简化规则在每一个单项指标上都优于现行机制，而是考察它在油价高波动、单边冲击和反转冲击下，是否仍能保持较低福利损失、较小平均调幅和较少大幅调价。",
        "",
        "## 2. 情景设置",
        "",
        markdown_table(
            scenario_meta,
            ["robustness_scenario", "scenario_label", "description"],
            ["情景代码", "情景名称", "含义"],
        ),
        "",
        "冲击处理方式为：先对历史国际油价路径施加情景扰动，再将油价路径变化折算为理论调幅和库存成本的变化。折算系数取 1 美元/桶约对应 50 元/吨，用于反映国际原油价格变化向成品油吨价的传导量级。",
        "",
        "## 3. 简化规则设定",
        "",
        "鲁棒性检验中采用的执行规则为：每 10 个工作日计算一次本期可释放压力；低于 50 元/吨不调价；超过门槛后按 150、300、500 元/吨划分为 25%、50%、75%、100% 四档释放；未释放部分进入下一期压力账户。若本期为下调且价格接近库存成本和合理利润底线，则下调比例降低一档；若国际油价处于高波动或调价方向频繁反转，则执行比例降低一档。",
        "",
        "该规则仍然是政策规则，不是重新优化模型。阈值采用整数档，是为了保证公众解释和部门执行的简洁性。",
        "",
        "## 4. 鲁棒性检验结果",
        "",
        markdown_table(
            display,
            [
                "scenario_label",
                "fuel",
                "mean_loss",
                "loss_reduction_vs_mechanism_pct",
                "mean_abs_action",
                "abs_action_reduction_vs_mechanism_pct",
                "large_adjustment_count_abs_ge_500",
                "direction_reversal_count",
            ],
            ["情景", "品种", "规则平均损失", "较现行机制损失下降", "规则平均绝对调幅", "平均调幅下降", "大幅调价次数", "方向反转次数"],
        ),
        "",
        "按情景汇总，两类油品平均表现如下：",
        "",
        markdown_table(
            avg,
            [
                "scenario_label",
                "mean_loss_reduction_pct",
                "mean_abs_action_reduction_pct",
                "total_large_count",
                "total_reversal_count",
            ],
            ["情景", "平均损失下降", "平均调幅下降", "两油品大幅调价次数合计", "两油品方向反转次数合计"],
        ),
        "",
        "从结果看，简化规则在多数情景下能够降低平均绝对调幅，并减少一次性大幅调价。其主要作用不是完全消除调价，而是把一次性冲击拆分为多期释放，并通过压力账户保证未释放部分继续被处理。这与任务二中价格预期平滑损失显著下降的结论一致。",
        "",
        "需要注意的是，简化规则在极端跳升或持续单边冲击下可能保留较多压力账户余额，因此部分福利损失不会单调下降。这说明简化规则不是任务二最优策略本身，而是复杂策略的可执行近似。它以透明性和稳定性换取一部分精细度，适合用于政策机制设计。",
        "",
        "## 5. 政策建议",
        "",
        "第一，保留现行机制的基本框架，但增加压力账户。现行 10 个工作日调价窗口和 50 元/吨门槛具有较强公众认知基础，不宜完全推翻。建议在此基础上建立调价压力账户，将应调未调部分显式记录并跨期处理。",
        "",
        "第二，将一次性调价改为分档比例释放。对超过门槛的调价压力，不宜简单全额释放，而应根据压力大小采用 25%、50%、75%、100% 的分档执行。这样可以减少短期价格跳变，又能避免长期偏离国际成本。",
        "",
        "第三，建立高波动时期的自动降档机制。当国际油价短期剧烈波动或调价方向连续反转时，建议自动降低一档执行比例，避免国内价格对短期外部扰动过度反应。",
        "",
        "第四，对下调设置成本安全约束。油价下跌应向消费者传导，但当价格接近库存成本和合理利润底线时，应降低下调比例，防止炼油企业利润和供应安全受到过度冲击。",
        "",
        "第五，加强政策解释。对公众解释时，不宜强调复杂优化模型，而应使用“该调多少先记账、压力小先缓一缓、压力大逐步释放、没调完下期继续处理”的表述。这样能降低公众对调价不及时或调价不完全的误解。",
        "",
        "## 6. 输出文件",
        "",
        "- `task3_public_rule_robustness_scenarios.csv`：鲁棒性情景说明。",
        "- `task3_public_rule_robustness_window_results.csv`：逐窗口模拟结果。",
        "- `task3_public_rule_robustness_summary.csv`：按情景、油品和政策汇总结果。",
    ]
    (OUT_DIR / "任务三_鲁棒性检验与政策建议.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = read_baseline_data()
    all_windows = []

    for fuel in ["gasoline", "diesel"]:
        fuel_base = base[base["fuel"].eq(fuel)].copy().reset_index(drop=True)
        calib = read_calibration(fuel)
        for scenario in SCENARIOS:
            scen_df = build_scenario_data(fuel_base, scenario)
            for policy in ["mechanism", "public_rule"]:
                all_windows.append(
                    simulate_policy(
                        scen_df,
                        calib,
                        fuel=fuel,
                        scenario=scenario,
                        policy=policy,
                    )
                )

    window = pd.concat(all_windows, ignore_index=True)
    summary = summarize(window)

    scenario_meta = pd.DataFrame(
        [
            {
                "robustness_scenario": "historical",
                "scenario_label": SCENARIOS["historical"],
                "description": "不额外施加冲击，复现历史国际油价波动。",
            },
            {
                "robustness_scenario": "high_volatility",
                "scenario_label": SCENARIOS["high_volatility"],
                "description": "在历史路径上加入周期性高波动扰动，检验规则对短期剧烈波动的适应性。",
            },
            {
                "robustness_scenario": "sustained_up",
                "scenario_label": SCENARIOS["sustained_up"],
                "description": "国际油价持续上行，检验规则是否避免上调压力长期累积。",
            },
            {
                "robustness_scenario": "sustained_down",
                "scenario_label": SCENARIOS["sustained_down"],
                "description": "国际油价持续下行，检验规则是否在让利消费者和保障供应之间折中。",
            },
            {
                "robustness_scenario": "up_then_down",
                "scenario_label": SCENARIOS["up_then_down"],
                "description": "国际油价先上涨后回落，检验方向反转时的平滑能力。",
            },
            {
                "robustness_scenario": "down_then_up",
                "scenario_label": SCENARIOS["down_then_up"],
                "description": "国际油价先下跌后回升，检验下调后反转时期的稳定性。",
            },
            {
                "robustness_scenario": "extreme_jump",
                "scenario_label": SCENARIOS["extreme_jump"],
                "description": "中段出现极端跳升冲击，随后逐步回落，检验极端风险下的缓冲能力。",
            },
        ]
    )

    scenario_meta.to_csv(OUT_DIR / "task3_public_rule_robustness_scenarios.csv", index=False, encoding="utf-8-sig")
    window.to_csv(OUT_DIR / "task3_public_rule_robustness_window_results.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "task3_public_rule_robustness_summary.csv", index=False, encoding="utf-8-sig")
    write_report(summary, scenario_meta)

    print(summary.to_string(index=False))
    print(f"\n已写出目录: {OUT_DIR}")


if __name__ == "__main__":
    main()
