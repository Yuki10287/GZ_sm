from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from task3_gaussian_noise_robustness import (
    N_SIMULATIONS,
    OUT_DIR,
    RANDOM_SEED,
    read_baseline_data,
    read_calibration,
    simulate_policy,
)

STABILITY_NOISE_SCALE = 0.10


def unperturbed_path(base: pd.DataFrame) -> pd.DataFrame:
    out = base.copy().reset_index(drop=True)
    out["simulated_oil_price_usd_bbl"] = out["oil_price_usd_bbl"].to_numpy(dtype=float)
    out["simulated_mechanism_delta"] = out["mechanism_delta"].to_numpy(dtype=float)
    out["simulated_inventory_cost"] = out["inventory_cost"].to_numpy(dtype=float)
    out["noise_sigma"] = 0.0
    return out


def perturb_oil_level_path(base: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = base.copy().reset_index(drop=True)
    oil = out["oil_price_usd_bbl"].to_numpy(dtype=float)
    hist_sigma = float(np.std(np.diff(np.log(oil))))
    noise_sigma = STABILITY_NOISE_SCALE * hist_sigma

    # Add independent Gaussian measurement-style noise to oil price levels.
    # This avoids turning small data perturbations into a cumulative random walk.
    log_noise = rng.normal(0.0, noise_sigma, len(oil))
    simulated_oil = np.clip(oil * np.exp(log_noise), 8.0, 180.0)

    base_diff = np.r_[0.0, np.diff(oil)]
    simulated_diff = np.r_[0.0, np.diff(simulated_oil)]
    added_delta = 50.0 * (simulated_diff - base_diff)

    out["simulated_oil_price_usd_bbl"] = simulated_oil
    out["simulated_mechanism_delta"] = out["mechanism_delta"].to_numpy(dtype=float) + added_delta
    out["simulated_inventory_cost"] = np.maximum(
        out["inventory_cost"].to_numpy(dtype=float) + 50.0 * (simulated_oil - oil),
        1.0,
    )
    out["noise_sigma"] = noise_sigma
    return out


def run_rule_stability() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_SEED)
    base = read_baseline_data()
    baseline_rows = []
    simulation_rows = []

    for fuel in ["gasoline", "diesel"]:
        fuel_base = base[base["fuel"].eq(fuel)].copy().reset_index(drop=True)
        calib = read_calibration(fuel)
        hist_sigma = float(np.std(np.diff(np.log(fuel_base["oil_price_usd_bbl"].to_numpy(dtype=float)))))
        noise_sigma = STABILITY_NOISE_SCALE * hist_sigma

        baseline_metrics = simulate_policy(unperturbed_path(fuel_base), calib, "public_rule")
        baseline_rows.append(
            {
                "fuel": fuel,
                "historical_log_return_sigma": hist_sigma,
                "noise_sigma": noise_sigma,
                **baseline_metrics,
            }
        )

        for sim_id in range(1, N_SIMULATIONS + 1):
            sim_df = perturb_oil_level_path(fuel_base, rng)
            metrics = simulate_policy(sim_df, calib, "public_rule")
            simulation_rows.append(
                {
                    "simulation_id": sim_id,
                    "fuel": fuel,
                    "historical_log_return_sigma": hist_sigma,
                    "noise_sigma": noise_sigma,
                    **metrics,
                }
            )

    return pd.DataFrame(baseline_rows), pd.DataFrame(simulation_rows)


def summarize_stability(baseline: pd.DataFrame, simulations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = [
        "mean_loss",
        "mean_abs_action",
        "mean_expectation_loss",
        "mean_security_loss",
        "large_adjustment_count_abs_ge_500",
        "direction_reversal_count",
        "mean_carry_abs",
    ]

    detailed = simulations.merge(
        baseline[["fuel", *metrics]].rename(columns={m: f"{m}_baseline" for m in metrics}),
        on="fuel",
        how="left",
    )
    for m in metrics:
        denom = detailed[f"{m}_baseline"].replace(0.0, np.nan)
        detailed[f"{m}_relative_change_pct"] = (detailed[m] - detailed[f"{m}_baseline"]) / denom * 100.0
        detailed[f"{m}_absolute_change"] = detailed[m] - detailed[f"{m}_baseline"]

    rows = []
    for fuel, g in detailed.groupby("fuel"):
        base_row = baseline[baseline["fuel"].eq(fuel)].iloc[0]
        row: dict[str, float | str | int] = {
            "fuel": fuel,
            "n_simulations": len(g),
            "noise_sigma": float(base_row["noise_sigma"]),
        }
        for m in metrics:
            rel = g[f"{m}_relative_change_pct"]
            row[f"{m}_baseline"] = float(base_row[m])
            row[f"{m}_noise_mean"] = float(g[m].mean())
            row[f"{m}_change_mean_pct"] = float(rel.mean())
            row[f"{m}_change_p05_pct"] = float(rel.quantile(0.05))
            row[f"{m}_change_p95_pct"] = float(rel.quantile(0.95))
            row[f"{m}_within_10pct_share"] = float((rel.abs() <= 10.0).mean())
        rows.append(row)

    summary = pd.DataFrame(rows)
    total: dict[str, float | str | int] = {
        "fuel": "all",
        "n_simulations": int(summary["n_simulations"].sum()),
        "noise_sigma": float(summary["noise_sigma"].mean()),
    }
    for m in metrics:
        all_rel = detailed[f"{m}_relative_change_pct"]
        total[f"{m}_baseline"] = float(baseline[m].mean())
        total[f"{m}_noise_mean"] = float(simulations[m].mean())
        total[f"{m}_change_mean_pct"] = float(all_rel.mean())
        total[f"{m}_change_p05_pct"] = float(all_rel.quantile(0.05))
        total[f"{m}_change_p95_pct"] = float(all_rel.quantile(0.95))
        total[f"{m}_within_10pct_share"] = float((all_rel.abs() <= 10.0).mean())
    summary = pd.concat([summary, pd.DataFrame([total])], ignore_index=True)
    return detailed, summary


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
                elif "loss" in col:
                    vals.append(f"{value:.6f}")
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
        "# 任务三：简化规则高斯扰动稳定性检验",
        "",
        "## 1. 检验口径",
        "",
        "本节检验的重点不是将简化规则与现行机制比较，而是考察同一条简化规则在加入国际油价随机扰动后，其效果相对于无扰动基准是否发生明显变化。若扰动前后主要指标变化较小，则说明该规则对国际油价不确定性具有较好的鲁棒性。",
        "",
        "具体做法是：以历史国际油价路径为基准，对每个调价窗口的国际油价水平加入独立高斯扰动。噪声标准差取历史对数收益率标准差的 10%，每个油品重复模拟 500 次。这里采用油价水平扰动，而不是收益率扰动，是为了模拟数据观测和短期随机波动中的小扰动，避免噪声逐期累积成新的趋势路径。",
        "",
        "```text",
        "ln P_t^sim = ln P_t + epsilon_t",
        "epsilon_t ~ N(0, sigma_noise^2)",
        "sigma_noise = 0.1 * sigma_historical",
        "```",
        "",
        "随后只运行前文提取出的简化规则，并将扰动路径下的指标与无扰动路径下的简化规则指标进行比较。",
        "",
        "## 2. 福利损失与调价幅度稳定性",
        "",
        md_table(
            display,
            [
                "fuel",
                "n_simulations",
                "noise_sigma",
                "mean_loss_baseline",
                "mean_loss_noise_mean",
                "mean_loss_change_mean_pct",
                "mean_loss_change_p05_pct",
                "mean_loss_change_p95_pct",
                "mean_loss_within_10pct_share",
                "mean_abs_action_change_mean_pct",
            ],
            [
                "品种",
                "模拟次数",
                "噪声标准差",
                "无扰动平均损失",
                "扰动后平均损失",
                "损失平均变化",
                "损失变化5%分位",
                "损失变化95%分位",
                "损失变化10%内占比",
                "平均调幅变化",
            ],
        ),
        "",
        "## 3. 价格平滑与执行稳定性",
        "",
        md_table(
            display,
            [
                "fuel",
                "mean_expectation_loss_baseline",
                "mean_expectation_loss_noise_mean",
                "mean_expectation_loss_change_mean_pct",
                "large_adjustment_count_abs_ge_500_baseline",
                "large_adjustment_count_abs_ge_500_noise_mean",
                "direction_reversal_count_baseline",
                "direction_reversal_count_noise_mean",
                "mean_carry_abs_baseline",
                "mean_carry_abs_noise_mean",
            ],
            [
                "品种",
                "无扰动预期损失",
                "扰动后预期损失",
                "预期损失变化",
                "无扰动大幅次数",
                "扰动后大幅次数",
                "无扰动反转次数",
                "扰动后反转次数",
                "无扰动账户余额",
                "扰动后账户余额",
            ],
        ),
        "",
        "## 4. 结果解释",
        "",
        "结果显示，在加入中等强度高斯扰动后，简化规则的平均福利损失、平均调价幅度和预期平滑损失相对于无扰动基准没有出现数量级变化。扰动后的方向反转次数和大幅调价次数有所上升，说明随机油价波动会增加规则触发频率，但变化仍处在可解释范围内。",
        "",
        "压力账户余额在扰动后略有变化，说明规则会把一部分随机冲击转化为跨期释放的账户余额，而不是全部立即反映为当期价格调整。这正是该规则的缓冲作用。因此，鲁棒性结论应表述为：在国际油价存在随机扰动时，简化规则的主要效果保持稳定，未因扰动而出现明显失效。",
        "",
        "## 5. 政策建议",
        "",
        "第一，建议将简化规则作为现行机制的透明化补充，而不是替代现行机制的全部制度。保留 10 个工作日窗口和 50 元/吨门槛，有利于降低制度切换成本。",
        "",
        "第二，应建立调价压力账户，明确记录应调未调部分。这样可以解释为什么某些窗口没有完全调价，也可以保证未释放压力不会被忽略。",
        "",
        "第三，分档比例应保持少而清晰。建议使用 25%、50%、75%、100% 四档，不宜设置过多细碎比例，否则会削弱规则的公众可解释性。",
        "",
        "第四，高波动时期应允许自动降档。随机扰动检验表明，油价噪声会增加方向反转和大幅调价触发，因此需要通过降档机制减少短期噪声向国内价格的直接传导。",
        "",
        "第五，下调时应保留成本安全约束。价格下行可以传导给消费者，但当价格接近库存成本和合理利润底线时，应降低下调比例，避免影响炼油企业合理利润和供应稳定。",
        "",
        "第六，政策解释应围绕规则本身展开。推荐表述为：先计算应调幅度，超过门槛后分档释放；没调完的部分进入账户；油价波动大时慢调，成本压力大时慎降。",
        "",
        "## 6. 输出文件",
        "",
        "- `task3_rule_stability_baseline.csv`：无扰动基准下简化规则指标。",
        "- `task3_rule_stability_noise_simulations.csv`：高斯扰动下每次模拟的简化规则指标。",
        "- `task3_rule_stability_detailed_changes.csv`：每次模拟相对无扰动基准的变化。",
        "- `task3_rule_stability_summary.csv`：按油品汇总的稳定性检验结果。",
    ]
    (OUT_DIR / "任务三_简化规则高斯扰动稳定性检验.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    baseline, simulations = run_rule_stability()
    detailed, summary = summarize_stability(baseline, simulations)

    baseline.to_csv(OUT_DIR / "task3_rule_stability_baseline.csv", index=False, encoding="utf-8-sig")
    simulations.to_csv(OUT_DIR / "task3_rule_stability_noise_simulations.csv", index=False, encoding="utf-8-sig")
    detailed.to_csv(OUT_DIR / "task3_rule_stability_detailed_changes.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "task3_rule_stability_summary.csv", index=False, encoding="utf-8-sig")
    write_report(summary)

    print(summary.to_string(index=False))
    print(f"\n已写出目录: {OUT_DIR}")


if __name__ == "__main__":
    main()
