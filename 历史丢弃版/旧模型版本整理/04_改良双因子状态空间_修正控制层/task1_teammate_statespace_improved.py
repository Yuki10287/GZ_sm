from __future__ import annotations

"""改良版队友双因子状态空间验证脚本。

本版本保留队友的双状态油价模型（X_t 和 delta_t），但将效果较差的
调价控制层替换为更符合实际机制的控制层：

1. 用本窗口 10 日均价与上一调价窗口 10 日均价比较；
2. 使用 1 天计价滞后；
3. 将 50 元/吨门槛和 40/130 美元政策边界作为硬性阶跃规则处理。

本版本不加入新数据，目的是单独检验 Gemini 提到的“控制层纠偏”
是否有实际价值。
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

import task1_price_mechanism as base
import task1_teammate_statespace as strict


ROOT = Path(__file__).resolve().parent
STRICT_STATE_PATH = ROOT / "outputs" / "task1_teammate_statespace" / "state_space_oil_index.csv"
OUTPUT_DIR = ROOT / "outputs" / "task1_teammate_statespace_improved"


def load_or_fit_state_panel(config: base.MechanismConfig, refit: bool, maxiter: int) -> tuple[pd.DataFrame, dict[str, object]]:
    """读取之前的双因子状态估计；如指定参数，也可以重新估计。

    默认复用严格模型中耗时较长的状态估计结果，从而单独隔离
    “修正控制层”带来的影响。
    """

    if STRICT_STATE_PATH.exists() and not refit:
        state_panel = pd.read_csv(STRICT_STATE_PATH, parse_dates=["date"])
        fit_info = {
            "source": str(STRICT_STATE_PATH),
            "refit": False,
            "note": "Loaded the strict two-factor state estimate from the previous run.",
        }
        return state_panel, fit_info

    panel = base.read_oil_panel()
    oil_df = strict.prepare_log_observations(panel, config.start_date)
    state_panel, fit = strict.fit_teammate_state_space(oil_df, maxiter=maxiter)
    fit_info = {"source": "refit", "refit": True, "fit": asdict(fit)}
    return state_panel, fit_info


def run_improved_validation(config: base.MechanismConfig, refit: bool, maxiter: int) -> tuple[pd.DataFrame, dict[str, object]]:
    """用修正后的机制控制层验证双因子状态指数。

    修正控制层与优化基准模型保持一致：上一窗口均价基准、1 天滞后、
    以及硬性政策阈值。
    """

    state_panel, fit_info = load_or_fit_state_panel(config, refit=refit, maxiter=maxiter)
    domestic = base.read_domestic_adjustments(config.start_date)
    validation = base.simulate_mechanism(domestic, state_panel, "state_space_index_usd", config)

    metrics = {
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
    }
    summary = {"config": asdict(config), "state_fit": fit_info, "metrics": metrics}
    return validation, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Improved two-factor state-space mechanism validation.")
    parser.add_argument("--start-date", default=base.MechanismConfig.start_date)
    parser.add_argument("--train-end", default=base.MechanismConfig.train_end)
    parser.add_argument("--window-size", type=int, default=base.MechanismConfig.window_size)
    parser.add_argument("--pricing-lag-days", type=int, default=1)
    parser.add_argument("--maxiter", type=int, default=250)
    parser.add_argument("--refit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = base.MechanismConfig(
        start_date=args.start_date,
        train_end=args.train_end,
        window_size=args.window_size,
        pricing_lag_days=args.pricing_lag_days,
    )
    validation, summary = run_improved_validation(config, refit=args.refit, maxiter=args.maxiter)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    validation.to_csv(OUTPUT_DIR / "improved_state_space_validation.csv", index=False, encoding="utf-8-sig")
    with (OUTPUT_DIR / "summary_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Improved teammate two-factor state-space validation")
    print("Gasoline:", summary["metrics"]["gasoline"])
    print("Diesel:", summary["metrics"]["diesel"])
    print("Test after train gasoline:", summary["metrics"]["test_after_train"]["gasoline"])
    print("Test after train diesel:", summary["metrics"]["test_after_train"]["diesel"])
    print(f"\nOutputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
