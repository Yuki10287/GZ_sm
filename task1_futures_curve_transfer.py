from __future__ import annotations

"""WTI 期限结构迁移实验。

我们只有 WTI 的 M1-M4 期货期限结构，没有 Brent/Dubai 的同期限期货。
本脚本做一个假设性实验：把 WTI 的期限结构比例 M2/M1、M3/M1、M4/M1
迁移到 Brent 和 Dubai 现货上，构造 Brent/Dubai 的伪 M2-M4 曲线。

这个实验不应作为真实市场数据使用，只用于检查“如果 Brent/Dubai 也有
类似期限结构信息，双因子状态空间模型是否可能改善”。
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

import task1_futures_statespace as futures_model
import task1_price_mechanism as base


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs" / "task1_futures_curve_transfer"


def build_transferred_curve_panel(start_date: str) -> tuple[pd.DataFrame, list[str]]:
    """构造 WTI 实际期限结构 + Brent/Dubai 伪期限结构观测面板。"""

    obs_panel, _ = futures_model.build_observation_panel(start_date)
    obs_panel = obs_panel.copy()

    # 用 WTI 期限结构比例迁移到 Brent 和 Dubai。
    # 例如 brent_m3_proxy = brent_spot * (wti_m3 / wti_m1)。
    for benchmark in ["brent", "dubai"]:
        obs_panel[f"{benchmark}_m1_proxy"] = obs_panel[benchmark]
        for maturity in ["m2", "m3", "m4"]:
            ratio = obs_panel[f"wti_{maturity}"] / obs_panel["wti_m1"]
            obs_panel[f"{benchmark}_{maturity}_proxy"] = obs_panel[benchmark] * ratio

    price_cols = [
        "brent_m1_proxy",
        "brent_m2_proxy",
        "brent_m3_proxy",
        "brent_m4_proxy",
        "dubai_m1_proxy",
        "dubai_m2_proxy",
        "dubai_m3_proxy",
        "dubai_m4_proxy",
        "wti_m1",
        "wti_m2",
        "wti_m3",
        "wti_m4",
    ]
    for col in price_cols:
        values = pd.to_numeric(obs_panel[col], errors="coerce").mask(lambda s: s <= 0)
        obs_panel[f"log_{col}"] = np.log(values)

    obs_cols = [f"log_{col}" for col in price_cols]
    obs_panel = obs_panel.dropna(subset=obs_cols).reset_index(drop=True)
    return obs_panel, obs_cols


def run_validation(config: base.MechanismConfig, maxiter: int, penalty_weight: float) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """拟合迁移期限结构模型，并用修正机制层做验证。"""

    obs_panel, obs_cols = build_transferred_curve_panel(config.start_date)
    state_panel, fit = futures_model.fit_futures_state_space(
        obs_panel,
        obs_cols,
        maxiter=maxiter,
        penalty_weight=penalty_weight,
    )
    domestic = base.read_domestic_adjustments(config.start_date)
    validation = base.simulate_mechanism(domestic, state_panel, "state_space_futures_index_usd", config)
    max_date = state_panel["date"].max()
    validation = validation[pd.to_datetime(validation["pricing_end"]) <= max_date].reset_index(drop=True)

    summary = {
        "config": asdict(config),
        "curve_transfer_assumption": "Brent/Dubai pseudo futures are generated using WTI Mk/M1 ratios.",
        "state_space_fit": asdict(fit),
        "futures_data_end": str(max_date.date()),
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
    parser = argparse.ArgumentParser(description="Transfer WTI futures curve to Brent/Dubai and validate two-factor model.")
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
    state_panel.to_csv(OUTPUT_DIR / "curve_transfer_state_index.csv", index=False, encoding="utf-8-sig")
    validation.to_csv(OUTPUT_DIR / "curve_transfer_validation.csv", index=False, encoding="utf-8-sig")
    with (OUTPUT_DIR / "summary_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("WTI 期限结构迁移到 Brent/Dubai 的双因子验证")
    print("Fit:", summary["state_space_fit"])
    print("Futures data end:", summary["futures_data_end"])
    print("Gasoline:", summary["metrics"]["gasoline"])
    print("Diesel:", summary["metrics"]["diesel"])
    print("Test after train gasoline:", summary["metrics"]["test_after_train"]["gasoline"])
    print("Test after train diesel:", summary["metrics"]["test_after_train"]["diesel"])
    print(f"\nOutputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
