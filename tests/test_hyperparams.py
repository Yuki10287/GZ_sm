"""并行超参数测试脚本。

用法:
    python tests/test_hyperparams.py

功能:
    - 自定义多组超参数组合
    - 并行运行所有组合（使用多进程）
    - 结果保存到 tests/output/<参数标签>/
    - 最后汇总输出比较表
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path

import pandas as pd

# 将项目根目录加入 sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import task1_real_m136_gemini as model
import task1_price_mechanism as base

TEST_OUTPUT = Path(__file__).resolve().parent / "output"


@dataclass
class ParamConfig:
    """一组超参数配置。"""

    label: str
    window_size: int = 10
    pricing_lag_days: int = 1
    maxiter: int = 250
    penalty_weight: float = 0.02
    start_date: str = base.MechanismConfig.start_date
    train_end: str = base.MechanismConfig.train_end


# ============================================================
# 自定义超参数组合 —— 在这里添加你要测试的配置
# ============================================================
PARAM_GRID: list[ParamConfig] = [
    ParamConfig(label="baseline", window_size=10, pricing_lag_days=1, maxiter=250, penalty_weight=0.02),
    ParamConfig(label="w5_lag1", window_size=5, pricing_lag_days=1, maxiter=250, penalty_weight=0.02),
    ParamConfig(label="w15_lag1", window_size=15, pricing_lag_days=1, maxiter=250, penalty_weight=0.02),
    ParamConfig(label="w10_lag2", window_size=10, pricing_lag_days=2, maxiter=250, penalty_weight=0.02),
    ParamConfig(label="w10_lag0", window_size=10, pricing_lag_days=0, maxiter=250, penalty_weight=0.02),
    ParamConfig(label="w10_lag1_pen05", window_size=10, pricing_lag_days=1, maxiter=250, penalty_weight=0.05),
    ParamConfig(label="w10_lag1_pen01", window_size=10, pricing_lag_days=1, maxiter=250, penalty_weight=0.01),
    ParamConfig(label="w10_lag1_iter500", window_size=10, pricing_lag_days=1, maxiter=500, penalty_weight=0.02),
]


def run_single(cfg: ParamConfig) -> dict:
    """运行单组超参数，返回结果摘要。"""

    out_dir = TEST_OUTPUT / cfg.label
    out_dir.mkdir(parents=True, exist_ok=True)

    config = base.MechanismConfig(
        start_date=cfg.start_date,
        train_end=cfg.train_end,
        window_size=cfg.window_size,
        pricing_lag_days=cfg.pricing_lag_days,
    )

    print(f"[START] {cfg.label}")
    t0 = time.time()

    try:
        state_panel, validation, summary = model.run_validation(
            config, maxiter=cfg.maxiter, penalty_weight=cfg.penalty_weight
        )

        # 保存输出
        state_panel.to_csv(out_dir / "state_index.csv", index=False, encoding="utf-8-sig")
        validation.to_csv(out_dir / "validation.csv", index=False, encoding="utf-8-sig")
        with (out_dir / "summary_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        elapsed = time.time() - t0
        fit_info = summary.get("state_space_fit", {})
        metrics = summary.get("metrics", {})

        result = {
            "label": cfg.label,
            "window_size": cfg.window_size,
            "pricing_lag_days": cfg.pricing_lag_days,
            "maxiter": cfg.maxiter,
            "penalty_weight": cfg.penalty_weight,
            "neg_loglike": fit_info.get("neg_loglike"),
            "fit_success": fit_info.get("success"),
            "gasoline_mae": metrics.get("gasoline", {}).get("MAE"),
            "gasoline_mape": metrics.get("gasoline", {}).get("MAPE"),
            "diesel_mae": metrics.get("diesel", {}).get("MAE"),
            "diesel_mape": metrics.get("diesel", {}).get("MAPE"),
            "test_gas_mae": metrics.get("test_after_train", {}).get("gasoline", {}).get("MAE"),
            "test_gas_mape": metrics.get("test_after_train", {}).get("gasoline", {}).get("MAPE"),
            "test_die_mae": metrics.get("test_after_train", {}).get("diesel", {}).get("MAE"),
            "test_die_mape": metrics.get("test_after_train", {}).get("diesel", {}).get("MAPE"),
            "elapsed_sec": round(elapsed, 1),
            "status": "OK",
        }
        print(f"[DONE]  {cfg.label}  ({elapsed:.1f}s)")
        return result

    except Exception as e:
        elapsed = time.time() - t0
        print(f"[FAIL]  {cfg.label}  ({elapsed:.1f}s): {e}")
        return {
            "label": cfg.label,
            "window_size": cfg.window_size,
            "pricing_lag_days": cfg.pricing_lag_days,
            "maxiter": cfg.maxiter,
            "penalty_weight": cfg.penalty_weight,
            "status": f"FAIL: {e}",
            "elapsed_sec": round(elapsed, 1),
        }


def main() -> None:
    n_workers = min(cpu_count(), len(PARAM_GRID))
    print(f"共 {len(PARAM_GRID)} 组参数，使用 {n_workers} 个并行进程\n")

    with Pool(processes=n_workers) as pool:
        results = pool.map(run_single, PARAM_GRID)

    # 汇总表
    df = pd.DataFrame(results)
    summary_path = TEST_OUTPUT / "comparison.csv"
    df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("汇总比较表")
    print("=" * 80)
    cols = [
        "label", "window_size", "pricing_lag_days", "penalty_weight",
        "gasoline_mape", "diesel_mape", "test_gas_mape", "test_die_mape",
        "elapsed_sec", "status",
    ]
    print(df[[c for c in cols if c in df.columns]].to_string(index=False))
    print(f"\n详细结果已保存到: {TEST_OUTPUT}")


if __name__ == "__main__":
    main()
