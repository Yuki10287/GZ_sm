"""并行日期范围测试脚本。

用法:
    python tests/test_date_ranges.py

功能:
    - 测试不同训练集/测试集日期范围组合
    - 并行运行所有组合
    - 结果保存到 tests/date_output/<标签>/
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import task1_real_m136_gemini as model
import task1_price_mechanism as base

TEST_OUTPUT = Path(__file__).resolve().parent / "date_output"


@dataclass
class DateConfig:
    """日期范围配置。"""

    label: str
    start_date: str
    train_end: str
    window_size: int = 10
    pricing_lag_days: int = 1
    maxiter: int = 250
    penalty_weight: float = 0.01


# ============================================================
# 8 个日期范围方案
# ============================================================
DATE_GRID: list[DateConfig] = [
    # A系列：基准 2016 起
    DateConfig(label="A1_2016_2022_test2025", start_date="2016-01-01", train_end="2022-12-31"),
    DateConfig(label="A2_2016_2022_test2026", start_date="2016-01-01", train_end="2022-12-31"),

    # B系列：精简 2018 起
    DateConfig(label="B1_2018_2022_test2025", start_date="2018-01-01", train_end="2022-12-31"),
    DateConfig(label="B2_2018_2022_test2026", start_date="2018-01-01", train_end="2022-12-31"),

    # C系列：扩展训练到 2023
    DateConfig(label="C1_2016_2023_test2025", start_date="2016-01-01", train_end="2023-12-31"),
    DateConfig(label="C2_2016_2023_test2026", start_date="2016-01-01", train_end="2023-12-31"),

    # D系列：聚焦近期 2020 起
    DateConfig(label="D1_2020_2023_test2025", start_date="2020-01-01", train_end="2023-12-31"),
    DateConfig(label="D2_2020_2023_test2026", start_date="2020-01-01", train_end="2023-12-31"),
]


def run_single(cfg: DateConfig) -> dict:
    """运行单组日期配置，返回结果摘要。"""

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

        state_panel.to_csv(out_dir / "state_index.csv", index=False, encoding="utf-8-sig")
        validation.to_csv(out_dir / "validation.csv", index=False, encoding="utf-8-sig")
        with (out_dir / "summary_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        elapsed = time.time() - t0
        fit = summary.get("state_space_fit", {})
        metrics = summary.get("metrics", {})
        gas = metrics.get("gasoline", {})
        die = metrics.get("diesel", {})
        test_gas = metrics.get("test_after_train", {}).get("gasoline", {})
        test_die = metrics.get("test_after_train", {}).get("diesel", {})

        result = {
            "label": cfg.label,
            "start_date": cfg.start_date,
            "train_end": cfg.train_end,
            "train_years": cfg.train_end[:4],
            "neg_loglike": round(fit.get("neg_loglike", 0), 2),
            "fit_success": fit.get("success"),
            "iterations": fit.get("iterations"),
            "gas_mae": round(gas.get("mae", 0), 2),
            "gas_rmse": round(gas.get("rmse", 0), 2),
            "gas_dir_acc": round(gas.get("direction_accuracy", 0) * 100, 1),
            "gas_corr": round(gas.get("corr", 0), 4),
            "die_mae": round(die.get("mae", 0), 2),
            "die_rmse": round(die.get("rmse", 0), 2),
            "die_dir_acc": round(die.get("direction_accuracy", 0) * 100, 1),
            "die_corr": round(die.get("corr", 0), 4),
            "test_gas_mae": round(test_gas.get("mae", 0), 2),
            "test_gas_rmse": round(test_gas.get("rmse", 0), 2),
            "test_gas_dir_acc": round(test_gas.get("direction_accuracy", 0) * 100, 1),
            "test_gas_corr": round(test_gas.get("corr", 0), 4),
            "test_die_mae": round(test_die.get("mae", 0), 2),
            "test_die_rmse": round(test_die.get("rmse", 0), 2),
            "test_die_dir_acc": round(test_die.get("direction_accuracy", 0) * 100, 1),
            "test_die_corr": round(test_die.get("corr", 0), 4),
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
            "start_date": cfg.start_date,
            "train_end": cfg.train_end,
            "status": f"FAIL: {e}",
            "elapsed_sec": round(elapsed, 1),
        }


def main() -> None:
    n_workers = min(cpu_count(), len(DATE_GRID))
    print(f"共 {len(DATE_GRID)} 组日期方案，使用 {n_workers} 个并行进程\n")

    with Pool(processes=n_workers) as pool:
        results = pool.map(run_single, DATE_GRID)

    df = pd.DataFrame(results)
    summary_path = TEST_OUTPUT / "date_comparison.csv"
    df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 140)
    print("日期范围测试结果汇总")
    print("=" * 140)
    print(f"{'label':<28} {'start':>10} {'train_end':>10} {'gas_mae':>9} {'gas_dir%':>8} {'die_mae':>9} {'die_dir%':>8} {'t_gas_mae':>10} {'t_gas_dir%':>10} {'t_die_mae':>10} {'t_die_dir%':>10} {'time':>6}")
    print("-" * 140)
    for r in results:
        if r.get("status") == "OK":
            print(f"{r['label']:<28} {r['start_date']:>10} {r['train_end']:>10} {r['gas_mae']:>9.2f} {r['gas_dir_acc']:>7.1f}% {r['die_mae']:>9.2f} {r['die_dir_acc']:>7.1f}% {r['test_gas_mae']:>10.2f} {r['test_gas_dir_acc']:>9.1f}% {r['test_die_mae']:>10.2f} {r['test_die_dir_acc']:>9.1f}% {r['elapsed_sec']:>5.0f}s")
        else:
            print(f"{r['label']:<28} {r.get('start_date','?'):>10} {r.get('train_end','?'):>10} {'FAILED':>9} {'':>8} {'':>9} {'':>8} {'':>10} {'':>10} {'':>10} {'':>10} {r['elapsed_sec']:>5.0f}s")

    print("\n说明: dir%=方向准确率, t_=测试集, 时间单位:秒")
    print(f"\n详细结果已保存到: {TEST_OUTPUT}")


if __name__ == "__main__":
    main()
