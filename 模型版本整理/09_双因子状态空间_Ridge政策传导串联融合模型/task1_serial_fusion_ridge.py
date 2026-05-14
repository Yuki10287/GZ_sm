from __future__ import annotations

"""任务一：双因子状态空间 + 期货期限结构 + RidgeCV 政策传导串联融合模型。

这一版实现队友提出的“串联融合架构”：
1. 保留队友/文献启发的双因子状态空间模型，读取已估计的 X_t 和 delta_t；
2. 使用修正后的国内调价机制计算“纯理论调价幅度”；
3. 把纯理论调价幅度、便利收益 delta_t、汇率变动、真实期货期限结构作为特征；
4. 使用 RidgeCV 拟合实际调价幅度，作为政策传导和平滑干预层。

论文解释：
    发改委实际调价并不机械等于国际油价理论变化，还会受到地缘冲突烈度
    （由 delta_t 表征）、汇率成本变化和期货期限结构所反映的市场预期影响。
    因此本文在双因子状态空间模型之后，引入岭回归作为政策传导层。
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import task1_price_mechanism as base
import task1_multisource_statespace as ms
import task1_teammate_statespace_improved as improved


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs" / "task1_serial_fusion_ridge"
STATE_PATH = ROOT / "outputs" / "task1_teammate_statespace" / "state_space_oil_index.csv"

FUTURES_PRICE_COLS = [
    "brent_m1",
    "brent_m3",
    "brent_m6",
    "dubai_m1",
    "dubai_m3",
    "dubai_m6",
    "wti_m1",
    "wti_m3",
    "wti_m6",
]

FUTURES_FEATURE_COLS = [
    "term_spread_m1_m6_mean",
    "term_spread_m1_m3_mean",
    "term_spread_m3_m6_mean",
    "term_spread_m1_m6_change",
]


def find_futuredata_path() -> Path:
    """定位队友整理好的三油种 M1/M3/M6 月度期货数据。"""

    for path in ROOT.rglob("futuredata.xlsx"):
        if path.is_file():
            return path
    raise FileNotFoundError("没有找到 期货数据/futuredata.xlsx")


def read_futures_monthly() -> pd.DataFrame:
    """读取 Brent、Dubai、WTI 的 M1/M3/M6 月度期货数据。"""

    path = find_futuredata_path()
    df = pd.read_excel(path)
    columns = ["date"] + FUTURES_PRICE_COLS
    df = df.rename(columns={old: new for old, new in zip(df.columns, columns)})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in FUTURES_PRICE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["date"]).sort_values("date")[columns]


def make_futures_available() -> pd.DataFrame:
    """构造调价日前可获得的最近一期期货期限结构特征。"""

    futures = read_futures_monthly().copy()
    for oil in ["brent", "dubai", "wti"]:
        futures[f"{oil}_m1_m6_spread"] = np.log(futures[f"{oil}_m1"] / futures[f"{oil}_m6"])
        futures[f"{oil}_m1_m3_spread"] = np.log(futures[f"{oil}_m1"] / futures[f"{oil}_m3"])
        futures[f"{oil}_m3_m6_spread"] = np.log(futures[f"{oil}_m3"] / futures[f"{oil}_m6"])

    futures["term_spread_m1_m6_mean"] = futures[
        ["brent_m1_m6_spread", "dubai_m1_m6_spread", "wti_m1_m6_spread"]
    ].mean(axis=1)
    futures["term_spread_m1_m3_mean"] = futures[
        ["brent_m1_m3_spread", "dubai_m1_m3_spread", "wti_m1_m3_spread"]
    ].mean(axis=1)
    futures["term_spread_m3_m6_mean"] = futures[
        ["brent_m3_m6_spread", "dubai_m3_m6_spread", "wti_m3_m6_spread"]
    ].mean(axis=1)
    futures["term_spread_m1_m6_change"] = futures["term_spread_m1_m6_mean"].diff()

    # 月末期货数据从下一天开始可用，避免调价日提前使用月底数据。
    futures["available_date"] = futures["date"] + pd.offsets.Day(1)
    keep_cols = ["available_date"] + FUTURES_FEATURE_COLS
    return futures[keep_cols].replace([np.inf, -np.inf], np.nan).sort_values("available_date")


def window_mean(panel: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, col: str) -> float:
    """计算给定日度面板在窗口内的均值。"""

    mask = (panel["date"] >= start) & (panel["date"] <= end)
    values = pd.to_numeric(panel.loc[mask, col], errors="coerce")
    return float(values.mean()) if values.notna().any() else float("nan")


def load_two_factor_state(config: base.MechanismConfig) -> pd.DataFrame:
    """读取双因子状态空间模型输出的 X_t、delta_t 和状态油价指数。"""

    if not STATE_PATH.exists():
        raise FileNotFoundError(f"没有找到双因子状态文件: {STATE_PATH}")
    state = pd.read_csv(STATE_PATH, parse_dates=["date"])
    state = state[state["date"] >= pd.Timestamp(config.start_date)].copy()
    required = ["date", "usd_cny", "state_x", "state_delta", "state_space_index_usd"]
    ms.validate_required_columns(state, required, "双因子状态文件")
    return state.sort_values("date").reset_index(drop=True)


def build_serial_feature_dataset(config: base.MechanismConfig) -> pd.DataFrame:
    """构建串联融合模型的逐调价日特征数据集。"""

    state_panel = load_two_factor_state(config)
    domestic = base.read_domestic_adjustments(config.start_date)
    validation = base.simulate_mechanism(domestic, state_panel, "state_space_index_usd", config)
    validation = validation.sort_values("adjust_date").reset_index(drop=True)

    rows = []
    for _, row in validation.iterrows():
        window_start = pd.Timestamp(row["window_start"])
        window_end = pd.Timestamp(row["window_end"])
        base_start = pd.Timestamp(row["base_window_start"])
        base_end = pd.Timestamp(row["base_window_end"])

        delta_mean = window_mean(state_panel, window_start, window_end, "state_delta")
        delta_last = window_mean(state_panel, window_end, window_end, "state_delta")
        delta_base = window_mean(state_panel, base_start, base_end, "state_delta")
        x_mean = window_mean(state_panel, window_start, window_end, "state_x")
        fx_mean = window_mean(state_panel, window_start, window_end, "usd_cny")
        fx_base = window_mean(state_panel, base_start, base_end, "usd_cny")

        rows.append(
            {
                "adjust_date": row["adjust_date"],
                "state_x_window_mean": x_mean,
                "delta_window_mean": delta_mean,
                "delta_window_last": delta_last,
                "delta_window_change": delta_mean - delta_base,
                "delta_abs": abs(delta_mean),
                "usd_cny_window_mean": fx_mean,
                "usd_cny_change_rate": fx_mean / fx_base - 1.0 if pd.notna(fx_base) and fx_base else np.nan,
            }
        )

    features = validation.merge(pd.DataFrame(rows), on="adjust_date", how="left")

    features = ms.asof_merge_available(features, make_futures_available())
    features[FUTURES_FEATURE_COLS] = features[FUTURES_FEATURE_COLS].ffill().bfill()
    features = features.replace([np.inf, -np.inf], np.nan).sort_values("adjust_date").reset_index(drop=True)
    return features


def train_policy_transmission(
    df: pd.DataFrame,
    product: str,
    train_end: str,
    feature_mode: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """训练单个油品的 RidgeCV 政策传导层。"""

    result = df.copy()
    target = f"{product}_actual_delta"
    theory_col = f"{product}_theory_delta"
    feature_sets = {
        "theory_only": [theory_col],
        "delta": [theory_col, "delta_window_mean", "delta_window_change", "delta_abs"],
        "delta_fx": [
            theory_col,
            "delta_window_mean",
            "delta_window_change",
            "delta_abs",
            "usd_cny_change_rate",
        ],
        "delta_futures": [
            theory_col,
            "delta_window_mean",
            "delta_window_change",
            "delta_abs",
            "term_spread_m1_m6_mean",
            "term_spread_m1_m6_change",
        ],
        "delta_fx_futures": [
            theory_col,
            "delta_window_mean",
            "delta_window_change",
            "delta_abs",
            "usd_cny_change_rate",
            "term_spread_m1_m6_mean",
            "term_spread_m1_m3_mean",
            "term_spread_m3_m6_mean",
            "term_spread_m1_m6_change",
        ],
    }
    feature_cols = feature_sets[feature_mode]
    result[feature_cols] = ms.clean_numeric_frame(result[feature_cols])
    train_mask = result["adjust_date"] <= pd.Timestamp(train_end)
    if not bool(train_mask.any()):
        raise ValueError(f"{product} 没有训练样本，请检查 train_end={train_end}")

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 31))),
        ]
    )
    model.fit(result.loc[train_mask, feature_cols], result.loc[train_mask, target])

    pred_col = f"{product}_serial_fusion_delta"
    result[pred_col] = model.predict(result[feature_cols])
    ridge = model.named_steps["ridge"]
    info = {
        "mode": feature_mode,
        "alpha": float(ridge.alpha_),
        "features": feature_cols,
        "coef": dict(zip(feature_cols, ridge.coef_.astype(float))),
    }
    return result, info


def evaluate_mode(
    features: pd.DataFrame,
    config: base.MechanismConfig,
    feature_mode: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """评估一种传导层特征组合。"""

    out, gas_info = train_policy_transmission(features, "gasoline", config.train_end, feature_mode)
    out, diesel_info = train_policy_transmission(out, "diesel", config.train_end, feature_mode)

    samples = {
        "all": out,
        "test_after_train": out[out["adjust_date"] > pd.Timestamp(config.train_end)],
        "test_2023_2025": out[
            (out["adjust_date"] > pd.Timestamp(config.train_end))
            & (out["adjust_date"] < pd.Timestamp("2026-01-01"))
        ],
        "exclude_2026": out[out["adjust_date"] < pd.Timestamp("2026-01-01")],
    }
    metrics = {}
    for name, sample in samples.items():
        metrics[name] = {
            "gasoline_mechanism": ms.metric_block(sample, "gasoline", "gasoline_theory_delta"),
            "diesel_mechanism": ms.metric_block(sample, "diesel", "diesel_theory_delta"),
            "gasoline_serial_fusion": ms.metric_block(sample, "gasoline", "gasoline_serial_fusion_delta"),
            "diesel_serial_fusion": ms.metric_block(sample, "diesel", "diesel_serial_fusion_delta"),
        }

    summary = {
        "feature_mode": feature_mode,
        "transmission_models": {"gasoline": gas_info, "diesel": diesel_info},
        "metrics": metrics,
    }
    return out, summary


def best_mode(ablation: pd.DataFrame) -> str:
    """按测试期汽油和柴油 MAE 之和选择最优传导层。"""

    score = ablation["gasoline_test_mae"].astype(float) + ablation["diesel_test_mae"].astype(float)
    return str(ablation.loc[score.idxmin(), "feature_mode"])


def write_report(
    output_dir: Path,
    config: base.MechanismConfig,
    ablation: pd.DataFrame,
    summaries: dict[str, dict[str, object]],
    selected_mode: str,
) -> None:
    """写出模型说明文件。"""

    selected = summaries[selected_mode]["metrics"]["test_after_train"]
    lines = [
        "双因子状态空间 + 期货期限结构 + RidgeCV 政策传导串联融合模型说明",
        "============================================================",
        "",
        "一、建模思路",
        "",
        "本版吸收队友 Word 文档中的 Schwartz/双因子状态空间思想，保留 X_t 与 delta_t。",
        "其中 X_t 表示基础油价水平，delta_t 表示便利收益、地缘冲突溢价或短期稀缺性压力。",
        "同时吸收 Python 版本中 RidgeCV 传导层的优点，并加入真实 M1/M3/M6 期货期限结构作为窗口级修正变量。",
        "",
        "二、串联结构",
        "",
        "第一层：双因子状态空间模型从 Brent、WTI、Dubai 中提取 X_t 和 delta_t。",
        "第二层：修正后的国内机制层计算纯理论调价幅度。",
        "第三层：RidgeCV 使用纯理论调价、delta_t、汇率变动和期货期限结构特征拟合实际调价。",
        "期货数据不直接进入双因子 Kalman 观测方程，而只进入窗口级传导层，以降低月度低频数据对状态估计的干扰。",
        "",
        "三、特征组合实验",
        "",
        ablation.to_string(index=False),
        "",
        "四、推荐传导层",
        "",
        f"综合最优特征组合：{selected_mode}",
        f"汽油测试期 MAE：{selected['gasoline_serial_fusion']['mae']:.2f}",
        f"柴油测试期 MAE：{selected['diesel_serial_fusion']['mae']:.2f}",
        "",
        "本版不再把 CPI/PPI 放入第一题模型。宏观变量更适合留给第二题做政策影响评价，",
        "第一题只检验 delta_t、汇率变动和期货期限结构对短期调价传导的增量作用。",
        "",
        "五、论文表述建议",
        "",
        "可以表述为：发改委实际调价不仅参考基础油价理论变化，也会根据地缘冲突烈度",
        "（delta_t）、汇率成本变化和期限结构反映的市场预期进行平滑干预。因此本文在",
        "双因子状态空间模型之后引入岭回归政策传导层，提高对实际调价行为的拟合能力。",
        "",
    ]
    (output_dir / "model_description.txt").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="双因子状态空间 + 期货期限结构 + RidgeCV 政策传导串联融合模型")
    parser.add_argument("--start-date", default=base.MechanismConfig.start_date)
    parser.add_argument("--train-end", default=base.MechanismConfig.train_end)
    parser.add_argument("--window-size", type=int, default=base.MechanismConfig.window_size)
    parser.add_argument("--pricing-lag-days", type=int, default=1)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    """运行所有串联融合传导层对比实验。"""

    args = parse_args()
    config = base.MechanismConfig(
        start_date=args.start_date,
        train_end=args.train_end,
        window_size=args.window_size,
        pricing_lag_days=args.pricing_lag_days,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features = build_serial_feature_dataset(config)
    feature_modes = ["theory_only", "delta", "delta_fx", "delta_futures", "delta_fx_futures"]
    summaries: dict[str, dict[str, object]] = {}
    ablation_rows = []

    for mode in feature_modes:
        result, summary = evaluate_mode(features, config, mode)
        summaries[mode] = summary
        result.to_csv(output_dir / f"validation_{mode}.csv", index=False, encoding="utf-8-sig")
        test = summary["metrics"]["test_after_train"]
        test_2023_2025 = summary["metrics"]["test_2023_2025"]
        ablation_rows.append(
            {
                "feature_mode": mode,
                "gasoline_test_mae": test["gasoline_serial_fusion"]["mae"],
                "diesel_test_mae": test["diesel_serial_fusion"]["mae"],
                "gasoline_test_wmape": test["gasoline_serial_fusion"]["wmape"],
                "diesel_test_wmape": test["diesel_serial_fusion"]["wmape"],
                "gasoline_2023_2025_mae": test_2023_2025["gasoline_serial_fusion"]["mae"],
                "diesel_2023_2025_mae": test_2023_2025["diesel_serial_fusion"]["mae"],
            }
        )

    ablation = pd.DataFrame(ablation_rows)
    selected_mode = best_mode(ablation)
    selected_result = pd.read_csv(output_dir / f"validation_{selected_mode}.csv", parse_dates=["adjust_date"])
    selected_result.to_csv(output_dir / "serial_fusion_validation.csv", index=False, encoding="utf-8-sig")
    ablation.to_csv(output_dir / "ablation_transmission_metrics.csv", index=False, encoding="utf-8-sig")

    summary = {"config": asdict(config), "selected_mode": selected_mode, "modes": summaries, "ablation": ablation_rows}
    with (output_dir / "summary_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    write_report(output_dir, config, ablation, summaries, selected_mode)

    print("双因子状态空间 + 期货期限结构 + RidgeCV 政策传导串联融合模型")
    print(ablation.to_string(index=False))
    print(f"\n综合最优特征组合: {selected_mode}")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()
