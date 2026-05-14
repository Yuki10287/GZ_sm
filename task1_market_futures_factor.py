from __future__ import annotations

"""任务一：市场核心变量 + 真实期货期限结构动态因子模型。

这一版用于把第一题模型和第二题宏观影响评价分开：
1. 第一题只关注短周期调价机制，因子层不再使用 CPI、PPI、PMI、GDP；
2. 在 Brent、WTI、Dubai 和汇率的核心信息上，加入真实 M1/M3/M6 期货期限结构；
3. 期货数据不直接使用价格水平，而使用 log(M1/M3)、log(M1/M6)、log(M3/M6)；
4. 月度期货数据按“已可获得的最近一期”合并，不线性插值，避免未来信息泄露；
5. 用 DynamicFactor 提取市场成本压力因子，再用 RidgeCV 做调价传导层。

本脚本与 task1_multisource_statespace.py 的区别：
- 不把 CPI/PPI/PMI/GDP 放进第一题模型；
- 新增真实期货期限结构消融实验；
- 消融时严格控制传导层变量，避免未加入进口成本的方案在 Ridge 层偷偷使用进口变量。
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import task1_multisource_statespace as ms


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs" / "task1_market_futures_factor"


@dataclass(frozen=True)
class MarketFuturesConfig:
    """市场核心因子模型配置。"""

    start_date: str = "2016-01-01"
    train_end: str = "2022-12-31"
    window_size: int = 10
    pricing_lag_days: int = 1
    factor_order: int = 1
    factor_maxiter: int = 1000
    main_variant: str = "C_oil_fx_futures"


@dataclass(frozen=True)
class MarketVariant:
    """市场核心变量消融方案。"""

    name: str
    title: str
    factor_variables: list[str]
    use_import_transmission: bool
    role: str
    description: str


OIL_FACTOR_COLS = ["log_brent_ma10", "log_wti_ma10", "log_dubai_ma10"]
FX_FACTOR_COLS = OIL_FACTOR_COLS + ["log_usd_cny_ma10"]
IMPORT_FACTOR_COLS = FX_FACTOR_COLS + ["log_import_cny_per_ton", "log_import_tons"]
FUTURES_SPREAD_COLS = [
    "brent_m1_m3_spread",
    "brent_m1_m6_spread",
    "brent_m3_m6_spread",
    "dubai_m1_m3_spread",
    "dubai_m1_m6_spread",
    "dubai_m3_m6_spread",
    "wti_m1_m3_spread",
    "wti_m1_m6_spread",
    "wti_m3_m6_spread",
]


VARIANTS = [
    MarketVariant(
        name="A_oil_only",
        title="A 三原油",
        factor_variables=OIL_FACTOR_COLS,
        use_import_transmission=False,
        role="基准",
        description="只使用 Brent、WTI、Dubai 的调价窗口均值提取市场因子。",
    ),
    MarketVariant(
        name="B_oil_fx",
        title="B 三原油 + 汇率",
        factor_variables=FX_FACTOR_COLS,
        use_import_transmission=False,
        role="核心市场因子",
        description="加入美元兑人民币汇率，刻画国内进口成本口径。",
    ),
    MarketVariant(
        name="C_oil_fx_futures",
        title="C 三原油 + 汇率 + 期货期限结构",
        factor_variables=FX_FACTOR_COLS + FUTURES_SPREAD_COLS,
        use_import_transmission=False,
        role="期货检验方案",
        description="在核心市场因子中加入真实 M1/M3/M6 期限价差，检验期货预期信息是否提升效果。",
    ),
    MarketVariant(
        name="D_oil_fx_import",
        title="D 三原油 + 汇率 + 海关进口成本",
        factor_variables=IMPORT_FACTOR_COLS,
        use_import_transmission=True,
        role="进口成本对照",
        description="加入海关进口单价和进口量，并允许传导层使用进口成本变化率。",
    ),
    MarketVariant(
        name="E_oil_fx_futures_import",
        title="E 三原油 + 汇率 + 期货期限结构 + 海关进口成本",
        factor_variables=IMPORT_FACTOR_COLS + FUTURES_SPREAD_COLS,
        use_import_transmission=True,
        role="完整市场成本方案",
        description="同时纳入期货期限结构和海关进口成本，检验二者是否提供互补信息。",
    ),
]


def find_futuredata_path() -> Path:
    """定位队友整理后的真实 M1/M3/M6 期货数据。"""

    for path in ROOT.rglob("futuredata.xlsx"):
        if path.is_file():
            return path
    raise FileNotFoundError("没有找到 期货数据/futuredata.xlsx")


def read_futures_monthly() -> pd.DataFrame:
    """读取 Brent、Dubai、WTI 的 M1/M3/M6 月度期货数据。"""

    path = find_futuredata_path()
    df = pd.read_excel(path)
    columns = [
        "date",
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
    df = df.rename(columns={old: new for old, new in zip(df.columns, columns)})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    return df[columns]


def make_futures_available(futures: pd.DataFrame) -> pd.DataFrame:
    """把月度期货数据转换为调价日可用数据。

    期货表是月末数据。为避免未来信息泄露，某个月末数据只从下一天开始可用；
    调价日匹配不晚于该日的最近一期期货数据。
    """

    out = futures.copy()
    out["available_date"] = out["date"] + pd.offsets.Day(1)
    return out.drop(columns=["date"]).sort_values("available_date")


def add_futures_spreads(features: pd.DataFrame) -> pd.DataFrame:
    """把真实期货期限价差合并到调价窗口样本中。"""

    out = ms.asof_merge_available(features, make_futures_available(read_futures_monthly()))
    for oil in ["brent", "dubai", "wti"]:
        out[f"{oil}_m1_m3_spread"] = np.log(out[f"{oil}_m1"] / out[f"{oil}_m3"])
        out[f"{oil}_m1_m6_spread"] = np.log(out[f"{oil}_m1"] / out[f"{oil}_m6"])
        out[f"{oil}_m3_m6_spread"] = np.log(out[f"{oil}_m3"] / out[f"{oil}_m6"])
    out[FUTURES_SPREAD_COLS] = out[FUTURES_SPREAD_COLS].replace([np.inf, -np.inf], np.nan).ffill().bfill()
    return out


def build_market_dataset(config: MarketFuturesConfig) -> pd.DataFrame:
    """构建第一题市场核心变量数据集。"""

    base_config = ms.MultiSourceConfig(
        start_date=config.start_date,
        train_end=config.train_end,
        window_size=config.window_size,
        pricing_lag_days=config.pricing_lag_days,
        factor_order=config.factor_order,
        factor_maxiter=config.factor_maxiter,
        main_variant=config.main_variant,
    )
    features = ms.build_window_dataset(base_config)
    features = add_futures_spreads(features)
    ms.validate_required_columns(features, FX_FACTOR_COLS + FUTURES_SPREAD_COLS, "市场期货因子数据集")
    return features


def train_predict_product(
    df: pd.DataFrame,
    product: str,
    config: MarketFuturesConfig,
    use_import_transmission: bool,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """训练汽油或柴油的 Ridge 传导层。"""

    result = df.copy()
    target = f"{product}_actual_delta"
    base_col = f"{product}_theory_delta"
    feature_cols = [base_col, "oil_change_rate", "factor_1", "factor_1_diff"]
    if use_import_transmission:
        feature_cols += ["import_cny_per_ton_change", "import_usd_per_ton_change"]

    feature_cols = [col for col in feature_cols if col in result.columns]
    result[feature_cols] = ms.clean_numeric_frame(result[feature_cols])
    train_mask = result["adjust_date"] <= pd.Timestamp(config.train_end)
    if not bool(train_mask.any()):
        raise ValueError(f"{product} 没有训练样本，请检查 train_end={config.train_end}")

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 25))),
        ]
    )
    model.fit(result.loc[train_mask, feature_cols], result.loc[train_mask, target])

    pred_col = f"{product}_market_factor_delta"
    result[pred_col] = model.predict(result[feature_cols])
    ridge = model.named_steps["ridge"]
    return result, {
        "alpha": float(ridge.alpha_),
        "features": feature_cols,
        "coef": dict(zip(feature_cols, ridge.coef_.astype(float))),
    }


def evaluate_variant(
    features: pd.DataFrame,
    config: MarketFuturesConfig,
    variant: MarketVariant,
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    """拟合一个市场核心因子方案并回测。"""

    state_df, fit_info, loadings = ms.fit_dynamic_factor(
        features,
        config,
        variant.factor_variables,
        variant.name,
    )
    state_df, gas_info = train_predict_product(state_df, "gasoline", config, variant.use_import_transmission)
    state_df, diesel_info = train_predict_product(state_df, "diesel", config, variant.use_import_transmission)

    samples = {
        "all": state_df,
        "test_after_train": state_df[state_df["adjust_date"] > pd.Timestamp(config.train_end)],
        "test_2023_2025": state_df[
            (state_df["adjust_date"] > pd.Timestamp(config.train_end))
            & (state_df["adjust_date"] < pd.Timestamp("2026-01-01"))
        ],
        "exclude_2026": state_df[state_df["adjust_date"] < pd.Timestamp("2026-01-01")],
    }

    metrics = {}
    for sample_name, sample in samples.items():
        metrics[sample_name] = {
            "gasoline_mechanism": ms.metric_block(sample, "gasoline", "gasoline_theory_delta"),
            "diesel_mechanism": ms.metric_block(sample, "diesel", "diesel_theory_delta"),
            "gasoline_market_factor": ms.metric_block(sample, "gasoline", "gasoline_market_factor_delta"),
            "diesel_market_factor": ms.metric_block(sample, "diesel", "diesel_market_factor_delta"),
        }

    summary = {
        "title": variant.title,
        "role": variant.role,
        "description": variant.description,
        "factor_variables": variant.factor_variables,
        "use_import_transmission": variant.use_import_transmission,
        "fit": fit_info,
        "transmission_models": {"gasoline": gas_info, "diesel": diesel_info},
        "metrics": metrics,
    }
    return state_df, summary, loadings


def best_variant_name(ablation: pd.DataFrame) -> str:
    """按汽油和柴油测试期 MAE 之和选择最优方案。"""

    score = ablation["gasoline_test_mae"].astype(float) + ablation["diesel_test_mae"].astype(float)
    return str(ablation.loc[score.idxmin(), "variant"])


def write_report(
    output_dir: Path,
    config: MarketFuturesConfig,
    ablation: pd.DataFrame,
    summaries: dict[str, dict[str, object]],
    best_variant: str,
) -> None:
    """写出这一版模型说明。"""

    main = summaries[config.main_variant]["metrics"]["test_after_train"]
    best = summaries[best_variant]["metrics"]["test_after_train"]
    lines = [
        "市场核心变量 + 真实期货期限结构动态因子模型说明",
        "================================================",
        "",
        "一、建模目的",
        "",
        "这一版把 CPI、PPI、PMI、GDP 从第一题模型中移除，将其保留给第二题的政策影响评价。",
        "第一题只使用更贴近短周期调价机制的市场变量：三原油、汇率、真实期货期限结构、海关进口成本。",
        "",
        "二、期货数据处理",
        "",
        "使用 期货数据/futuredata.xlsx 中 Brent、Dubai、WTI 的 M1、M3、M6 数据。",
        "不直接使用期货价格水平，而构造 log(M1/M3)、log(M1/M6)、log(M3/M6) 期限价差。",
        "月度期货数据只从月末后一日开始可用，并用 merge_asof 取调价日前最近一期，避免未来信息泄露。",
        "",
        "三、模型结构",
        "",
        "DynamicFactor 提取一维市场成本压力因子；RidgeCV 传导层把基础机制调价值、油价变化率、",
        "因子水平和因子变化量映射到汽油/柴油实际调价幅度。只有加入海关进口成本的方案，",
        "传导层才使用进口成本变化率。",
        "",
        "四、消融实验",
        "",
        ablation.to_string(index=False),
        "",
        "五、推荐方案和结果",
        "",
        f"默认重点检验方案：{config.main_variant}",
        f"汽油测试期 MAE：{main['gasoline_market_factor']['mae']:.2f}",
        f"柴油测试期 MAE：{main['diesel_market_factor']['mae']:.2f}",
        "",
        f"综合最优方案：{best_variant}",
        f"汽油测试期 MAE：{best['gasoline_market_factor']['mae']:.2f}",
        f"柴油测试期 MAE：{best['diesel_market_factor']['mae']:.2f}",
        "",
        "六、论文建议",
        "",
        "如果期货期限结构方案没有明显优于三原油+汇率，应如实说明：",
        "真实期货期限结构提供了市场预期信息，但在当前样本中对短周期调价幅度的数值预测提升有限。",
        "CPI/PPI/PMI/GDP 更适合放到第二题分析调价机制对宏观经济和价格传导的影响。",
        "",
    ]
    (output_dir / "model_description.txt").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="市场核心变量 + 真实期货期限结构动态因子模型")
    parser.add_argument("--start-date", default=MarketFuturesConfig.start_date)
    parser.add_argument("--train-end", default=MarketFuturesConfig.train_end)
    parser.add_argument("--window-size", type=int, default=MarketFuturesConfig.window_size)
    parser.add_argument("--pricing-lag-days", type=int, default=MarketFuturesConfig.pricing_lag_days)
    parser.add_argument("--factor-order", type=int, default=MarketFuturesConfig.factor_order)
    parser.add_argument("--factor-maxiter", type=int, default=MarketFuturesConfig.factor_maxiter)
    parser.add_argument("--main-variant", default=MarketFuturesConfig.main_variant, choices=[v.name for v in VARIANTS])
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    """运行所有市场核心变量消融实验。"""

    args = parse_args()
    config = MarketFuturesConfig(
        start_date=args.start_date,
        train_end=args.train_end,
        window_size=args.window_size,
        pricing_lag_days=args.pricing_lag_days,
        factor_order=args.factor_order,
        factor_maxiter=args.factor_maxiter,
        main_variant=args.main_variant,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features = build_market_dataset(config)
    summaries: dict[str, dict[str, object]] = {}
    ablation_rows = []

    for variant in VARIANTS:
        state_df, summary, loadings = evaluate_variant(features, config, variant)
        summaries[variant.name] = summary

        state_df.to_csv(output_dir / f"validation_{variant.name}.csv", index=False, encoding="utf-8-sig")
        loadings.to_csv(output_dir / f"loadings_{variant.name}.csv", index=False, encoding="utf-8-sig")
        ms.plot_factor(state_df, variant.name, output_dir)

        if variant.name == config.main_variant:
            state_df.to_csv(output_dir / "market_futures_factor_validation.csv", index=False, encoding="utf-8-sig")
            loadings.to_csv(output_dir / "loadings_market_futures_factor.csv", index=False, encoding="utf-8-sig")

        test = summary["metrics"]["test_after_train"]
        test_2023_2025 = summary["metrics"]["test_2023_2025"]
        ablation_rows.append(
            {
                "variant": variant.name,
                "title": variant.title,
                "role": variant.role,
                "factor_variables": ", ".join(variant.factor_variables),
                "use_import_transmission": variant.use_import_transmission,
                "converged": summary["fit"]["converged"],
                "gasoline_test_mae": test["gasoline_market_factor"]["mae"],
                "diesel_test_mae": test["diesel_market_factor"]["mae"],
                "gasoline_2023_2025_mae": test_2023_2025["gasoline_market_factor"]["mae"],
                "diesel_2023_2025_mae": test_2023_2025["diesel_market_factor"]["mae"],
            }
        )

    ablation = pd.DataFrame(ablation_rows)
    ablation.to_csv(output_dir / "ablation_test_metrics.csv", index=False, encoding="utf-8-sig")
    result = {"config": asdict(config), "futures_path": str(find_futuredata_path()), "variants": summaries, "ablation": ablation_rows}
    with (output_dir / "summary_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    best_variant = best_variant_name(ablation)
    write_report(output_dir, config, ablation, summaries, best_variant)

    print("市场核心变量 + 真实期货期限结构动态因子模型")
    print(ablation.to_string(index=False))
    print(f"\n默认重点检验方案: {config.main_variant}")
    print(f"综合最优方案: {best_variant}")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()
