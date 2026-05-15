"""
任务一变体：队友提出的定价公式验证
=============================================
核心功能：
    实现队友提出的成品油定价公式，与基础模块的公式进行对比验证。

队友公式与基础公式的区别：
    1. 变化率计算方式不同：
       - 基础公式：变化率 = 当前均价 / 基期均价 - 1（使用两次调价日的窗口均价之比）
       - 队友公式：变化率 = (当前均价 - 窗口起始价) / 窗口起始价（使用窗口内首尾价差）
    2. 区间调节函数不同：
       - 基础公式：地板价/天花板价规则（二元判断）
       - 队友公式：φ函数（连续分段：地板价=0，天花板=0.2，正常=1.0）
    3. 调价滞后不同：
       - 基础公式：pricing_lag_days=1（调价日前1天截止计价）
       - 队友公式：pricing_lag_days=0（调价日当天截止计价）
    4. 窗口选择不同：
       - 基础公式：取截止日前最近window_size条数据
       - 队友公式：严格取两次调价日之间的数据，不足时向前补足
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

import task1_price_mechanism as base  # 引入基础模块


# ==================== 路径配置 ====================

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "task1_teammate"


# ==================== 队友公式核心函数 ====================

def teammate_phi(ma_usd: float, config: base.MechanismConfig) -> float:
    """
    队友公式的区间调节函数 φ(price)

    与基础公式的区别：
        - 基础公式对涨跌幅做截断（地板价不降、天花板价限制涨幅）
        - 队友公式对整个调价乘数做调节（直接乘以φ系数）

    参数：
        ma_usd: 当前窗口油价均值（美元/桶）
        config: 机制配置
    返回：
        φ系数值：
            - 地板价区间（≤40美元/桶）：φ=0（完全不调价）
            - 天花板价区间（≥130美元/桶）：φ=0.2（只调20%）
            - 正常区间：φ=1.0（全额调价）
    """
    if ma_usd <= config.floor_usd_per_bbl:
        return 0.0       # 地板价区间：完全不调
    if ma_usd >= config.ceiling_usd_per_bbl:
        return config.ceiling_up_factor  # 天花板区间：只调20%
    return 1.0           # 正常区间：全额调


def teammate_window(
    panel: pd.DataFrame,
    prev_date: pd.Timestamp,
    date: pd.Timestamp,
    price_col: str,
    config: base.MechanismConfig,
) -> pd.DataFrame:
    """
    队友公式的窗口数据获取

    与基础公式的区别：
        - 基础公式：取截止日前最近window_size条数据（可能跨多个调价周期）
        - 队友公式：严格取两次调价日之间的数据，不足window_size条时向前补足

    参数：
        panel: 油价面板数据
        prev_date: 上一次调价日期
        date: 本次调价日期
        price_col: 价格列名
        config: 机制配置
    返回：
        窗口内的数据子集
    """
    # 计算计价窗口截止日
    end = date - pd.Timedelta(days=config.pricing_lag_days)

    # 取两次调价日之间的数据
    window = panel[(panel["date"] > prev_date) & (panel["date"] <= end)].copy()

    # 如果数据不足window_size条，向前补足
    if len(window) < config.window_size:
        window = panel[panel["date"] <= end].tail(config.window_size).copy()
    else:
        window = window.head(config.window_size)

    return window[["date", price_col, "usd_cny"]].dropna()


def simulate_teammate_control(
    domestic: pd.DataFrame,
    panel: pd.DataFrame,
    price_col: str,
    config: base.MechanismConfig,
) -> pd.DataFrame:
    """
    模拟队友的定价公式

    队友公式：
        变化率 = (窗口均价 - 窗口起始价) / 窗口起始价
        汽油调价 = 窗口均价 × 变化率 × 汽油吨桶比 × 汇率 × 税费系数 × φ
        柴油调价 = 窗口均价 × 变化率 × 柴油吨桶比 × 汇率 × 税费系数 × φ

    与基础公式的对比：
        基础公式：变化率 = 当前均价_ref / 基期均价_ref - 1（使用两次调价日的窗口均价之比）
        队友公式：变化率 = (当前均价 - 起始价) / 起始价（使用单个窗口内的首尾价差）

    参数：
        domestic: 国内调价记录
        panel: 国际油价面板数据
        price_col: 使用的价格指标列名
        config: 机制配置
    返回：
        模拟结果DataFrame
    """
    rows: list[dict[str, object]] = []
    carry_gasoline = 0.0  # 汽油累加余额
    carry_diesel = 0.0    # 柴油累加余额

    for i in range(1, len(domestic)):
        prev_date = domestic.loc[i - 1, "adjust_date"]  # 上一次调价日
        date = domestic.loc[i, "adjust_date"]            # 本次调价日

        # 获取窗口数据
        window = teammate_window(panel, prev_date, date, price_col, config)
        if window.empty:
            continue

        # 计算窗口统计量
        start_usd = float(window[price_col].iloc[0])   # 窗口起始价
        ma_usd = float(window[price_col].mean())        # 窗口均价
        fx_avg = float(window["usd_cny"].mean())        # 窗口平均汇率

        if not np.isfinite(start_usd) or start_usd == 0:
            continue

        # 计算变化率（队友公式：用窗口首尾价差）
        change_rate = (ma_usd - start_usd) / start_usd

        # 获取区间调节系数φ
        phi = teammate_phi(ma_usd, config)

        # 计算原始调价幅度
        gasoline_raw = ma_usd * change_rate * config.gasoline_bbl_per_ton * fx_avg * config.tax_factor
        diesel_raw = ma_usd * change_rate * config.diesel_bbl_per_ton * fx_avg * config.tax_factor

        # 应用φ系数和累加机制
        gasoline_total = gasoline_raw * phi + carry_gasoline
        diesel_total = diesel_raw * phi + carry_diesel

        # 判断是否达到调价门槛（50元/吨）
        gasoline_theory = gasoline_total if abs(gasoline_total) >= config.threshold_yuan_per_ton else 0.0
        diesel_theory = diesel_total if abs(diesel_total) >= config.threshold_yuan_per_ton else 0.0

        # 更新累加余额
        carry_gasoline = 0.0 if gasoline_theory else gasoline_total
        carry_diesel = 0.0 if diesel_theory else diesel_total

        # 记录本轮模拟结果
        rows.append(
            {
                "adjust_date": date,                        # 调价日期
                "prev_adjust_date": prev_date,              # 上次调价日期
                "oil_index": price_col,                     # 使用的油价指标
                "window_observations": len(window),         # 窗口观测数
                "window_start": window["date"].min(),       # 窗口起始日
                "window_end": window["date"].max(),         # 窗口结束日
                "start_usd_per_bbl": start_usd,             # 窗口起始价
                "ma_usd_per_bbl": ma_usd,                   # 窗口均价
                "avg_usd_cny": fx_avg,                      # 窗口平均汇率
                "oil_change_rate": change_rate,             # 油价变化率
                "phi": phi,                                 # 区间调节系数φ
                "zone": base.zone_for_price(ma_usd, config),  # 价格区间
                "gasoline_formula_delta": gasoline_raw,     # 汽油公式原始值
                "diesel_formula_delta": diesel_raw,         # 柴油公式原始值
                "gasoline_theory_delta": gasoline_theory,   # 汽油理论调价（含φ和门槛）
                "diesel_theory_delta": diesel_theory,       # 柴油理论调价
                "gasoline_carry_after": carry_gasoline,     # 汽油累加余额
                "diesel_carry_after": carry_diesel,         # 柴油累加余额
                "gasoline_actual_delta": domestic.loc[i, "gasoline_actual_delta"],  # 汽油实际调价
                "diesel_actual_delta": domestic.loc[i, "diesel_actual_delta"],      # 柴油实际调价
                "gasoline_price": domestic.loc[i, "gasoline_price"],                # 汽油价格
                "diesel_price": domestic.loc[i, "diesel_price"],                    # 柴油价格
                "ref_ma_usd_per_bbl": ma_usd,               # 参考均价（用于NARDL检验）
            }
        )

    return pd.DataFrame(rows)


# ==================== 评估函数 ====================

def build_summary(df: pd.DataFrame) -> dict[str, object]:
    """
    构建评估摘要

    包含：
        - 汽油和柴油的整体评估指标
        - 按价格区间分组的评估指标

    参数：
        df: 模拟结果DataFrame
    返回：
        评估摘要字典
    """
    return {
        "gasoline": base.metric_block(df, "gasoline", "gasoline_theory_delta"),
        "diesel": base.metric_block(df, "diesel", "diesel_theory_delta"),
        "zone_summary": [
            {
                "zone": zone,
                "n": int(len(group)),
                "gasoline": base.metric_block(group, "gasoline", "gasoline_theory_delta"),
                "diesel": base.metric_block(group, "diesel", "diesel_theory_delta"),
            }
            for zone, group in df.groupby("zone", dropna=False)
        ],
    }


# ==================== 命令行参数 ====================

def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="运行队友的原始定价公式验证")
    parser.add_argument("--start-date", default=base.MechanismConfig.start_date, help="分析起始日期")
    parser.add_argument("--window-size", type=int, default=base.MechanismConfig.window_size, help="窗口大小")
    parser.add_argument("--pricing-lag-days", type=int, default=0, help="定价滞后天数（队友默认为0）")
    return parser.parse_args()


# ==================== 主函数 ====================

def main() -> None:
    """
    主函数：执行队友公式的验证流程

    流程：
        1. 解析参数
        2. 读取油价面板和国内调价数据
        3. 分别用两种油价指数（固定篮子、PCA+卡尔曼）模拟
        4. 计算评估指标和NARDL检验
        5. 输出结果
    """
    args = parse_args()
    config = base.MechanismConfig(
        start_date=args.start_date,
        window_size=args.window_size,
        pricing_lag_days=args.pricing_lag_days,
    )

    # 读取数据
    panel = base.read_oil_panel()
    domestic = base.read_domestic_adjustments(config.start_date)

    # 分别用两种油价指数进行模拟
    outputs: dict[str, pd.DataFrame] = {}
    summaries: dict[str, object] = {}

    for name, price_col in {
        "fixed_basket": "basket_usd",           # 固定加权篮子
        "pca_kalman": "kalman_index_usd",       # PCA+卡尔曼平滑指数
    }.items():
        df = simulate_teammate_control(domestic, panel, price_col, config)
        outputs[name] = df
        summaries[name] = {
            "oil_index_column": price_col,
            "pca_weights": panel.attrs.get("pca_weights", {}),
            "metrics": build_summary(df),
            "nardl_like": {
                "gasoline": base.run_nardl_like_test(df, "gasoline"),
                "diesel": base.run_nardl_like_test(df, "diesel"),
            },
        }

    # 输出结果
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in outputs.items():
        df.to_csv(OUTPUT_DIR / f"teammate_validation_{name}.csv", index=False, encoding="utf-8-sig")
    with (OUTPUT_DIR / "summary_metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"config": asdict(config), "summaries": summaries}, f, ensure_ascii=False, indent=2)

    # 打印报告
    print("队友原始公式验证结果")
    for name, summary in summaries.items():
        print(f"\n[{name}]")
        print("PCA权重:", summary["pca_weights"])
        print("汽油:", summary["metrics"]["gasoline"])
        print("柴油:", summary["metrics"]["diesel"])
        print("NARDL汽油:", summary["nardl_like"]["gasoline"])
        print("NARDL柴油:", summary["nardl_like"]["diesel"])
    print(f"\n输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
