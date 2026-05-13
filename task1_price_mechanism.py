"""
任务一：成品油价格调控机制验证模块
=============================================
核心功能：
    1. 读取国际原油价格数据（Brent/WTI/Dubai）和汇率数据
    2. 构建原油价格指数（固定加权 + PCA+卡尔曼平滑两种方案）
    3. 模拟中国成品油定价机制，计算理论调价幅度
    4. 与实际调价数据对比验证，评估模型精度
    5. 进行NARDL-like非对称传导检验

数学建模思路：
    - 中国成品油定价机制：以国际原油价格为基准，每10个工作日为一个调价窗口
    - 调价公式：ΔP = P_ref × 桶/吨 × 汇率 × 变化率 × 税费系数
    - 地板价/天花板价机制：油价低于40美元/桶不降，高于130美元/桶不全额涨
    - 50元/吨门槛：调价幅度不足50元/吨时累加到下一周期
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant


# ==================== 路径配置 ====================

ROOT = Path(__file__).resolve().parent                    # 项目根目录
OIL_DIR = ROOT / "国际原油价格数据"                        # 国际油价数据目录
DOMESTIC_PATH = ROOT / "国内柴油汽油调价" / "柴油汽油调价2002-2026.xlsx"  # 国内调价记录
FX_PATH = ROOT / "汇率" / "DEXCHUS.csv"                  # 美元兑人民币汇率
OUTPUT_DIR = ROOT / "outputs" / "task1"                   # 输出目录


# ==================== 参数配置 ====================

@dataclass(frozen=True)
class MechanismConfig:
    """
    成品油定价机制参数配置

    属性：
        start_date: 分析起始日期
        window_size: 调价窗口大小（工作日数）
        pricing_lag_days: 定价滞后天数（调价日之前的数据截止日）
        threshold_yuan_per_ton: 调价门槛（元/吨），不足此值不调价
        floor_usd_per_bbl: 地板价（美元/桶），低于此价不降价
        ceiling_usd_per_bbl: 天花板价（美元/桶），高于此价限制涨幅
        ceiling_up_factor: 天花板价区间的涨幅限制系数
        tax_factor: 税费系数（含增值税等）
        gasoline_bbl_per_ton: 汽油吨桶比（1吨汽油≈8.6桶）
        diesel_bbl_per_ton: 柴油吨桶比（1吨柴油≈7.3桶）
        train_end: 训练集截止日期（用于校准）
    """
    start_date: str = "2016-01-01"
    window_size: int = 10
    pricing_lag_days: int = 1
    threshold_yuan_per_ton: float = 50.0
    floor_usd_per_bbl: float = 40.0
    ceiling_usd_per_bbl: float = 130.0
    ceiling_up_factor: float = 0.2
    tax_factor: float = 1.13
    gasoline_bbl_per_ton: float = 8.6
    diesel_bbl_per_ton: float = 7.3
    train_end: str = "2022-12-31"


# ==================== 数据读取函数 ====================

def _clean_numeric(series: pd.Series) -> pd.Series:
    """
    清洗数值列：去除逗号和百分号，转换为数值类型

    参数：
        series: 原始字符串Series
    返回：
        数值类型的Series，无法转换的值为NaN
    """
    return pd.to_numeric(
        series.astype("string").str.replace(",", "", regex=False).str.replace("%", "", regex=False),
        errors="coerce",
    )


def read_eia_oil_csv(path: Path, value_name: str) -> pd.DataFrame:
    """
    读取EIA（美国能源信息署）油价CSV文件

    参数：
        path: CSV文件路径
        value_name: 价格列的名称（如"brent"、"wti"）
    返回：
        包含date和价格列的DataFrame
    """
    df = pd.read_csv(path, skiprows=4)  # 跳过前4行元数据
    df = df.rename(columns={df.columns[0]: "date", df.columns[1]: value_name})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df[value_name] = _clean_numeric(df[value_name])
    return df[["date", value_name]].dropna().sort_values("date")


def read_dubai_csv(path: Path) -> pd.DataFrame:
    """
    读取迪拜原油价格CSV文件

    参数：
        path: CSV文件路径
    返回：
        包含date和dubai列的DataFrame
    """
    df = pd.read_csv(path)
    df = df.rename(columns={"Date": "date", "Price": "dubai"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["dubai"] = _clean_numeric(df["dubai"])
    return df[["date", "dubai"]].dropna().sort_values("date")


def read_fx(path: Path) -> pd.DataFrame:
    """
    读取美元兑人民币汇率数据

    参数：
        path: CSV文件路径
    返回：
        包含date和usd_cny列的DataFrame
    """
    df = pd.read_csv(path)
    df = df.rename(columns={"observation_date": "date", "DEXCHUS": "usd_cny"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["usd_cny"] = _clean_numeric(df["usd_cny"])
    return df[["date", "usd_cny"]].dropna().sort_values("date")


def read_oil_panel() -> pd.DataFrame:
    """
    构建原油价格面板数据

    处理流程：
        1. 读取三种国际原油价格（Brent、WTI、Dubai）
        2. 读取汇率数据
        3. 按日期外连接合并
        4. 前向填充缺失值
        5. 计算固定加权篮子价格：basket = 0.4×Brent + 0.1×WTI + 0.5×Dubai
        6. 计算PCA+卡尔曼平滑指数

    返回：
        包含所有价格指标的面板DataFrame
    """
    # 读取各数据源
    brent = read_eia_oil_csv(OIL_DIR / "Europe_Brent_Spot_Price_FOB.csv", "brent")
    wti = read_eia_oil_csv(OIL_DIR / "Cushing_OK_WTI_Spot_Price_FOB.csv", "wti")
    dubai = read_dubai_csv(OIL_DIR / "Dubai Crude Oil (Platts) Financial Futures Historical Data 2010-2026.csv")
    fx = read_fx(FX_PATH)

    # 合并所有数据源（外连接，保留所有日期）
    panel = brent.merge(wti, on="date", how="outer").merge(dubai, on="date", how="outer")
    panel = panel.merge(fx, on="date", how="outer").sort_values("date")

    # 前向填充缺失值（使用最近的有效值）
    panel[["brent", "wti", "dubai", "usd_cny"]] = panel[["brent", "wti", "dubai", "usd_cny"]].ffill()
    panel = panel.dropna(subset=["brent", "wti", "dubai", "usd_cny"]).reset_index(drop=True)

    # 计算固定加权篮子价格（中国参考的一揽子油价）
    panel["basket_usd"] = 0.4 * panel["brent"] + 0.1 * panel["wti"] + 0.5 * panel["dubai"]

    # 计算PCA+卡尔曼平滑指数
    panel, weights = add_pca_kalman_index(panel)
    panel.attrs["pca_weights"] = weights
    return panel


# ==================== 卡尔曼滤波 ====================

def local_level_kalman_smooth(observed: np.ndarray) -> np.ndarray:
    """
    局部水平模型的卡尔曼平滑（Rauch-Tung-Striebel平滑器）

    模型设定：
        状态方程：x_t = x_{t-1} + w_t,  w_t ~ N(0, q)
        观测方程：y_t = x_t + v_t,       v_t ~ N(0, r)

    用途：对油价对数序列进行平滑，去除短期噪声，提取长期趋势

    参数：
        observed: 观测值序列（油价对数）
    返回：
        平滑后的状态估计序列

    算法步骤：
        1. 前向滤波（Kalman Filter）：逐时刻更新状态估计
        2. 后向平滑（RTS Smoother）：利用未来信息修正历史估计
    """
    observed = np.asarray(observed, dtype=float)

    # 估计观测噪声方差（基于一阶差分）
    diff_var = float(np.nanvar(np.diff(observed)))
    if not np.isfinite(diff_var) or diff_var <= 0:
        return observed.copy()

    # 过程噪声和观测噪声的方差设定
    q = max(diff_var * 0.05, 1e-8)  # 过程噪声方差（较小，表示状态变化缓慢）
    r = max(diff_var * 0.50, 1e-8)  # 观测噪声方差（较大，表示观测有噪声）
    n = observed.size

    # === 前向滤波 ===
    level = np.zeros(n)       # 状态估计
    level_var = np.zeros(n)   # 状态估计方差
    pred = np.zeros(n)        # 预测值
    pred_var = np.zeros(n)    # 预测方差

    level[0] = observed[0]    # 初始状态设为第一个观测值
    level_var[0] = r          # 初始方差设为观测噪声

    for t in range(1, n):
        # 预测步：基于上一时刻状态预测当前状态
        pred[t] = level[t - 1]
        pred_var[t] = level_var[t - 1] + q

        # 更新步：根据观测值修正预测
        gain = pred_var[t] / (pred_var[t] + r)  # 卡尔曼增益
        level[t] = pred[t] + gain * (observed[t] - pred[t])  # 状态更新
        level_var[t] = (1.0 - gain) * pred_var[t]            # 方差更新

    # === 后向平滑（RTS平滑器） ===
    smooth = level.copy()
    smooth_var = level_var.copy()

    for t in range(n - 2, -1, -1):
        # 利用未来信息修正当前时刻的状态估计
        gain = level_var[t] / (level_var[t] + q)
        smooth[t] = level[t] + gain * (smooth[t + 1] - level[t])
        smooth_var[t] = level_var[t] + gain * gain * (smooth_var[t + 1] - level_var[t] - q)

    return smooth


# ==================== PCA+卡尔曼指数构建 ====================

def add_pca_kalman_index(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    构建PCA+卡尔曼平滑油价指数

    算法流程：
        1. 对Brent/WTI/Dubai取对数
        2. 标准化后做PCA，提取第一主成分作为权重
        3. 用加权对数价格构建指数
        4. 对指数进行卡尔曼平滑，得到去噪后的指数

    参数：
        panel: 包含brent/wti/dubai列的面板数据
    返回：
        (添加了pca_index和kalman_index的面板, PCA权重字典)
    """
    result = panel.copy()
    pca_prices = result[["brent", "wti", "dubai"]].copy()

    # 处理非正值（取对数前必须为正）
    pca_prices = pca_prices.mask(pca_prices <= 0)
    pca_prices = pca_prices.interpolate(limit_direction="both").ffill().bfill()

    # 取对数并标准化
    log_prices = np.log(pca_prices)
    scaled = StandardScaler().fit_transform(log_prices)

    # PCA提取第一主成分
    pca = PCA(n_components=1)
    pca.fit(scaled)
    loadings = pca.components_[0]

    # 确保载荷方向一致（与第一个变量正相关）
    if loadings.sum() < 0:
        loadings = -loadings

    # 转换为非负权重并归一化
    weights = np.maximum(loadings, 0)
    if weights.sum() == 0:
        weights = np.abs(loadings)
    weights = weights / weights.sum()

    # 构建权重映射
    weight_map = {name: float(weight) for name, weight in zip(["brent", "wti", "dubai"], weights.round(6))}

    # 计算加权对数指数并转回原始尺度
    log_index = log_prices.to_numpy(dtype=float) @ weights
    result["pca_index_usd"] = np.exp(log_index)                    # PCA指数
    result["kalman_index_usd"] = np.exp(local_level_kalman_smooth(log_index))  # 卡尔曼平滑指数

    return result, weight_map


# ==================== 国内调价数据读取 ====================

def read_domestic_adjustments(start_date: str) -> pd.DataFrame:
    """
    读取国内成品油调价记录

    参数：
        start_date: 起始日期，只保留此日期之后的数据
    返回：
        包含调价日期、汽柴油价格和涨跌幅度的DataFrame
    """
    df = pd.read_excel(DOMESTIC_PATH, sheet_name="Sheet1")

    # 列名映射（中文→英文）
    df = df.rename(
        columns={
            "调整日期": "adjust_date",
            "汽油_价格": "gasoline_price",
            "汽油_涨跌": "gasoline_actual_delta",
            "柴油_价格": "diesel_price",
            "柴油_涨跌": "diesel_actual_delta",
        }
    )

    df["adjust_date"] = pd.to_datetime(df["adjust_date"], errors="coerce")

    # 清洗数值列
    numeric_cols = ["gasoline_price", "gasoline_actual_delta", "diesel_price", "diesel_actual_delta"]
    for col in numeric_cols:
        df[col] = _clean_numeric(df[col])

    df = df.dropna(subset=["adjust_date"]).sort_values("adjust_date").reset_index(drop=True)
    df = df[df["adjust_date"] >= pd.Timestamp(start_date)].reset_index(drop=True)
    return df


# ==================== 窗口计算工具函数 ====================

def last_value_before(panel: pd.DataFrame, date: pd.Timestamp, col: str) -> float:
    """
    获取指定日期之前某列的最新值

    参数：
        panel: 面板数据
        date: 截止日期
        col: 列名
    返回：
        最新值，无数据时返回NaN
    """
    values = panel.loc[panel["date"] <= date, col]
    if values.empty:
        return float("nan")
    return float(values.iloc[-1])


def window_values(panel: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, col: str, size: int) -> pd.DataFrame:
    """
    获取指定时间窗口内的数据

    参数：
        panel: 面板数据
        start: 窗口起始日期
        end: 窗口结束日期
        col: 价格列名
        size: 窗口大小
    返回：
        窗口内的数据子集
    """
    window = panel[(panel["date"] > start) & (panel["date"] <= end)].copy()
    if window.empty:
        # 如果窗口内无数据，取截止日期前最近的size条
        window = panel[panel["date"] <= end].tail(size).copy()
    else:
        window = window.tail(size)
    return window[["date", col, "usd_cny"]].dropna()


def trailing_window(panel: pd.DataFrame, end: pd.Timestamp, col: str, size: int) -> pd.DataFrame:
    """
    获取截止日期前的滑动窗口数据

    参数：
        panel: 面板数据
        end: 截止日期
        col: 价格列名
        size: 窗口大小（数据条数）
    返回：
        最近size条有效数据
    """
    return panel.loc[panel["date"] <= end, ["date", col, "usd_cny"]].dropna().tail(size).copy()


# ==================== 价格区间判定 ====================

def zone_for_price(price: float, config: MechanismConfig) -> str:
    """
    判断油价所处区间

    参数：
        price: 当前油价（美元/桶）
        config: 机制配置
    返回：
        "floor"（地板价区）、"ceiling"（天花板价区）或 "normal"（正常区间）
    """
    if price <= config.floor_usd_per_bbl:
        return "floor"
    if price >= config.ceiling_usd_per_bbl:
        return "ceiling"
    return "normal"


def apply_zone_rule(delta: float, ma_usd: float, config: MechanismConfig) -> float:
    """
    应用地板价/天花板价规则

    规则说明：
        - 地板价区间（≤40美元/桶）：如果计算结果为降价，则不调（返回0）
        - 天花板价区间（≥130美元/桶）：如果计算结果为涨价，只涨20%
        - 正常区间：按计算结果调价

    参数：
        delta: 原始计算的调价幅度
        ma_usd: 当前油价均值
        config: 机制配置
    返回：
        经过区间规则调整后的调价幅度
    """
    if ma_usd <= config.floor_usd_per_bbl and delta < 0:
        return 0.0  # 地板价区间不降价
    if ma_usd >= config.ceiling_usd_per_bbl and delta > 0:
        return delta * config.ceiling_up_factor  # 天花板区间限制涨幅
    return delta


# ==================== 核心模拟函数 ====================

def simulate_mechanism(
    domestic: pd.DataFrame,
    panel: pd.DataFrame,
    price_col: str,
    config: MechanismConfig,
) -> pd.DataFrame:
    """
    模拟成品油定价机制

    核心公式：
        变化率 = (当前窗口均价 / 基期窗口均价) - 1
        汽油调价 = 当前均价 × 汽油吨桶比 × 汇率 × 变化率 × 税费系数
        柴油调价 = 当前均价 × 柴油吨桶比 × 汇率 × 变化率 × 税费系数

    累加机制：
        - 如果某次调价幅度不足50元/吨，则累加到下次
        - 累加后达到门槛才触发调价

    参数：
        domestic: 国内调价记录
        panel: 国际油价面板数据
        price_col: 使用的价格指标列名
        config: 机制配置
    返回：
        模拟结果DataFrame，包含理论调价幅度和实际调价幅度
    """
    rows: list[dict[str, object]] = []
    carry_gasoline = 0.0  # 汽油累加余额
    carry_diesel = 0.0    # 柴油累加余额

    for i in range(1, len(domestic)):
        prev_date = domestic.loc[i - 1, "adjust_date"]  # 上一次调价日
        date = domestic.loc[i, "adjust_date"]            # 本次调价日

        # 计价窗口结束日（调价日前pricing_lag_days天）
        pricing_end = date - pd.Timedelta(days=config.pricing_lag_days)
        base_pricing_end = prev_date - pd.Timedelta(days=config.pricing_lag_days)

        # 获取当前窗口和基期窗口的数据
        window = trailing_window(panel, pricing_end, price_col, config.window_size)
        if window.empty:
            continue

        # 计算当前窗口均价
        current_ma = float(window[price_col].mean())
        current_ma_ref = max(current_ma, config.floor_usd_per_bbl)  # 地板价保护

        # 计算基期窗口均价
        base_window = trailing_window(panel, base_pricing_end, price_col, config.window_size)
        base_usd = float(base_window[price_col].mean()) if not base_window.empty else float("nan")
        base_ref = max(base_usd, config.floor_usd_per_bbl)  # 地板价保护

        # 计算窗口内平均汇率
        fx_avg = float(window["usd_cny"].mean())

        if not np.isfinite(base_ref) or base_ref == 0:
            continue

        # 计算价格变化率
        change_rate = current_ma_ref / base_ref - 1.0

        # 计算理论调价幅度（元/吨）
        # 公式：油价 × 吨桶比 × 汇率 × 变化率 × 税费系数
        gasoline_raw = (
            current_ma_ref
            * config.gasoline_bbl_per_ton
            * fx_avg
            * change_rate
            * config.tax_factor
        )
        diesel_raw = current_ma_ref * config.diesel_bbl_per_ton * fx_avg * change_rate * config.tax_factor

        # 应用地板价/天花板价规则
        gasoline_policy = apply_zone_rule(gasoline_raw, current_ma, config)
        diesel_policy = apply_zone_rule(diesel_raw, current_ma, config)

        # 应用累加机制（不足50元/吨门槛则累加）
        gasoline_total = gasoline_policy + carry_gasoline
        diesel_total = diesel_policy + carry_diesel

        # 判断是否达到调价门槛
        gasoline_theory = gasoline_total if abs(gasoline_total) >= config.threshold_yuan_per_ton else 0.0
        diesel_theory = diesel_total if abs(diesel_total) >= config.threshold_yuan_per_ton else 0.0

        # 更新累加余额
        carry_gasoline = 0.0 if gasoline_theory else gasoline_total
        carry_diesel = 0.0 if diesel_theory else diesel_total

        # 记录本轮模拟结果
        rows.append(
            {
                "adjust_date": date,                    # 调价日期
                "prev_adjust_date": prev_date,          # 上次调价日期
                "oil_index": price_col,                 # 使用的油价指标
                "pricing_end": pricing_end,             # 计价窗口截止日
                "window_observations": len(window),     # 窗口观测数
                "window_start": window["date"].min(),   # 窗口起始日
                "window_end": window["date"].max(),     # 窗口结束日
                "base_usd_per_bbl": base_usd,           # 基期油价
                "base_window_observations": len(base_window),  # 基期窗口观测数
                "base_window_start": base_window["date"].min() if not base_window.empty else pd.NaT,
                "base_window_end": base_window["date"].max() if not base_window.empty else pd.NaT,
                "ma_usd_per_bbl": current_ma,           # 当前窗口均价
                "ref_ma_usd_per_bbl": current_ma_ref,   # 参考均价（含地板价保护）
                "avg_usd_cny": fx_avg,                  # 窗口平均汇率
                "oil_change_rate": change_rate,         # 油价变化率
                "zone": zone_for_price(current_ma, config),  # 价格区间
                "gasoline_formula_delta": gasoline_raw,      # 汽油公式原始值
                "diesel_formula_delta": diesel_raw,          # 柴油公式原始值
                "gasoline_theory_delta": gasoline_theory,    # 汽油理论调价（含规则和门槛）
                "diesel_theory_delta": diesel_theory,        # 柴油理论调价
                "gasoline_carry_after": carry_gasoline,      # 汽油累加余额
                "diesel_carry_after": carry_diesel,          # 柴油累加余额
                "gasoline_actual_delta": domestic.loc[i, "gasoline_actual_delta"],  # 汽油实际调价
                "diesel_actual_delta": domestic.loc[i, "diesel_actual_delta"],      # 柴油实际调价
                "gasoline_price": domestic.loc[i, "gasoline_price"],                # 汽油价格
                "diesel_price": domestic.loc[i, "diesel_price"],                    # 柴油价格
            }
        )

    return pd.DataFrame(rows)


# ==================== 模型校准 ====================

def fit_scale(train_actual: pd.Series, train_theory: pd.Series) -> float:
    """
    最小二乘校准：计算理论值到实际值的缩放系数

    公式：scale = Σ(x·y) / Σ(x²)
    其中x为理论值，y为实际值

    参数：
        train_actual: 训练集实际调价幅度
        train_theory: 训练集理论调价幅度
    返回：
        缩放系数
    """
    mask = train_actual.notna() & train_theory.notna() & (train_theory.abs() > 1e-9)
    if mask.sum() < 5:
        return 1.0  # 样本不足时返回1（不缩放）
    x = train_theory[mask].to_numpy(dtype=float)
    y = train_actual[mask].to_numpy(dtype=float)
    return float(np.dot(x, y) / np.dot(x, x))


def add_calibrated_predictions(df: pd.DataFrame, config: MechanismConfig) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    添加校准后的预测值

    使用训练集（train_end之前）拟合缩放系数，然后应用到全部数据

    参数：
        df: 模拟结果DataFrame
        config: 机制配置
    返回：
        (添加了校准预测列的DataFrame, 缩放系数字典)
    """
    result = df.copy()
    train_mask = result["adjust_date"] <= pd.Timestamp(config.train_end)

    # 分别为汽油和柴油拟合缩放系数
    scales = {
        "gasoline": fit_scale(
            result.loc[train_mask, "gasoline_actual_delta"],
            result.loc[train_mask, "gasoline_theory_delta"],
        ),
        "diesel": fit_scale(
            result.loc[train_mask, "diesel_actual_delta"],
            result.loc[train_mask, "diesel_theory_delta"],
        ),
    }

    # 应用缩放系数得到校准预测值
    result["gasoline_calibrated_delta"] = result["gasoline_theory_delta"] * scales["gasoline"]
    result["diesel_calibrated_delta"] = result["diesel_theory_delta"] * scales["diesel"]
    return result, scales


# ==================== 评估指标 ====================

def metric_block(df: pd.DataFrame, product: str, pred_col: str) -> dict[str, float]:
    """
    计算单个产品的预测评估指标

    指标说明：
        - n: 样本量
        - mae: 平均绝对误差
        - rmse: 均方根误差
        - mean_error: 平均误差（正负偏差）
        - direction_accuracy: 方向准确率（涨跌方向是否一致）
        - threshold_accuracy: 门槛准确率（是否正确判断是否达到50元/吨门槛）
        - corr: 相关系数

    参数：
        df: 模拟结果
        product: 产品名（"gasoline"或"diesel"）
        pred_col: 预测值列名
    返回：
        评估指标字典
    """
    actual_col = f"{product}_actual_delta"
    mask = df[actual_col].notna() & df[pred_col].notna()
    actual = df.loc[mask, actual_col].to_numpy(dtype=float)
    pred = df.loc[mask, pred_col].to_numpy(dtype=float)

    if actual.size == 0:
        return {"n": 0}

    err = pred - actual
    nonzero_mask = actual != 0

    # 方向准确率：预测方向与实际方向一致的比例
    direction_accuracy = np.mean(np.sign(pred[nonzero_mask]) == np.sign(actual[nonzero_mask])) if nonzero_mask.any() else np.nan

    # 门槛准确率：是否正确判断达到50元/吨门槛
    threshold_accuracy = np.mean((np.abs(pred) >= 50) == (np.abs(actual) >= 50))

    return {
        "n": int(actual.size),
        "mae": float(np.mean(np.abs(err))),                    # 平均绝对误差
        "rmse": float(math.sqrt(np.mean(err * err))),          # 均方根误差
        "mean_error": float(np.mean(err)),                     # 平均误差
        "direction_accuracy": float(direction_accuracy) if np.isfinite(direction_accuracy) else None,
        "threshold_accuracy": float(threshold_accuracy),
        "corr": float(np.corrcoef(actual, pred)[0, 1]) if actual.size > 1 and np.std(pred) > 0 else None,
    }


def build_metrics(df: pd.DataFrame, config: MechanismConfig) -> dict[str, object]:
    """
    构建完整的评估指标集

    按样本范围（全部/测试集）和价格区间分别计算指标

    参数：
        df: 模拟结果
        config: 机制配置
    返回：
        包含各维度评估指标的字典
    """
    metrics: dict[str, object] = {}

    # 按样本范围分组评估
    for sample_name, sample in {
        "all": df,                                          # 全部样本
        "test_after_train": df[df["adjust_date"] > pd.Timestamp(config.train_end)],  # 测试集
    }.items():
        metrics[sample_name] = {
            "gasoline_formula": metric_block(sample, "gasoline", "gasoline_theory_delta"),
            "diesel_formula": metric_block(sample, "diesel", "diesel_theory_delta"),
            "gasoline_calibrated": metric_block(sample, "gasoline", "gasoline_calibrated_delta"),
            "diesel_calibrated": metric_block(sample, "diesel", "diesel_calibrated_delta"),
        }

    # 按价格区间分组评估
    zone_summary = []
    for zone, group in df.groupby("zone", dropna=False):
        row = {"zone": zone, "n": int(len(group))}
        row.update({f"gasoline_{k}": v for k, v in metric_block(group, "gasoline", "gasoline_calibrated_delta").items()})
        row.update({f"diesel_{k}": v for k, v in metric_block(group, "diesel", "diesel_calibrated_delta").items()})
        zone_summary.append(row)
    metrics["zone_summary"] = zone_summary
    return metrics


# ==================== NARDL-like非对称传导检验 ====================

def make_lagged(series: pd.Series, lag: int) -> pd.Series:
    """生成滞后序列"""
    return series.shift(lag)


def run_nardl_like_test(df: pd.DataFrame, product: str) -> dict[str, object]:
    """
    运行NARDL-like（非线性自回归分布滞后）非对称传导检验

    检验思路：
        将油价变化分解为正向变化（oil_pos）和负向变化（oil_neg），
        分别检验油价上涨和下跌对成品油价格的传导是否对称。

    模型设定：
        ΔP_t = ρ·P_{t-1} + φ⁺·POS_{t-1} + φ⁻·NEG_{t-1}
               + Σ(α⁺ᵢ·ΔPOS_{t-i}) + Σ(α⁻ᵢ·ΔNEG_{t-i}) + ε_t

    其中：
        POS = cumsum(max(Δoil, 0))  累积正向变化
        NEG = cumsum(min(Δoil, 0))  累积负向变化

    检验内容：
        - 短期对称性Wald检验：H0: α⁺ = α⁻
        - 长期对称性Wald检验：H0: φ⁺ = φ⁻

    参数：
        df: 模拟结果
        product: 产品名（"gasoline"或"diesel"）
    返回：
        包含回归结果和检验统计量的字典
    """
    price_col = f"{product}_price"
    data = df[["adjust_date", price_col, "ref_ma_usd_per_bbl", f"{product}_actual_delta"]].copy()

    # 计算一阶差分
    data["d_price"] = data[price_col].diff()
    data["d_oil"] = data["ref_ma_usd_per_bbl"].diff()

    # 分解为正向和负向变化
    data["d_oil_pos"] = data["d_oil"].clip(lower=0)  # 正向变化（取max(Δ, 0)）
    data["d_oil_neg"] = data["d_oil"].clip(upper=0)  # 负向变化（取min(Δ, 0)）

    # 累积正向/负向变化
    data["oil_pos_cum"] = data["d_oil_pos"].cumsum()
    data["oil_neg_cum"] = data["d_oil_neg"].cumsum()

    # 构建回归变量
    regressors = pd.DataFrame(
        {
            "price_lag1": make_lagged(data[price_col], 1),           # P_{t-1}
            "oil_pos_cum_lag1": make_lagged(data["oil_pos_cum"], 1), # POS_{t-1}
            "oil_neg_cum_lag1": make_lagged(data["oil_neg_cum"], 1), # NEG_{t-1}
            "d_price_lag1": make_lagged(data["d_price"], 1),         # ΔP_{t-1}
            "d_oil_pos": data["d_oil_pos"],                          # ΔPOS_t
            "d_oil_neg": data["d_oil_neg"],                          # ΔNEG_t
            "d_oil_pos_lag1": make_lagged(data["d_oil_pos"], 1),     # ΔPOS_{t-1}
            "d_oil_neg_lag1": make_lagged(data["d_oil_neg"], 1),     # ΔNEG_{t-1}
        }
    )

    # 合并因变量和自变量，删除缺失值
    y = data["d_price"]
    model_data = pd.concat([y.rename("y"), regressors], axis=1).dropna()

    if len(model_data) < 20:
        return {"n": int(len(model_data)), "error": "not enough observations"}

    # OLS回归（使用HC1异方差稳健标准误）
    x = add_constant(model_data.drop(columns=["y"]))
    model = OLS(model_data["y"], x).fit(cov_type="HC1")

    # === 短期对称性Wald检验 ===
    # H0: ΔPOS的系数之和 = ΔNEG的系数之和（即油价上涨和下跌的短期传导对称）
    short_terms = ["d_oil_pos", "d_oil_pos_lag1", "d_oil_neg", "d_oil_neg_lag1"]
    restriction = np.zeros((1, len(model.params)))
    names = list(model.params.index)
    for name in ["d_oil_pos", "d_oil_pos_lag1"]:
        restriction[0, names.index(name)] = 1.0
    for name in ["d_oil_neg", "d_oil_neg_lag1"]:
        restriction[0, names.index(name)] = -1.0
    short_wald = model.wald_test(restriction, scalar=True)

    # === 长期对称性Wald检验 ===
    # H0: POS的长期系数 = NEG的长期系数
    long_restriction = np.zeros((1, len(model.params)))
    long_restriction[0, names.index("oil_pos_cum_lag1")] = 1.0
    long_restriction[0, names.index("oil_neg_cum_lag1")] = -1.0
    long_wald = model.wald_test(long_restriction, scalar=True)

    # 计算长期传导系数
    rho = model.params.get("price_lag1", np.nan)
    phi_pos = model.params.get("oil_pos_cum_lag1", np.nan)
    phi_neg = model.params.get("oil_neg_cum_lag1", np.nan)
    long_pos = float(-phi_pos / rho) if np.isfinite(rho) and abs(rho) > 1e-12 else None
    long_neg = float(-phi_neg / rho) if np.isfinite(rho) and abs(rho) > 1e-12 else None

    return {
        "n": int(model.nobs),
        "r_squared": float(model.rsquared),
        "long_run_positive": long_pos,             # 正向长期传导系数
        "long_run_negative": long_neg,             # 负向长期传导系数
        "short_wald_pvalue": float(short_wald.pvalue),  # 短期对称性检验p值
        "long_wald_pvalue": float(long_wald.pvalue),    # 长期对称性检验p值
        "params": {k: float(v) for k, v in model.params.items()},
    }


# ==================== 输出函数 ====================

def write_outputs(
    panel: pd.DataFrame,
    simulations: dict[str, pd.DataFrame],
    summaries: dict[str, object],
    config: MechanismConfig,
) -> None:
    """
    将结果写入文件

    输出文件：
        - oil_panel_with_indices.csv: 原油价格面板（含各种指数）
        - mechanism_validation_{name}.csv: 各方案的模拟验证结果
        - summary_metrics.json: 汇总评估指标

    参数：
        panel: 原油价格面板
        simulations: 各方案的模拟结果字典
        summaries: 各方案的评估指标字典
        config: 机制配置
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUTPUT_DIR / "oil_panel_with_indices.csv", index=False, encoding="utf-8-sig")
    for name, df in simulations.items():
        df.to_csv(OUTPUT_DIR / f"mechanism_validation_{name}.csv", index=False, encoding="utf-8-sig")
    with (OUTPUT_DIR / "summary_metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"config": asdict(config), "summaries": summaries}, f, ensure_ascii=False, indent=2)


def print_short_report(summaries: dict[str, object]) -> None:
    """
    打印简要报告

    参数：
        summaries: 各方案的评估指标字典
    """
    print("Task 1 价格机制验证结果")
    for name, summary in summaries.items():
        print(f"\n[{name}]")
        print("PCA权重:", summary["pca_weights"])
        print("校准缩放系数:", summary["calibration_scales"])
        metrics = summary["metrics"]["all"]
        for key, values in metrics.items():
            print(
                f"{key}: n={values.get('n')}, "
                f"MAE={values.get('mae'):.2f}, "
                f"RMSE={values.get('rmse'):.2f}, "
                f"方向准确率={values.get('direction_accuracy')}"
            )
        print("NARDL汽油:", summary["nardl_like"]["gasoline"])
        print("NARDL柴油:", summary["nardl_like"]["diesel"])


# ==================== 命令行参数 ====================

def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="验证中国成品油价格调控机制")
    parser.add_argument("--start-date", default=MechanismConfig.start_date, help="分析起始日期")
    parser.add_argument("--train-end", default=MechanismConfig.train_end, help="训练集截止日期")
    parser.add_argument("--window-size", type=int, default=MechanismConfig.window_size, help="调价窗口大小")
    parser.add_argument("--pricing-lag-days", type=int, default=MechanismConfig.pricing_lag_days, help="定价滞后天数")
    return parser.parse_args()


# ==================== 主函数 ====================

def main() -> None:
    """
    主函数：执行完整的验证流程

    流程：
        1. 解析参数，构建配置
        2. 读取国际油价面板数据
        3. 读取国内调价记录
        4. 分别用两种油价指数（固定篮子、PCA+卡尔曼）模拟定价机制
        5. 校准预测值
        6. 计算评估指标
        7. 运行NARDL-like检验
        8. 输出结果
    """
    args = parse_args()
    config = MechanismConfig(
        start_date=args.start_date,
        train_end=args.train_end,
        window_size=args.window_size,
        pricing_lag_days=args.pricing_lag_days,
    )

    # 读取数据
    panel = read_oil_panel()
    domestic = read_domestic_adjustments(config.start_date)

    # 分别用两种油价指数进行模拟
    simulations: dict[str, pd.DataFrame] = {}
    summaries: dict[str, object] = {}

    for name, price_col in {
        "fixed_basket": "basket_usd",           # 方案1：固定加权篮子（0.4Brent+0.1WTI+0.5Dubai）
        "pca_kalman": "kalman_index_usd",       # 方案2：PCA+卡尔曼平滑指数
    }.items():
        # 模拟定价机制
        simulated = simulate_mechanism(domestic, panel, price_col, config)
        # 校准预测值
        simulated, scales = add_calibrated_predictions(simulated, config)

        simulations[name] = simulated
        summaries[name] = {
            "oil_index_column": price_col,
            "pca_weights": panel.attrs.get("pca_weights", {}),
            "calibration_scales": scales,
            "metrics": build_metrics(simulated, config),
            "nardl_like": {
                "gasoline": run_nardl_like_test(simulated, "gasoline"),
                "diesel": run_nardl_like_test(simulated, "diesel"),
            },
        }

    # 输出结果
    write_outputs(panel, simulations, summaries, config)
    print_short_report(summaries)
    print(f"\n输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
