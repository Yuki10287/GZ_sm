"""
任务一进阶：多源动态因子状态空间模型
=============================================
核心功能：
    在基础定价机制验证的基础上，引入更多数据源（海关进口数据、CPI/PPI/PMI/GDP），
    使用动态因子模型（Dynamic Factor Model）提取潜在成本压力因子，
    再用Ridge回归预测成品油调价幅度。

建模思路：
    1. 数据层面：整合油价、汇率、进口成本、宏观指标等多源数据
    2. 因子层面：用状态空间动态因子模型提取一个潜在"成本压力因子"
    3. 预测层面：将因子及其他特征输入Ridge回归，预测调价幅度
    4. 消融实验：对比不同变量组合的预测效果

相比基础模型的改进：
    - 基础模型仅使用油价和汇率
    - 本模型额外纳入进口成本、CPI/PPI/PMI/GDP等宏观变量
    - 使用动态因子模型捕捉多变量间的共同趋势
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor

import task1_price_mechanism as base  # 引入基础模块


# ==================== 路径和配置 ====================

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs" / "task1_multisource_statespace"


@dataclass(frozen=True)
class MultiSourceConfig:
    """
    多源模型配置参数

    属性：
        start_date: 分析起始日期
        train_end: 训练集截止日期
        window_size: 滑动窗口大小
        pricing_lag_days: 定价滞后天数
        factor_order: 动态因子模型的自回归阶数
    """
    start_date: str = "2016-01-01"
    train_end: str = "2022-12-31"
    window_size: int = 10
    pricing_lag_days: int = 1
    factor_order: int = 1  # 因子的AR(1)自回归阶数


# ==================== 观测变量组合定义 ====================
# 不同实验方案使用的观测变量集合

OBS_SETS = {
    # 方案A：仅油价（基准）
    "A_oil_only": [
        "log_brent_ma10",      # Brent油价10日均值对数
        "log_wti_ma10",        # WTI油价10日均值对数
        "log_dubai_ma10",      # Dubai油价10日均值对数
    ],
    # 方案B：油价 + 汇率
    "B_oil_fx": [
        "log_brent_ma10",
        "log_wti_ma10",
        "log_dubai_ma10",
        "log_usd_cny_ma10",    # 美元兑人民币汇率10日均值对数
    ],
    # 方案C：油价 + 汇率 + 进口成本
    "C_cost_factor": [
        "log_brent_ma10",
        "log_wti_ma10",
        "log_dubai_ma10",
        "log_usd_cny_ma10",
        "log_import_cny_per_ton",  # 进口原油每吨人民币价格对数
        "log_import_tons",         # 进口原油数量（吨）对数
    ],
    # 方案D：全部变量（油价 + 汇率 + 进口 + 宏观经济）
    "D_full_factor": [
        "log_brent_ma10",
        "log_wti_ma10",
        "log_dubai_ma10",
        "log_usd_cny_ma10",
        "log_import_cny_per_ton",
        "log_import_tons",
        "cpi_yoy",              # CPI同比增速
        "ppi_yoy",              # PPI同比增速
        "pmi_manufacturing",    # 制造业PMI
        "gdp_yoy",              # GDP同比增速
    ],
}

# 宏观政策变量（用于Ridge回归，不纳入动态因子）
POLICY_VARS = [
    "cpi_yoy",                # CPI同比
    "ppi_yoy",                # PPI同比
    "cpi_ppi_gap",            # CPI-PPI剪刀差
    "pmi_manufacturing",      # 制造业PMI
    "pmi_nonmanufacturing",   # 非制造业PMI
    "gdp_yoy",                # GDP同比
]


# ==================== 日期工具函数 ====================

def month_end_from_ym(series: pd.Series) -> pd.Series:
    """
    将年月字符串转换为月末日期

    参数：
        series: 包含年月信息的Series（如"2024-01"）
    返回：
        月末日期Series
    """
    return pd.to_datetime(series.astype("string") + "-01", errors="coerce") + pd.offsets.MonthEnd(0)


def make_monthly_available(df: pd.DataFrame) -> pd.DataFrame:
    """
    设置月度数据的可用日期

    注意：在不知道具体发布日期的情况下，保守假设月度数据
    从下月1日起才可用（避免前视偏差）

    参数：
        df: 包含date列的月度数据
    返回：
        将date列改为available_date（下月1日）的DataFrame
    """
    out = df.copy()
    out["available_date"] = out["date"] + pd.offsets.Day(1)
    return out.drop(columns=["date"]).sort_values("available_date")


def make_quarterly_available(df: pd.DataFrame) -> pd.DataFrame:
    """
    设置季度数据的可用日期

    注意：GDP等季度数据，保守假设在季度结束后才可用

    参数：
        df: 包含date列的季度数据
    返回：
        将date列改为available_date的DataFrame
    """
    out = df.copy()
    out["available_date"] = out["date"] + pd.offsets.Day(1)
    return out.drop(columns=["date"]).sort_values("available_date")


# ==================== 多源数据读取 ====================

def read_customs_monthly() -> pd.DataFrame:
    """
    读取海关月度进口数据

    数据来源：main.py生成的合并总表
    包含指标：进口数量（千克/吨）、金额（人民币/美元）、单价

    返回：
        月度进口数据DataFrame，date列为月末日期
    """
    df = pd.read_csv(ROOT / "海关进出口数量数据" / "进口原油数量和金额_合并总表.csv", encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["数据年月"].astype(str) + "01", format="%Y%m%d", errors="coerce") + pd.offsets.MonthEnd(0)

    # 清洗数值列
    numeric_cols = ["第一数量", "金额_人民币", "金额_美元", "每吨人民币", "每吨美元"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 选取并重命名列
    out = df[["date", "第一数量", "金额_人民币", "金额_美元", "每吨人民币", "每吨美元"]].copy()
    out = out.rename(
        columns={
            "第一数量": "import_kg",             # 进口量（千克）
            "金额_人民币": "import_amount_cny",   # 进口金额（人民币）
            "金额_美元": "import_amount_usd",     # 进口金额（美元）
            "每吨人民币": "import_cny_per_ton",   # 每吨单价（人民币）
            "每吨美元": "import_usd_per_ton",     # 每吨单价（美元）
        }
    )
    out["import_tons"] = out["import_kg"] / 1000.0  # 千克转吨
    return make_monthly_available(out.sort_values("date"))


def read_macro_monthly() -> pd.DataFrame:
    """
    读取月度宏观经济数据

    数据来源：
        - CPI：2026-2008_cpi.xlsx
        - PPI：2026-2006_ppi.xlsx
        - PMI：2008-2026_pmi.xlsx

    返回：
        合并后的月度宏观数据DataFrame
    """
    # 读取CPI数据
    cpi = pd.read_excel(ROOT / "国内的一些数据" / "2026-2008_cpi.xlsx")
    cpi["date"] = month_end_from_ym(cpi["月份"])
    cpi = cpi.rename(
        columns={
            "全国_当月": "cpi_index",        # CPI指数
            "全国_同比增长": "cpi_yoy",      # CPI同比
            "全国_环比增长": "cpi_mom",      # CPI环比
        }
    )[["date", "cpi_index", "cpi_yoy", "cpi_mom"]]

    # 读取PPI数据
    ppi = pd.read_excel(ROOT / "国内的一些数据" / "2026-2006_ppi.xlsx")
    ppi["date"] = month_end_from_ym(ppi["月份"])
    ppi = ppi.rename(columns={"PPI_当月": "ppi_index", "PPI_当月同比增长": "ppi_yoy"})[
        ["date", "ppi_index", "ppi_yoy"]
    ]

    # 读取PMI数据
    pmi = pd.read_excel(ROOT / "国内的一些数据" / "2008-2026_pmi.xlsx")
    pmi["date"] = month_end_from_ym(pmi["月份"])
    pmi = pmi.rename(
        columns={
            "制造业_指数": "pmi_manufacturing",         # 制造业PMI
            "非制造业_指数": "pmi_nonmanufacturing",    # 非制造业PMI
            "制造业_同比增长": "pmi_manufacturing_yoy", # 制造业PMI同比
        }
    )[["date", "pmi_manufacturing", "pmi_nonmanufacturing", "pmi_manufacturing_yoy"]]

    # 合并所有宏观数据
    macro = cpi.merge(ppi, on="date", how="outer").merge(pmi, on="date", how="outer")
    return make_monthly_available(macro.sort_values("date"))


def read_gdp_quarterly() -> pd.DataFrame:
    """
    读取季度GDP数据

    数据来源：2006-2026_gdp.xlsx

    返回：
        季度GDP数据DataFrame
    """
    df = pd.read_excel(ROOT / "国内的一些数据" / "2006-2026_gdp.xlsx")

    # 解析季度信息（如"2024Q1"→年=2024, 月=3）
    year = df["标准季度"].astype(str).str.slice(0, 4).astype(int)
    quarter = df["标准季度"].astype(str).str.extract(r"Q(\d)").iloc[:, 0].astype(int)
    month = quarter * 3  # 季度末月份
    df["date"] = pd.to_datetime(dict(year=year, month=month, day=1)) + pd.offsets.MonthEnd(0)

    df = df.rename(columns={"GDP_当季绝对值": "gdp_current", "GDP_累计同比": "gdp_yoy"})
    return make_quarterly_available(df[["date", "gdp_current", "gdp_yoy"]].sort_values("date"))


# ==================== 数据合并 ====================

def asof_merge_available(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """
    按可用日期进行asof合并（时间点匹配）

    用途：将调价日与最近可用的宏观数据匹配
    逻辑：对每个调价日，找到不晚于该日的最新可用数据

    参数：
        left: 左表（调价数据，含adjust_date列）
        right: 右表（宏观数据，含available_date列）
    返回：
        合并后的DataFrame
    """
    return pd.merge_asof(
        left.sort_values("adjust_date"),
        right.sort_values("available_date"),
        left_on="adjust_date",
        right_on="available_date",
        direction="backward",  # 取最近的过去值
    ).drop(columns=["available_date"], errors="ignore")


def build_window_dataset(config: MultiSourceConfig) -> pd.DataFrame:
    """
    构建多源窗口数据集

    处理流程：
        1. 调用基础模块的定价机制模拟
        2. 计算每日油价特征（10日移动平均的对数）
        3. 合并海关月度进口数据
        4. 合并月度宏观数据（CPI/PPI/PMI）
        5. 合并季度GDP数据
        6. 计算衍生变量（对数、变化率、CPI-PPI剪刀差）

    参数：
        config: 多源模型配置
    返回：
        包含所有特征的完整数据集
    """
    # 构建基础机制配置
    mechanism_config = base.MechanismConfig(
        start_date=config.start_date,
        train_end=config.train_end,
        window_size=config.window_size,
        pricing_lag_days=config.pricing_lag_days,
    )

    # 读取数据并运行基础模拟
    panel = base.read_oil_panel()
    domestic = base.read_domestic_adjustments(config.start_date)
    mechanism = base.simulate_mechanism(domestic, panel, "kalman_index_usd", mechanism_config)

    # 计算每日油价特征（窗口内均值）
    daily_features = []
    for _, row in mechanism.iterrows():
        pricing_end = pd.Timestamp(row["pricing_end"])
        win = base.trailing_window(panel, pricing_end, "kalman_index_usd", config.window_size)
        raw_win = panel[panel["date"].isin(win["date"])]
        record = {"adjust_date": row["adjust_date"]}
        # 计算各油种和汇率的窗口均值
        for col in ["brent", "wti", "dubai", "usd_cny"]:
            record[f"{col}_ma10"] = float(raw_win[col].mean())
        daily_features.append(record)

    # 合并特征
    features = mechanism.merge(pd.DataFrame(daily_features), on="adjust_date", how="left")

    # 依次合并多源数据（使用asof合并避免前视偏差）
    features = asof_merge_available(features, read_customs_monthly())   # 海关数据
    features = asof_merge_available(features, read_macro_monthly())     # 宏观数据
    features = asof_merge_available(features, read_gdp_quarterly())     # GDP数据
    features = features.sort_values("adjust_date").reset_index(drop=True)

    # 计算对数变量（用于动态因子模型，对数化使变量量纲可比）
    for col in [
        "brent_ma10", "wti_ma10", "dubai_ma10", "usd_cny_ma10",
        "import_cny_per_ton", "import_usd_per_ton", "import_tons",
        "cpi_index", "ppi_index", "pmi_manufacturing", "pmi_nonmanufacturing", "gdp_current",
    ]:
        if col in features:
            values = pd.to_numeric(features[col], errors="coerce").clip(lower=1e-9)
            features[f"log_{col}"] = np.log(values)

    # 计算衍生变量
    features["import_cny_per_ton_change"] = features["import_cny_per_ton"].pct_change()  # 进口单价变化率
    features["import_usd_per_ton_change"] = features["import_usd_per_ton"].pct_change()  # 进口单价变化率（美元）
    features["cpi_ppi_gap"] = features["cpi_yoy"] - features["ppi_yoy"]                  # CPI-PPI剪刀差

    return features


# ==================== 动态因子模型 ====================

def extract_loadings(result, obs_cols: list[str], sign: float) -> pd.DataFrame:
    """
    提取动态因子模型的因子载荷

    因子载荷表示各观测变量与潜在因子的相关程度，
    载荷绝对值越大，说明该变量对因子的贡献越大。

    参数：
        result: 动态因子模型拟合结果
        obs_cols: 观测变量名列表
        sign: 符号修正系数（确保因子方向一致）
    返回：
        按绝对载荷排序的DataFrame
    """
    rows = []
    params = dict(zip(result.param_names, np.asarray(result.params, dtype=float)))
    for col in obs_cols:
        value = params.get(f"loading.f1.{col}")
        if value is not None:
            rows.append({"variable": col, "loading": float(value * sign), "abs_loading": float(abs(value))})
    return pd.DataFrame(rows).sort_values("abs_loading", ascending=False)


def fit_dynamic_factor(
    features: pd.DataFrame,
    config: MultiSourceConfig,
    obs_cols: list[str],
    variant_name: str,
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    """
    拟合单因子动态因子模型

    模型设定：
        观测方程：y_t = H × f_t + ε_t
        状态方程：f_t = A × f_{t-1} + η_t

    其中：
        y_t: 观测变量向量（标准化后的多源数据）
        f_t: 潜在因子（一维，代表"成本压力"）
        H: 因子载荷矩阵
        A: 因子自回归系数

    参数：
        features: 完整特征数据集
        config: 模型配置
        obs_cols: 观测变量列表
        variant_name: 实验方案名称
    返回：
        (添加了因子列的DataFrame, 拟合信息字典, 因子载荷DataFrame)
    """
    # 提取观测变量并处理异常值
    obs = features[obs_cols].copy()
    obs = obs.replace([np.inf, -np.inf], np.nan).ffill().bfill()

    # 标准化观测变量
    standardized = pd.DataFrame(
        StandardScaler().fit_transform(obs),
        index=pd.DatetimeIndex(features["adjust_date"]),
        columns=obs_cols,
    )

    # 拟合动态因子模型
    model = DynamicFactor(
        standardized,
        k_factors=1,              # 一个潜在因子
        factor_order=config.factor_order,  # 因子AR阶数
        error_cov_type="diagonal",         # 误差协方差设为对角阵
    )
    result = model.fit(method="lbfgs", maxiter=1000, disp=False)  # L-BFGS-B优化

    # 提取平滑后的因子
    factor = np.asarray(result.factors.smoothed[0], dtype=float)

    # 符号修正：确保因子与第一个观测变量正相关
    sign = 1.0
    reference = standardized[obs_cols[0]].to_numpy(dtype=float)
    if np.corrcoef(factor, reference)[0, 1] < 0:
        sign = -1.0
        factor = -factor

    # 将因子添加到特征数据中
    out = features.copy()
    out["factor_1"] = factor                           # 因子水平值
    out["factor_1_diff"] = out["factor_1"].diff().fillna(0.0)  # 因子变化量

    # 提取因子载荷
    loadings = extract_loadings(result, obs_cols, sign)

    # 记录拟合信息
    fit_info = {
        "variant": variant_name,
        "obs_cols": obs_cols,
        "llf": float(result.llf),           # 对数似然
        "aic": float(result.aic),           # AIC信息准则
        "bic": float(result.bic),           # BIC信息准则
        "converged": bool(result.mle_retvals.get("converged", False)),  # 是否收敛
        "iterations": int(result.mle_retvals.get("iterations", -1)),    # 迭代次数
    }
    return out, fit_info, loadings


# ==================== Ridge回归预测 ====================

def train_predict_product(
    df: pd.DataFrame,
    product: str,
    config: MultiSourceConfig,
    include_policy_vars: bool,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """
    训练Ridge回归模型预测单个产品的调价幅度

    特征包括：
        - 基础机制的理论调价值
        - 油价变化率
        - 动态因子及其变化量
        - 进口成本变化率
        - 可选：宏观政策变量（CPI/PPI/PMI/GDP）

    参数：
        df: 特征数据集
        product: 产品名（"gasoline"或"diesel"）
        config: 模型配置
        include_policy_vars: 是否包含宏观政策变量
    返回：
        (添加了预测列的DataFrame, 模型信息字典)
    """
    result = df.copy()
    target = f"{product}_actual_delta"      # 目标变量
    base_col = f"{product}_theory_delta"    # 基础机制理论值

    # 定义特征列
    feature_cols = [
        base_col,                           # 基础机制理论调价
        "oil_change_rate",                  # 油价变化率
        "factor_1",                         # 动态因子水平值
        "factor_1_diff",                    # 动态因子变化量
        "import_cny_per_ton_change",        # 进口单价变化率（人民币）
        "import_usd_per_ton_change",        # 进口单价变化率（美元）
    ]
    if include_policy_vars:
        feature_cols += POLICY_VARS  # 添加宏观政策变量

    # 过滤存在的列并处理异常值
    feature_cols = [col for col in feature_cols if col in result.columns]
    result[feature_cols] = result[feature_cols].replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)

    # 训练Ridge回归（带交叉验证选择正则化参数）
    train_mask = result["adjust_date"] <= pd.Timestamp(config.train_end)
    model = Pipeline(
        [
            ("scale", StandardScaler()),                            # 标准化
            ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 25))),      # Ridge回归，25个候选alpha
        ]
    )
    model.fit(result.loc[train_mask, feature_cols], result.loc[train_mask, target])

    # 预测
    pred_col = f"{product}_multisource_delta"
    result[pred_col] = model.predict(result[feature_cols])

    # 提取模型信息
    ridge = model.named_steps["ridge"]
    return result, {
        "alpha": float(ridge.alpha_),       # 最优正则化参数
        "features": feature_cols,           # 使用的特征
        "coef": dict(zip(feature_cols, ridge.coef_.astype(float))),  # 回归系数
    }


# ==================== 评估函数 ====================

def metric_block(df: pd.DataFrame, product: str, pred_col: str) -> dict[str, float | int | None]:
    """
    计算评估指标（在基础模块的指标上增加WMAPE）

    WMAPE（加权平均绝对百分比误差）：Σ|实际-预测| / Σ|实际|

    参数：
        df: 模拟结果
        product: 产品名
        pred_col: 预测列名
    返回：
        评估指标字典
    """
    block = base.metric_block(df, product, pred_col)
    actual = df[f"{product}_actual_delta"]
    pred = df[pred_col]
    denom = actual.abs().sum()
    block["wmape"] = float((pred - actual).abs().sum() / denom) if denom else None
    return block


def evaluate_variant(
    features: pd.DataFrame,
    config: MultiSourceConfig,
    variant_name: str,
    obs_cols: list[str],
    include_policy_vars: bool,
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    """
    评估单个实验方案

    流程：
        1. 拟合动态因子模型
        2. 训练Ridge回归预测汽油和柴油
        3. 在不同样本范围计算评估指标

    参数：
        features: 特征数据集
        config: 模型配置
        variant_name: 方案名称
        obs_cols: 观测变量列表
        include_policy_vars: 是否包含宏观政策变量
    返回：
        (结果DataFrame, 评估摘要, 因子载荷)
    """
    # 拟合动态因子
    state_df, fit_info, loadings = fit_dynamic_factor(features, config, obs_cols, variant_name)

    # 训练预测模型
    state_df, gas_info = train_predict_product(state_df, "gasoline", config, include_policy_vars)
    state_df, diesel_info = train_predict_product(state_df, "diesel", config, include_policy_vars)

    # 定义评估样本范围
    samples = {
        "all": state_df,                                                    # 全部样本
        "test_after_train": state_df[state_df["adjust_date"] > pd.Timestamp(config.train_end)],  # 测试集
        "normal_exclude_explicit": state_df[                               # 正常区间（排除特殊日期）
            (state_df["zone"] == "normal")
            & (~state_df["adjust_date"].isin(pd.to_datetime(["2026-03-24", "2026-04-08"])))
        ],
    }

    # 计算各范围的评估指标
    metrics = {}
    for sample_name, sample in samples.items():
        metrics[sample_name] = {
            "gasoline_mechanism": metric_block(sample, "gasoline", "gasoline_theory_delta"),
            "diesel_mechanism": metric_block(sample, "diesel", "diesel_theory_delta"),
            "gasoline_multisource": metric_block(sample, "gasoline", "gasoline_multisource_delta"),
            "diesel_multisource": metric_block(sample, "diesel", "diesel_multisource_delta"),
        }

    summary = {
        "fit": fit_info,
        "include_policy_vars_in_ridge": include_policy_vars,
        "transmission_models": {"gasoline": gas_info, "diesel": diesel_info},
        "metrics": metrics,
    }
    return state_df, summary, loadings


# ==================== 可视化 ====================

def plot_factor(df: pd.DataFrame, variant_name: str, output_dir: Path) -> None:
    """
    绘制动态因子路径图

    图中包含：
        - 因子时间序列曲线
        - 三个特殊时期的阴影标注（2020疫情冲击、2022高油价、2026冲突）

    参数：
        df: 包含factor_1列的DataFrame
        variant_name: 方案名称（用于标题和文件名）
        output_dir: 输出目录
    """
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(pd.to_datetime(df["adjust_date"]), df["factor_1"], color="#1f77b4", linewidth=1.8)

    # 标注特殊时期
    periods = {
        "2020疫情冲击": ("2020-02-01", "2020-06-30"),
        "2022高油价": ("2022-02-01", "2022-10-31"),
        "2026冲突": ("2026-03-01", "2026-05-31"),
    }
    colors = ["#f4a261", "#e76f51", "#2a9d8f"]
    for (label, (start, end)), color in zip(periods.items(), colors):
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.16, color=color, label=label)

    ax.set_title(f"潜在成本压力因子 - {variant_name}")
    ax.set_xlabel("调价日期")
    ax.set_ylabel("平滑因子值")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / f"factor_path_{variant_name}.png", dpi=180)
    plt.close(fig)


# ==================== 命令行参数 ====================

def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="多源动态因子状态空间实验")
    parser.add_argument("--start-date", default=MultiSourceConfig.start_date, help="分析起始日期")
    parser.add_argument("--train-end", default=MultiSourceConfig.train_end, help="训练集截止日期")
    parser.add_argument("--window-size", type=int, default=MultiSourceConfig.window_size, help="窗口大小")
    parser.add_argument("--pricing-lag-days", type=int, default=MultiSourceConfig.pricing_lag_days, help="定价滞后天数")
    return parser.parse_args()


# ==================== 主函数 ====================

def main() -> None:
    """
    主函数：运行所有消融实验

    实验方案：
        A_oil_only:                    仅油价因子
        B_oil_fx:                      油价 + 汇率因子
        C_cost_factor:                 油价 + 汇率 + 进口成本因子
        D_cost_factor_policy_ridge:    成本因子 + 宏观政策变量（主方案）
        E_full_variable_factor:        全变量因子

    输出：
        - 各方案的验证结果CSV
        - 因子载荷CSV
        - 因子路径图PNG
        - 消融实验指标汇总
        - summary_metrics.json
    """
    args = parse_args()
    config = MultiSourceConfig(
        start_date=args.start_date,
        train_end=args.train_end,
        window_size=args.window_size,
        pricing_lag_days=args.pricing_lag_days,
    )

    # 构建数据集
    features = build_window_dataset(config)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 定义实验方案
    variants = {
        "A_oil_only": (OBS_SETS["A_oil_only"], False),                          # 仅油价
        "B_oil_fx": (OBS_SETS["B_oil_fx"], False),                              # 油价+汇率
        "C_cost_factor": (OBS_SETS["C_cost_factor"], False),                    # 成本因子
        "D_cost_factor_policy_ridge": (OBS_SETS["C_cost_factor"], True),        # 成本因子+政策变量
        "E_full_variable_factor": (OBS_SETS["D_full_factor"], False),           # 全变量
    }

    # 运行所有实验
    summaries = {}
    ablation_rows = []
    main_validation = None

    for variant_name, (obs_cols, include_policy_vars) in variants.items():
        state_df, summary, loadings = evaluate_variant(
            features, config, variant_name, obs_cols, include_policy_vars,
        )
        summaries[variant_name] = summary

        # 保存结果
        state_df.to_csv(OUTPUT_DIR / f"validation_{variant_name}.csv", index=False, encoding="utf-8-sig")
        loadings.to_csv(OUTPUT_DIR / f"loadings_{variant_name}.csv", index=False, encoding="utf-8-sig")
        plot_factor(state_df, variant_name, OUTPUT_DIR)

        # 保存主方案（D方案）的完整验证结果
        if variant_name == "D_cost_factor_policy_ridge":
            main_validation = state_df
            state_df.to_csv(OUTPUT_DIR / "multisource_state_space_validation.csv", index=False, encoding="utf-8-sig")
            loadings.to_csv(OUTPUT_DIR / "loadings_cost_factor.csv", index=False, encoding="utf-8-sig")

        # 收集消融实验结果
        test = summary["metrics"]["test_after_train"]
        ablation_rows.append(
            {
                "variant": variant_name,
                "factor_variables": ", ".join(obs_cols),
                "policy_vars_in_ridge": include_policy_vars,
                "converged": summary["fit"]["converged"],
                "gasoline_test_mae": test["gasoline_multisource"]["mae"],
                "gasoline_test_wmape": test["gasoline_multisource"]["wmape"],
                "diesel_test_mae": test["diesel_multisource"]["mae"],
                "diesel_test_wmape": test["diesel_multisource"]["wmape"],
            }
        )

    # 保存消融实验汇总
    ablation = pd.DataFrame(ablation_rows)
    ablation.to_csv(OUTPUT_DIR / "ablation_test_metrics.csv", index=False, encoding="utf-8-sig")

    # 保存完整汇总JSON
    summary = {"config": asdict(config), "variants": summaries, "ablation_test_metrics": ablation_rows}
    with (OUTPUT_DIR / "summary_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 打印结果
    print("多源动态因子状态空间实验结果")
    print(ablation.to_string(index=False))

    # 找出最佳方案（测试集MAE最小）
    best_idx = (
        (ablation["gasoline_test_mae"] + ablation["diesel_test_mae"])
        .astype(float)
        .idxmin()
    )
    best_variant = str(ablation.loc[best_idx, "variant"])
    best = summaries[best_variant]["metrics"]
    print(f"\n最佳测试MAE方案: {best_variant}")
    print("汽油全部样本:", best["all"]["gasoline_multisource"])
    print("柴油全部样本:", best["all"]["diesel_multisource"])
    print("成本因子载荷已保存至 loadings_cost_factor.csv")
    print(f"\n输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
