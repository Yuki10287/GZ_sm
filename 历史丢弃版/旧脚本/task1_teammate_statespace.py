"""
任务一变体：队友的双因子状态空间模型
=============================================
核心功能：
    使用双因子状态空间模型拟合国际原油价格，提取潜在的价格趋势因子，
    然后用队友的定价公式进行调价验证。

建模思路：
    1. 模型设定：双因子模型（价格水平 + 价格趋势）
        - 观测方程：log(P_i,t) = h_i × x_t + ε_i,t
        - 状态方程：x_t = μ + α·δ_t + x_{t-1} + η_{1,t}
                   δ_t = κ·α + (1-κ)·δ_{t-1} + η_{2,t}
    2. 其中x_t为价格水平因子，δ_t为价格趋势（漂移）因子
    3. 使用卡尔曼滤波进行状态估计
    4. 用MLE（最大似然估计）估计模型参数
    5. 将估计的价格指数代入队友定价公式验证

与task1_multisource_statespace的区别：
    - 本模型：手工实现的双因子状态空间，用MLE拟合
    - 多源模型：使用statsmodels的DynamicFactor，单因子，纳入更多数据源
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import task1_price_mechanism as base      # 基础模块
import task1_teammate_model as teammate   # 队友公式模块


# ==================== 路径配置 ====================

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "task1_teammate_statespace"


# ==================== 数据类定义 ====================

@dataclass(frozen=True)
class StateSpaceFit:
    """
    状态空间模型拟合结果

    属性：
        params: 模型参数字典
        neg_loglike: 负对数似然值（越小越好）
        success: 优化是否成功
        message: 优化器返回的消息
        iterations: 迭代次数
    """
    params: dict[str, float]
    neg_loglike: float
    success: bool
    message: str
    iterations: int


# ==================== 数据预处理 ====================

def prepare_log_observations(panel: pd.DataFrame, start_date: str) -> pd.DataFrame:
    """
    准备对数观测数据

    处理流程：
        1. 筛选起始日期之后的数据
        2. 处理非正值（取对数前必须为正）
        3. 对三种油种取自然对数

    参数：
        panel: 油价面板数据
        start_date: 起始日期
    返回：
        包含brent_log/wti_log/dubai_log的DataFrame
    """
    df = panel[panel["date"] >= pd.Timestamp(start_date)].copy()

    # 提取价格并处理非正值
    prices = df[["brent", "wti", "dubai"]].copy()
    prices = prices.mask(prices <= 0)
    prices = prices.interpolate(limit_direction="both").ffill().bfill()

    # 取对数
    df[["brent_log", "wti_log", "dubai_log"]] = np.log(prices)
    return df.dropna(subset=["brent_log", "wti_log", "dubai_log"]).reset_index(drop=True)


# ==================== 状态空间模型核心 ====================

def unpack(theta: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """
    将参数向量解包为状态空间矩阵

    模型参数（theta的各分量）：
        theta[0] = μ (mu): 价格水平的长期漂移
        theta[1] = α (alpha): 趋势因子的均值
        theta[2] = log(κ): 趋势因子的均值回复速度（取exp确保正值）
        theta[3] = log(σ₁): 价格水平的波动率
        theta[4] = log(σ₂): 趋势因子的波动率
        theta[5] = atanh(ρ): 两个状态噪声的相关系数（取tanh确保在(-1,1)）
        theta[6:9] = h: 三个油种的因子载荷
        theta[9:12] = log(观测噪声标准差)

    状态空间表示：
        状态转移：[x_{t+1}, δ_{t+1}]' = A·[x_t, δ_t]' + c + η_t
        观测方程：[log_Brent, log_WTI, log_Dubai]' = H·[x_t, δ_t]' + ε_t

    参数：
        theta: 参数向量（12维）
        dt: 时间步长（1/252表示日频数据）
    返回：
        (状态转移矩阵A, 偏移向量c, 过程噪声协方差Q, 观测矩阵H, 观测噪声协方差R, 参数名字典)
    """
    # 解包参数
    mu = theta[0]           # 价格水平漂移
    alpha = theta[1]        # 趋势均值
    kappa = np.exp(theta[2])  # 均值回复速度（>0）
    sigma1 = np.exp(theta[3])  # 水平波动率（>0）
    sigma2 = np.exp(theta[4])  # 趋势波动率（>0）
    rho = np.tanh(theta[5])    # 噪声相关系数（-1到1）
    h = theta[6:9]             # 因子载荷（3个油种）
    obs_sigma = np.exp(theta[9:12])  # 观测噪声标准差（>0）

    # 构建状态转移矩阵 A
    # [x_{t+1}]   [1    -dt ] [x_t]   [(μ - 0.5σ₁²)·dt]
    # [δ_{t+1}] = [0  1-κ·dt] [δ_t] + [  κ·α·dt       ]
    a = np.array([[1.0, -dt], [0.0, 1.0 - kappa * dt]], dtype=float)

    # 偏移向量 c
    c = np.array([(mu - 0.5 * sigma1 * sigma1) * dt, kappa * alpha * dt], dtype=float)

    # 过程噪声协方差 Q
    # Q = [σ₁²·dt        ρ·σ₁·σ₂·dt]
    #     [ρ·σ₁·σ₂·dt    σ₂²·dt     ]
    q = np.array(
        [
            [sigma1 * sigma1 * dt, rho * sigma1 * sigma2 * dt],
            [rho * sigma1 * sigma2 * dt, sigma2 * sigma2 * dt],
        ],
        dtype=float,
    )

    # 观测矩阵 H（3×2）
    # [log_Brent]   [1  h₁] [x_t]
    # [log_WTI  ] = [1  h₂] [δ_t]
    # [log_Dubai]   [1  h₃]
    hmat = np.column_stack([np.ones(3), h])

    # 观测噪声协方差 R（对角阵）
    r = np.diag(obs_sigma * obs_sigma)

    # 参数名字典（用于结果展示）
    named = {
        "mu": float(mu),
        "alpha": float(alpha),
        "kappa": float(kappa),
        "sigma1": float(sigma1),
        "sigma2": float(sigma2),
        "rho": float(rho),
        "h_brent": float(h[0]),
        "h_wti": float(h[1]),
        "h_dubai": float(h[2]),
        "obs_sigma_brent": float(obs_sigma[0]),
        "obs_sigma_wti": float(obs_sigma[1]),
        "obs_sigma_dubai": float(obs_sigma[2]),
    }
    return a, c, q, hmat, r, named


def kalman_filter(
    y: np.ndarray,
    theta: np.ndarray,
    dt: float,
    return_states: bool = False,
) -> tuple[float, np.ndarray | None]:
    """
    卡尔曼滤波器

    用途：
        1. 计算给定参数下的负对数似然（用于MLE优化）
        2. 可选返回滤波后的状态估计（用于最终拟合）

    算法步骤（逐时刻）：
        预测步：x_{t|t-1} = A·x_{t-1|t-1} + c
               P_{t|t-1} = A·P_{t-1|t-1}·A' + Q
        更新步：K_t = P_{t|t-1}·H'·(H·P_{t|t-1}·H' + R)⁻¹
               x_{t|t} = x_{t|t-1} + K_t·(y_t - H·x_{t|t-1})
               P_{t|t} = (I - K_t·H)·P_{t|t-1}·(I - K_t·H)' + K_t·R·K_t'

    参数：
        y: 观测矩阵（T×3，三油种的对数价格）
        theta: 参数向量
        dt: 时间步长
        return_states: 是否返回滤波后的状态序列
    返回：
        (负对数似然值, 状态序列或None)
    """
    a, c, q, hmat, r, _ = unpack(theta, dt)
    n = y.shape[0]

    # 初始化状态：x₀设为第一个观测的均值，δ₀设为0
    state = np.array([float(np.nanmean(y[:, 0])), 0.0], dtype=float)
    cov = np.diag([0.25, 0.25])  # 初始协方差
    identity = np.eye(2)
    loglike = 0.0
    filtered = np.zeros((n, 2), dtype=float)

    for t in range(n):
        # === 预测步 ===
        if t > 0:
            state = a @ state + c          # 状态预测
            cov = a @ cov @ a.T + q        # 协方差预测
            cov = (cov + cov.T) / 2.0      # 确保对称

        # === 更新步 ===
        innovation = y[t] - hmat @ state                    # 新息（观测残差）
        innovation_cov = hmat @ cov @ hmat.T + r            # 新息协方差
        innovation_cov = (innovation_cov + innovation_cov.T) / 2.0  # 确保对称

        # 计算对数似然增量
        try:
            sign, logdet = np.linalg.slogdet(innovation_cov)
            if sign <= 0:
                return 1e12, None  # 协方差矩阵非正定，返回极大惩罚值
            solved = np.linalg.solve(innovation_cov, innovation)
        except np.linalg.LinAlgError:
            return 1e12, None

        # 对数似然：-0.5·(d·ln(2π) + ln|S| + v'·S⁻¹·v)
        loglike += -0.5 * (3 * np.log(2 * np.pi) + logdet + innovation @ solved)

        # 卡尔曼增益和状态更新
        gain = cov @ hmat.T @ np.linalg.inv(innovation_cov)
        state = state + gain @ innovation
        # Joseph形式的协方差更新（数值更稳定）
        cov = (identity - gain @ hmat) @ cov @ (identity - gain @ hmat).T + gain @ r @ gain.T
        cov = (cov + cov.T) / 2.0

        filtered[t] = state

        # 检查数值稳定性
        if not np.all(np.isfinite(state)) or not np.all(np.isfinite(cov)):
            return 1e12, None

    neg_loglike = float(-loglike)
    return (neg_loglike, filtered) if return_states else (neg_loglike, None)


def initial_theta(y: np.ndarray) -> np.ndarray:
    """
    生成初始参数向量

    基于数据的统计特性设置合理的初始值：
        - μ=0, α=0: 初始漂移设为0
        - κ=log(1): 均值回复速度初始为1
        - σ₁: 基于对数收益率的年化波动率
        - σ₂=0.20: 趋势波动率初始为20%
        - ρ=0: 初始不假设相关
        - h=[0.1, 0.1, 0.1]: 因子载荷初始值
        - 观测噪声: 3%

    参数：
        y: 观测矩阵
    返回：
        12维初始参数向量
    """
    log_returns = np.diff(y[:, 0])                          # 对数收益率
    daily_vol = float(np.nanstd(log_returns))               # 日波动率
    daily_vol = daily_vol if np.isfinite(daily_vol) and daily_vol > 1e-5 else 0.02
    annual_vol = daily_vol * np.sqrt(252)                   # 年化波动率

    return np.array(
        [
            0.0,                                    # μ: 价格漂移
            0.0,                                    # α: 趋势均值
            np.log(1.0),                            # log(κ): 均值回复速度
            np.log(max(annual_vol, 0.05)),          # log(σ₁): 水平波动率
            np.log(0.20),                           # log(σ₂): 趋势波动率
            0.0,                                    # atanh(ρ): 噪声相关
            0.10, 0.10, 0.10,                      # h: 因子载荷
            np.log(0.03), np.log(0.03), np.log(0.03),  # log(观测噪声)
        ],
        dtype=float,
    )


def fit_teammate_state_space(oil_df: pd.DataFrame, maxiter: int) -> tuple[pd.DataFrame, StateSpaceFit]:
    """
    拟合双因子状态空间模型

    流程：
        1. 准备对数观测数据
        2. 设置参数边界
        3. 用L-BFGS-B优化负对数似然
        4. 用最优参数做卡尔曼滤波得到状态估计
        5. 将价格水平因子转回原始尺度作为油价指数

    参数：
        oil_df: 对数油价数据
        maxiter: 最大迭代次数
    返回：
        (包含状态估计的DataFrame, 拟合结果)
    """
    y = oil_df[["brent_log", "wti_log", "dubai_log"]].to_numpy(dtype=float)
    dt = 1.0 / 252.0  # 日频数据的时间步长（1年≈252个交易日）

    # 生成初始参数
    theta0 = initial_theta(y)

    # 参数边界（确保参数在合理范围内）
    bounds = [
        (-1.0, 1.0),                       # μ: 价格漂移
        (-2.0, 2.0),                        # α: 趋势均值
        (np.log(0.05), np.log(20.0)),       # log(κ): 均值回复速度
        (np.log(0.01), np.log(2.0)),        # log(σ₁): 水平波动率
        (np.log(0.01), np.log(2.0)),        # log(σ₂): 趋势波动率
        (-3.0, 3.0),                        # atanh(ρ): 噪声相关
        (-3.0, 3.0), (-3.0, 3.0), (-3.0, 3.0),  # h: 因子载荷
        (np.log(0.002), np.log(0.50)),      # log(观测噪声_brent)
        (np.log(0.002), np.log(0.50)),      # log(观测噪声_wti)
        (np.log(0.002), np.log(0.50)),      # log(观测噪声_dubai)
    ]

    # L-BFGS-B优化
    result = minimize(
        lambda th: kalman_filter(y, th, dt, return_states=False)[0],  # 目标函数：负对数似然
        theta0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": maxiter, "ftol": 1e-7, "maxls": 30},
    )

    # 用最优参数做滤波，获取状态估计
    neg_ll, filtered = kalman_filter(y, result.x, dt, return_states=True)
    _, _, _, _, _, named = unpack(result.x, dt)

    # 构建输出DataFrame
    out = oil_df[["date", "usd_cny", "brent", "wti", "dubai"]].copy()
    out["state_x"] = filtered[:, 0]                 # 价格水平因子
    out["state_delta"] = filtered[:, 1]             # 价格趋势因子
    out["state_space_index_usd"] = np.exp(out["state_x"])  # 转回原始尺度的油价指数

    # 记录拟合结果
    fit = StateSpaceFit(
        params=named,
        neg_loglike=float(neg_ll),
        success=bool(result.success),
        message=str(result.message),
        iterations=int(result.nit),
    )
    return out, fit


# ==================== 验证流程 ====================

def run_validation(config: base.MechanismConfig, maxiter: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """
    运行完整的验证流程

    流程：
        1. 读取油价面板数据
        2. 拟合双因子状态空间模型
        3. 用状态空间指数代入队友定价公式
        4. 计算评估指标

    参数：
        config: 机制配置
        maxiter: 最大迭代次数
    返回：
        (状态面板DataFrame, 验证结果DataFrame, 评估摘要字典)
    """
    # 读取数据
    panel = base.read_oil_panel()
    oil_df = prepare_log_observations(panel, config.start_date)

    # 拟合状态空间模型
    state_panel, fit = fit_teammate_state_space(oil_df, maxiter=maxiter)

    # 用状态空间指数运行队友公式验证
    domestic = base.read_domestic_adjustments(config.start_date)
    validation = teammate.simulate_teammate_control(domestic, state_panel, "state_space_index_usd", config)

    # 构建评估摘要
    summary = {
        "state_space_fit": asdict(fit),
        "metrics": teammate.build_summary(validation),
        "nardl_like": {
            "gasoline": base.run_nardl_like_test(validation, "gasoline"),
            "diesel": base.run_nardl_like_test(validation, "diesel"),
        },
    }
    return state_panel, validation, summary


# ==================== 命令行参数 ====================

def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="队友双因子状态空间验证")
    parser.add_argument("--start-date", default=base.MechanismConfig.start_date, help="分析起始日期")
    parser.add_argument("--window-size", type=int, default=base.MechanismConfig.window_size, help="窗口大小")
    parser.add_argument("--pricing-lag-days", type=int, default=0, help="定价滞后天数（队友默认为0）")
    parser.add_argument("--maxiter", type=int, default=250, help="MLE最大迭代次数")
    return parser.parse_args()


# ==================== 主函数 ====================

def main() -> None:
    """
    主函数：执行双因子状态空间模型验证

    输出文件：
        - state_space_oil_index.csv: 状态空间油价指数
        - state_space_teammate_validation.csv: 队友公式验证结果
        - summary_metrics.json: 评估摘要
    """
    args = parse_args()
    config = base.MechanismConfig(
        start_date=args.start_date,
        window_size=args.window_size,
        pricing_lag_days=args.pricing_lag_days,
    )

    # 运行验证
    state_panel, validation, summary = run_validation(config, maxiter=args.maxiter)

    # 输出结果
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    state_panel.to_csv(OUTPUT_DIR / "state_space_oil_index.csv", index=False, encoding="utf-8-sig")
    validation.to_csv(OUTPUT_DIR / "state_space_teammate_validation.csv", index=False, encoding="utf-8-sig")
    with (OUTPUT_DIR / "summary_metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"config": asdict(config), "summary": summary}, f, ensure_ascii=False, indent=2)

    # 打印报告
    print("队友双因子状态空间验证结果")
    print("模型拟合:", summary["state_space_fit"])
    print("汽油:", summary["metrics"]["gasoline"])
    print("柴油:", summary["metrics"]["diesel"])
    print("NARDL汽油:", summary["nardl_like"]["gasoline"])
    print("NARDL柴油:", summary["nardl_like"]["diesel"])
    print(f"\n输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
