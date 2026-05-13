"""
数据预处理模块：合并海关原油进口数据
=============================================
功能：将分年份存储的海关进口原油CSV文件（人民币/美元两种币种）合并为一张总表，
     并计算每吨原油的人民币和美元单价。

输入：海关进出口数量数据/ 目录下的CSV文件
输出：进口原油数量和金额_合并总表.csv 和 .xlsx
"""

from pathlib import Path
import re

import pandas as pd


# ==================== 常量定义 ====================

# 数据文件所在目录
DATA_DIR = Path("海关进出口数量数据")

# 文件名正则匹配模式，例如："2020年进口原油数量和金额（人民币）.csv"
# 提取年份和币种（人民币/美元）
FILE_RE = re.compile(r"(?P<year>\d{4})年进口原油数量和金额（(?P<currency>人民币|美元）\.csv$")

# 所有CSV文件共有的列名
COMMON_COLUMNS = [
    "数据年月",        # 格式如 "202001"
    "商品编码",        # HS编码
    "商品名称",        # 商品名称
    "第一数量",        # 数量（千克）
    "第一计量单位",    # 单位
    "第二数量",        # 第二数量
    "第二计量单位",    # 第二单位
]


# ==================== 工具函数 ====================

def clean_number(series: pd.Series) -> pd.Series:
    """
    清洗数值列：去除千分位逗号，将空字符串转为NA，转换为Int64类型

    参数：
        series: 包含数值的Series，可能带有逗号分隔符
    返回：
        清洗后的Int64类型Series
    """
    return (
        series.astype("string")
        .str.replace(",", "", regex=False)  # 去除千分位逗号
        .replace({"": pd.NA})               # 空字符串转为NA
        .astype("Int64")                     # 转为可空整数类型
    )


def read_source_file(path: Path) -> pd.DataFrame:
    """
    读取单个海关CSV数据文件

    参数：
        path: CSV文件路径
    返回：
        清洗后的DataFrame，包含标准列和金额列（按币种命名）

    处理逻辑：
        1. 从文件名解析年份和币种
        2. 尝试多种编码读取（utf-8-sig, gb18030, gbk）
        3. 删除全空列，清洗列名
        4. 验证必需列是否存在
        5. 清洗数值列
    """
    # 从文件名解析信息
    match = FILE_RE.match(path.name)
    if not match:
        raise ValueError(f"文件名不符合预期格式: {path.name}")

    currency = match.group("currency")  # 提取币种：人民币或美元

    # 尝试多种编码读取CSV
    last_error: UnicodeDecodeError | None = None
    for encoding in ("utf-8-sig", "gb18030", "gbk"):
        try:
            df = pd.read_csv(path, dtype="string", encoding=encoding)
            break
        except UnicodeDecodeError as error:
            last_error = error
    else:
        raise last_error  # 所有编码都失败则抛出最后一个错误

    # 数据清洗
    df = df.dropna(axis=1, how="all")           # 删除全空列
    df.columns = [column.strip() for column in df.columns]  # 去除列名空格

    # 将币种列重命名为带币种标识的金额列
    amount_column = f"金额_{currency}"
    df = df.rename(columns={currency: amount_column})

    # 验证必需列是否存在
    expected_columns = COMMON_COLUMNS + [amount_column]
    missing_columns = [column for column in expected_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{path.name} 缺少列: {', '.join(missing_columns)}")

    # 选取需要的列并清洗数值
    df = df[expected_columns].copy()
    df["第一数量"] = clean_number(df["第一数量"])
    df["第二数量"] = clean_number(df["第二数量"])
    df[amount_column] = clean_number(df[amount_column])
    return df


def build_merged_table(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """
    构建合并总表

    处理流程：
        1. 扫描目录下所有符合命名规范的CSV文件
        2. 按币种（人民币/美元）分组读取
        3. 分别纵向合并同币种数据
        4. 按公共列横向合并人民币和美元数据
        5. 提取年份、月份
        6. 计算每吨单价（人民币和美元）

    参数：
        data_dir: 数据文件目录路径
    返回：
        合并后的完整DataFrame
    """
    # 扫描并排序CSV文件
    source_files = sorted(path for path in data_dir.glob("*.csv") if FILE_RE.match(path.name))
    if not source_files:
        raise FileNotFoundError(f"没有在 {data_dir} 找到待合并 CSV 文件")

    # 按币种分组存储
    frames_by_currency = {"人民币": [], "美元": []}
    for path in source_files:
        currency = FILE_RE.match(path.name).group("currency")
        frames_by_currency[currency].append(read_source_file(path))

    # 检查是否两种币种的数据都有
    missing_currency = [currency for currency, frames in frames_by_currency.items() if not frames]
    if missing_currency:
        raise ValueError(f"缺少币种文件: {', '.join(missing_currency)}")

    # 纵向合并各币种数据
    rmb = pd.concat(frames_by_currency["人民币"], ignore_index=True)
    usd = pd.concat(frames_by_currency["美元"], ignore_index=True)

    # 按公共列横向合并（一对一匹配）
    merged = rmb.merge(usd, on=COMMON_COLUMNS, how="outer", validate="one_to_one")

    # 提取年份和月份
    merged["年"] = merged["数据年月"].str.slice(0, 4).astype("Int64")
    merged["月"] = merged["数据年月"].str.slice(4, 6).astype("Int64")

    # 计算每吨单价（原始数据第一数量单位为千克，除以1000得吨）
    quantity_tons = merged["第一数量"] / 1000
    merged["每吨人民币"] = (merged["金额_人民币"] / quantity_tons).round(2)
    merged["每吨美元"] = (merged["金额_美元"] / quantity_tons).round(2)

    # 定义输出列顺序
    output_columns = [
        "年",
        "月",
        "数据年月",
        "商品编码",
        "商品名称",
        "第一数量",
        "第一计量单位",
        "第二数量",
        "第二计量单位",
        "金额_人民币",
        "金额_美元",
        "每吨人民币",
        "每吨美元",
    ]
    return merged[output_columns].sort_values("数据年月").reset_index(drop=True)


def main() -> None:
    """主函数：执行数据合并并输出CSV和Excel文件"""
    merged = build_merged_table()

    # 定义输出路径
    csv_path = DATA_DIR / "进口原油数量和金额_合并总表.csv"
    xlsx_path = DATA_DIR / "进口原油数量和金额_合并总表.xlsx"

    # 保存结果
    merged.to_csv(csv_path, index=False, encoding="utf-8-sig")
    merged.to_excel(xlsx_path, index=False)

    print(f"已合并 {len(merged)} 行")
    print(f"CSV: {csv_path}")
    print(f"Excel: {xlsx_path}")


if __name__ == "__main__":
    main()
