from pathlib import Path
import re

import pandas as pd


DATA_DIR = Path("海关进出口数量数据")
FILE_RE = re.compile(r"(?P<year>\d{4})年进口原油数量和金额（(?P<currency>人民币|美元)）\.csv$")
COMMON_COLUMNS = [
    "数据年月",
    "商品编码",
    "商品名称",
    "第一数量",
    "第一计量单位",
    "第二数量",
    "第二计量单位",
]


def clean_number(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.replace(",", "", regex=False)
        .replace({"": pd.NA})
        .astype("Int64")
    )


def read_source_file(path: Path) -> pd.DataFrame:
    match = FILE_RE.match(path.name)
    if not match:
        raise ValueError(f"文件名不符合预期格式: {path.name}")

    currency = match.group("currency")
    last_error: UnicodeDecodeError | None = None
    for encoding in ("utf-8-sig", "gb18030", "gbk"):
        try:
            df = pd.read_csv(path, dtype="string", encoding=encoding)
            break
        except UnicodeDecodeError as error:
            last_error = error
    else:
        raise last_error
    df = df.dropna(axis=1, how="all")
    df.columns = [column.strip() for column in df.columns]

    amount_column = f"金额_{currency}"
    df = df.rename(columns={currency: amount_column})

    expected_columns = COMMON_COLUMNS + [amount_column]
    missing_columns = [column for column in expected_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{path.name} 缺少列: {', '.join(missing_columns)}")

    df = df[expected_columns].copy()
    df["第一数量"] = clean_number(df["第一数量"])
    df["第二数量"] = clean_number(df["第二数量"])
    df[amount_column] = clean_number(df[amount_column])
    return df


def build_merged_table(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    source_files = sorted(path for path in data_dir.glob("*.csv") if FILE_RE.match(path.name))
    if not source_files:
        raise FileNotFoundError(f"没有在 {data_dir} 找到待合并 CSV 文件")

    frames_by_currency = {"人民币": [], "美元": []}
    for path in source_files:
        currency = FILE_RE.match(path.name).group("currency")
        frames_by_currency[currency].append(read_source_file(path))

    missing_currency = [currency for currency, frames in frames_by_currency.items() if not frames]
    if missing_currency:
        raise ValueError(f"缺少币种文件: {', '.join(missing_currency)}")

    rmb = pd.concat(frames_by_currency["人民币"], ignore_index=True)
    usd = pd.concat(frames_by_currency["美元"], ignore_index=True)

    merged = rmb.merge(usd, on=COMMON_COLUMNS, how="outer", validate="one_to_one")
    merged["年"] = merged["数据年月"].str.slice(0, 4).astype("Int64")
    merged["月"] = merged["数据年月"].str.slice(4, 6).astype("Int64")
    quantity_tons = merged["第一数量"] / 1000
    merged["每吨人民币"] = (merged["金额_人民币"] / quantity_tons).round(2)
    merged["每吨美元"] = (merged["金额_美元"] / quantity_tons).round(2)

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
    merged = build_merged_table()
    csv_path = DATA_DIR / "进口原油数量和金额_合并总表.csv"
    xlsx_path = DATA_DIR / "进口原油数量和金额_合并总表.xlsx"

    merged.to_csv(csv_path, index=False, encoding="utf-8-sig")
    merged.to_excel(xlsx_path, index=False)

    print(f"已合并 {len(merged)} 行")
    print(f"CSV: {csv_path}")
    print(f"Excel: {xlsx_path}")


if __name__ == "__main__":
    main()
