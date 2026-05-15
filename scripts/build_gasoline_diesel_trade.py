"""Build monthly gasoline/diesel trade data from Customs exports.

Input files are expected under:
    data/task2/customs_raw/

File names must follow:
    gasoline_import_2016.xlsx
    gasoline_export_2016.csv
    diesel_import_2016.xlsx
    diesel_export_2016.xlsx

Output:
    data/task2/gasoline_diesel_trade_monthly.csv
    data/task2/gasoline_diesel_trade_long.csv
    data/task2/gasoline_diesel_trade_check_report.txt
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "task2"
RAW_DIR = DATA_DIR / "customs_raw"
OUT_MONTHLY = DATA_DIR / "gasoline_diesel_trade_monthly.csv"
OUT_LONG = DATA_DIR / "gasoline_diesel_trade_long.csv"
OUT_REPORT = DATA_DIR / "gasoline_diesel_trade_check_report.txt"

FILE_RE = re.compile(r"^(gasoline|diesel)_(import|export)_(20\d{2})\.(xlsx|xls|csv)$", re.I)
MIXED_FILE_RE = re.compile(r"^customs_(import|export)_(20\d{2})\.(xlsx|xls|csv)$", re.I)
EXPECTED_SERIES = [
    ("gasoline", "import"),
    ("gasoline", "export"),
    ("diesel", "import"),
    ("diesel", "export"),
]

COMMODITY_TO_FUEL = {
    "27101210": "gasoline",
    "2710121": "gasoline",
    "27101923": "diesel",
    "27101926": "diesel",
}


@dataclass(frozen=True)
class FileMeta:
    path: Path
    fuel: str | None
    flow: str
    year: int


def clean_number(value) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text in {"", "-", "--", "—", "nan", "None", "?"}:
        return np.nan
    text = text.replace(",", "").replace("，", "")
    text = text.replace(" ", "")
    try:
        return float(text)
    except ValueError:
        return np.nan


def normalize_col(col) -> str:
    return re.sub(r"\s+", "", str(col).strip())


def find_files(raw_dir: Path) -> list[FileMeta]:
    files: list[FileMeta] = []
    if not raw_dir.exists():
        return files
    for path in raw_dir.iterdir():
        if not path.is_file():
            continue
        match = FILE_RE.match(path.name)
        if match:
            fuel, flow, year, _ = match.groups()
            files.append(FileMeta(path=path, fuel=fuel.lower(), flow=flow.lower(), year=int(year)))
            continue
        mixed_match = MIXED_FILE_RE.match(path.name)
        if mixed_match:
            flow, year, _ = mixed_match.groups()
            files.append(FileMeta(path=path, fuel=None, flow=flow.lower(), year=int(year)))
    return sorted(files, key=lambda item: (item.fuel or "mixed", item.flow, item.year, item.path.name))


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        for encoding in ["utf-8-sig", "gb18030", "gbk"]:
            try:
                return pd.read_csv(path, dtype=str, encoding=encoding)
            except UnicodeDecodeError:
                continue
        return pd.read_csv(path, dtype=str)

    # Customs Excel files sometimes contain title rows before the real header.
    for header in range(0, 8):
        df = pd.read_excel(path, dtype=str, header=header)
        cols = [normalize_col(c) for c in df.columns]
        joined = "|".join(cols)
        if any(key in joined for key in ["数据年月", "年月", "月份"]) and any("数量" in c for c in cols):
            df.columns = cols
            return df

    df = pd.read_excel(path, dtype=str)
    df.columns = [normalize_col(c) for c in df.columns]
    return df


def pick_column(columns: list[str], patterns: list[str]) -> str | None:
    for pattern in patterns:
        regex = re.compile(pattern)
        for col in columns:
            if regex.search(col):
                return col
    return None


def parse_month(df: pd.DataFrame, fallback_year: int) -> pd.Series:
    cols = [normalize_col(c) for c in df.columns]
    df = df.copy()
    df.columns = cols

    date_col = pick_column(cols, [r"数据年月", r"年月", r"月份", r"时间", r"日期"])
    if date_col:
        raw = df[date_col].astype(str).str.replace(r"\D", "", regex=True)
        ym = raw.str.extract(r"((?:19|20)\d{2})(\d{1,2})")
        month = pd.to_numeric(ym[1], errors="coerce")
        parsed = ym[0] + "-" + month.astype("Int64").astype(str).str.zfill(2)
        parsed = parsed.where(month.between(1, 12))
        if parsed.notna().any():
            return parsed

    year_col = pick_column(cols, [r"^年$", r"年份"])
    month_col = pick_column(cols, [r"^月$", r"月份"])
    if month_col:
        if year_col:
            year = pd.to_numeric(df[year_col], errors="coerce")
        else:
            year = pd.Series([fallback_year] * len(df), index=df.index)
        month = pd.to_numeric(df[month_col].astype(str).str.replace(r"\D", "", regex=True), errors="coerce")
        return year.astype("Int64").astype(str) + "-" + month.astype("Int64").astype(str).str.zfill(2)

    raise ValueError("无法识别月份列")


def quantity_to_10k_tonnes(quantity: pd.Series, unit: pd.Series | str) -> pd.Series:
    qty = quantity.map(clean_number)
    if isinstance(unit, str):
        unit_series = pd.Series([unit] * len(qty), index=qty.index)
    else:
        unit_series = unit.fillna("").astype(str)

    out = pd.Series(np.nan, index=qty.index, dtype=float)
    unit_norm = unit_series.str.replace(r"\s+", "", regex=True)

    kg = unit_norm.str.contains("千克|公斤|kg", case=False, regex=True, na=False)
    tonne = unit_norm.str.fullmatch(r"吨|公吨|t|ton|tonne", case=False, na=False)
    ten_k_tonne = unit_norm.str.contains("万吨|万公吨", regex=True, na=False)
    k_tonne = unit_norm.str.contains("千吨|kt", case=False, regex=True, na=False)

    out.loc[kg] = qty.loc[kg] / 10_000_000.0
    out.loc[tonne] = qty.loc[tonne] / 10_000.0
    out.loc[ten_k_tonne] = qty.loc[ten_k_tonne]
    out.loc[k_tonne] = qty.loc[k_tonne] / 10.0
    out.loc[out.isna()] = qty.loc[out.isna()] / 10_000_000.0
    return out


def parse_one_file(meta: FileMeta) -> pd.DataFrame:
    df = read_table(meta.path)
    df.columns = [normalize_col(c) for c in df.columns]
    df = df.dropna(how="all")
    cols = list(df.columns)

    month = parse_month(df, meta.year)
    qty_col = pick_column(cols, [r"^第一数量$", r"数量"])
    unit_col = pick_column(cols, [r"^第一计量单位$", r"计量单位", r"单位"])
    amount_col = pick_column(cols, [r"人民币", r"美元", r"金额"])
    code_col = pick_column(cols, [r"商品编码", r"商品代码", r"编码"])
    name_col = pick_column(cols, [r"商品名称", r"商品"])

    if qty_col is None:
        raise ValueError("无法识别数量列")

    if meta.fuel is None:
        if code_col is None and name_col is None:
            raise ValueError("混合文件无法识别商品编码/名称列")
        code = df[code_col].astype(str).str.replace(r"\D", "", regex=True) if code_col else ""
        fuel = code.map(COMMODITY_TO_FUEL) if code_col else pd.Series([None] * len(df), index=df.index)
        if name_col:
            name = df[name_col].astype(str)
            fuel = fuel.where(~fuel.isna(), np.where(name.str.contains("汽油", na=False), "gasoline", None))
            fuel = pd.Series(fuel, index=df.index)
            fuel = fuel.where(~fuel.isna(), np.where(name.str.contains("柴油", na=False), "diesel", None))
            fuel = pd.Series(fuel, index=df.index)
    else:
        fuel = pd.Series([meta.fuel] * len(df), index=df.index)

    unit = df[unit_col] if unit_col else "千克"
    amount = df[amount_col].map(clean_number) if amount_col else np.nan

    out = pd.DataFrame(
        {
            "month": month,
            "fuel": fuel,
            "flow": meta.flow,
            "year": meta.year,
            "source_file": meta.path.name,
            "quantity_raw": df[qty_col].map(clean_number),
            "quantity_unit": unit if isinstance(unit, str) else unit.astype(str),
            "quantity_10k_tonnes": quantity_to_10k_tonnes(df[qty_col], unit),
            "amount_raw": amount,
            "amount_column": amount_col or "",
        }
    )
    out = out.dropna(subset=["month"])
    out = out[out["fuel"].isin(["gasoline", "diesel"])]
    out = out[out["month"].str.match(r"^20\d{2}-\d{2}$", na=False)]
    out = out[out["quantity_10k_tonnes"].notna()]
    return out


def expected_months(start_year: int, end_month: str) -> pd.Index:
    start = f"{start_year}-01"
    return pd.period_range(start=start, end=end_month, freq="M").astype(str)


def build_report(
    long_df: pd.DataFrame,
    monthly: pd.DataFrame,
    files: list[FileMeta],
    errors: list[str],
    start_year: int,
    end_month: str,
) -> str:
    lines: list[str] = []
    lines.append("汽油/柴油进出口数据清洗检查报告")
    lines.append("=" * 40)
    lines.append(f"原始目录：{RAW_DIR}")
    lines.append(f"识别到文件数：{len(files)}")
    lines.append(f"输出月度表：{OUT_MONTHLY}")
    lines.append("")
    lines.append("单位换算说明：")
    lines.append("- 千克/公斤/kg -> 除以 10,000,000，得到万吨。")
    lines.append("- 吨/公吨/t -> 除以 10,000，得到万吨。")
    lines.append("- 万吨/万公吨 -> 保持不变。")
    lines.append("- 千吨/kt -> 除以 10，得到万吨。")
    lines.append("- 未识别单位时暂按千克处理，并建议人工核对原始文件。")
    lines.append("")
    lines.append("缺失处理说明：")
    lines.append("- 海关表中未出现的月份不再按 0 处理。")
    lines.append("- 输出月度表对进出口数量使用线性插值；序列两端缺失用最近有效值延伸。")
    lines.append("")

    if errors:
        lines.append("读取失败或格式异常文件：")
        lines.extend(f"- {err}" for err in errors)
        lines.append("")

    expected = expected_months(start_year, end_month)
    expected_years = sorted({int(month[:4]) for month in expected})
    seen_files = {(item.fuel, item.flow, item.year) for item in files}
    seen_mixed_files = {(item.flow, item.year) for item in files if item.fuel is None}
    lines.append("缺失年份检查：")
    for fuel, flow in EXPECTED_SERIES:
        missing_years = [
            year
            for year in expected_years
            if (fuel, flow, year) not in seen_files and (flow, year) not in seen_mixed_files
        ]
        lines.append(f"- {fuel}_{flow}: {missing_years if missing_years else '无'}")
    lines.append("")

    lines.append("缺失月份检查：")
    for fuel, flow in EXPECTED_SERIES:
        col = f"{fuel}_{flow}"
        present = set(monthly.loc[monthly[col].notna(), "month"]) if col in monthly else set()
        missing = [month for month in expected if month not in present]
        shown = ", ".join(missing[:30])
        suffix = " ..." if len(missing) > 30 else ""
        lines.append(f"- {col}: 缺失 {len(missing)} 个月；{shown}{suffix}")
    lines.append("")

    lines.append("异常值检查：")
    for col in [f"{fuel}_{flow}" for fuel, flow in EXPECTED_SERIES]:
        if col not in monthly:
            continue
        ser = monthly[col].dropna()
        if ser.empty:
            lines.append(f"- {col}: 无有效数据")
            continue
        neg_count = int((ser < 0).sum())
        q1 = ser.quantile(0.25)
        q3 = ser.quantile(0.75)
        iqr = q3 - q1
        upper = q3 + 3 * iqr
        extreme = monthly.loc[monthly[col] > upper, ["month", col]]
        lines.append(
            f"- {col}: 均值={ser.mean():.4f}, 最小={ser.min():.4f}, 最大={ser.max():.4f}, "
            f"负值={neg_count}, 高于Q3+3IQR={len(extreme)}"
        )
        if len(extreme) > 0:
            lines.append("  " + extreme.head(10).to_string(index=False))
    lines.append("")

    if not long_df.empty:
        lines.append("原始单位分布：")
        unit_counts = (
            long_df.groupby(["fuel", "flow", "quantity_unit"], dropna=False)
            .size()
            .reset_index(name="rows")
            .sort_values(["fuel", "flow", "rows"], ascending=[True, True, False])
        )
        lines.append(unit_counts.to_string(index=False))
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    files = find_files(args.raw_dir)
    frames: list[pd.DataFrame] = []
    errors: list[str] = []

    for meta in files:
        try:
            frames.append(parse_one_file(meta))
        except Exception as exc:  # noqa: BLE001 - report all bad raw files.
            errors.append(f"{meta.path.name}: {exc}")

    if frames:
        long_df = pd.concat(frames, ignore_index=True)
        long_df = (
            long_df.groupby(["month", "fuel", "flow"], as_index=False)
            .agg(
                quantity_10k_tonnes=("quantity_10k_tonnes", "sum"),
                quantity_raw=("quantity_raw", "sum"),
                amount_raw=("amount_raw", "sum"),
                source_file=("source_file", lambda x: ";".join(sorted(set(map(str, x))))),
                quantity_unit=("quantity_unit", lambda x: ";".join(sorted(set(map(str, x))))),
            )
            .sort_values(["month", "fuel", "flow"])
        )
    else:
        long_df = pd.DataFrame(
            columns=[
                "month",
                "fuel",
                "flow",
                "quantity_10k_tonnes",
                "quantity_raw",
                "amount_raw",
                "source_file",
                "quantity_unit",
            ]
        )

    if long_df.empty:
        monthly = pd.DataFrame({"month": expected_months(args.start_year, args.end_month)})
    else:
        monthly = long_df.pivot_table(
            index="month",
            columns=["fuel", "flow"],
            values="quantity_10k_tonnes",
            aggfunc="sum",
        )
        monthly.columns = [f"{fuel}_{flow}" for fuel, flow in monthly.columns]
        monthly = monthly.reset_index()
        expected = pd.DataFrame({"month": expected_months(args.start_year, args.end_month)})
        monthly = expected.merge(monthly, on="month", how="left")

    for fuel, flow in EXPECTED_SERIES:
        col = f"{fuel}_{flow}"
        if col not in monthly.columns:
            monthly[col] = np.nan

    trade_cols = [f"{fuel}_{flow}" for fuel, flow in EXPECTED_SERIES]
    monthly[trade_cols] = monthly[trade_cols].apply(pd.to_numeric, errors="coerce")
    monthly[trade_cols] = monthly[trade_cols].interpolate(
        method="linear",
        limit_direction="both",
    )

    monthly = monthly[
        ["month", "gasoline_import", "gasoline_export", "diesel_import", "diesel_export"]
    ]
    monthly.to_csv(args.output, index=False, encoding="utf-8-sig")
    long_df.to_csv(args.long_output, index=False, encoding="utf-8-sig")

    report = build_report(long_df, monthly, files, errors, args.start_year, args.end_month)
    args.report.write_text(report, encoding="utf-8")

    print(f"已输出：{args.output}")
    print(f"已输出：{args.long_output}")
    print(f"已输出：{args.report}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output", type=Path, default=OUT_MONTHLY)
    parser.add_argument("--long-output", type=Path, default=OUT_LONG)
    parser.add_argument("--report", type=Path, default=OUT_REPORT)
    parser.add_argument("--start-year", type=int, default=2016)
    parser.add_argument("--end-month", default="2026-05")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args(sys.argv[1:]))
