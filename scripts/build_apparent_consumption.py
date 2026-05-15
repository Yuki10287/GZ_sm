"""Merge gasoline/diesel output with customs trade to build apparent consumption.

Required trade input:
    data/task2/gasoline_diesel_trade_monthly.csv

Production input is discovered under data/task2/. The script supports either:
1. one wide file containing month, gasoline_output, diesel_output; or
2. separate files whose names include gasoline/汽油 and diesel/柴油 plus output/产量.

All quantity columns are converted to ten-thousand tonnes before merging.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "task2"
TRADE_CSV = DATA_DIR / "gasoline_diesel_trade_monthly.csv"
OUT_CSV = ROOT / "outputs" / "task2_dynamic_pricing" / "task2_apparent_consumption_monthly.csv"
OUT_REPORT = DATA_DIR / "task2_apparent_consumption_check_report.txt"


def clean_number(value) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().replace(",", "").replace("，", "")
    if text in {"", "-", "--", "—", "nan", "None", "?"}:
        return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def normalize_col(col) -> str:
    return re.sub(r"\s+", "", str(col).strip())


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        for encoding in ["utf-8-sig", "gb18030", "gbk"]:
            try:
                df = pd.read_csv(path, dtype=str, encoding=encoding)
                df.columns = [normalize_col(c) for c in df.columns]
                return df
            except UnicodeDecodeError:
                continue
        df = pd.read_csv(path, dtype=str)
    else:
        df = pd.read_excel(path, dtype=str)
    df.columns = [normalize_col(c) for c in df.columns]
    return df


def read_csv_with_header(path: Path, header: int) -> pd.DataFrame:
    for encoding in ["utf-8-sig", "gb18030", "gbk"]:
        try:
            df = pd.read_csv(path, dtype=str, encoding=encoding, header=header)
            df.columns = [normalize_col(c) for c in df.columns]
            return df
        except UnicodeDecodeError:
            continue
    df = pd.read_csv(path, dtype=str, header=header)
    df.columns = [normalize_col(c) for c in df.columns]
    return df


def parse_month(df: pd.DataFrame) -> pd.Series:
    cols = list(df.columns)
    for col in cols:
        if any(key in col for key in ["month", "年月", "月份", "日期", "时间"]):
            raw = df[col].astype(str)
            if raw.str.match(r"^20\d{2}-\d{1,2}$").any():
                split = raw.str.extract(r"(20\d{2})-(\d{1,2})")
                month = pd.to_numeric(split[1], errors="coerce")
                return split[0] + "-" + month.astype("Int64").astype(str).str.zfill(2)
            digits = raw.str.replace(r"\D", "", regex=True)
            split = digits.str.extract(r"((?:19|20)\d{2})(\d{1,2})")
            if split[0].notna().any():
                return split[0] + "-" + split[1].astype(float).astype("Int64").astype(str).str.zfill(2)

    year_col = next((c for c in cols if c in {"年", "年份", "year"}), None)
    month_col = next((c for c in cols if c in {"月", "月份"}), None)
    if year_col and month_col:
        year = pd.to_numeric(df[year_col], errors="coerce").astype("Int64").astype(str)
        month = pd.to_numeric(df[month_col], errors="coerce").astype("Int64").astype(str).str.zfill(2)
        return year + "-" + month
    raise ValueError("无法识别产量文件中的月份列")


def infer_unit_factor(col_name: str, file_name: str) -> float:
    text = f"{col_name} {file_name}".lower()
    if "万吨" in text:
        return 1.0
    if "千吨" in text or "kt" in text:
        return 0.1
    if "吨" in text or "ton" in text:
        return 1 / 10_000.0
    if "千克" in text or "公斤" in text or "kg" in text:
        return 1 / 10_000_000.0
    # National Bureau monthly output tables are commonly in ten-thousand tonnes.
    return 1.0


def parse_nbs_wide_output(path: Path) -> pd.DataFrame | None:
    """Parse NBS-style tables with indicators in rows and months in columns."""
    if path.suffix.lower() != ".csv":
        return None

    last_error: Exception | None = None
    for header in range(0, 8):
        try:
            df = read_csv_with_header(path, header)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue

        month_cols = []
        for col in df.columns:
            match = re.fullmatch(r"(20\d{2})年(\d{1,2})月", col)
            if match:
                month_cols.append((col, f"{match.group(1)}-{int(match.group(2)):02d}"))
        indicator_col = next((c for c in df.columns if "指标" in c), None)
        if not indicator_col or not month_cols:
            continue

        rows: list[pd.DataFrame] = []
        indicator = df[indicator_col].astype(str).str.replace(r"\s+", "", regex=True)
        for fuel, cn_name in [("gasoline", "汽油"), ("diesel", "柴油")]:
            mask = indicator.str.contains(cn_name, na=False)
            mask &= indicator.str.contains("产量", na=False)
            mask &= indicator.str.contains("当期", na=False)
            mask &= ~indicator.str.contains("累计", na=False)
            if not mask.any():
                continue
            row = df.loc[mask].iloc[0]
            part = pd.DataFrame(
                {
                    "month": [month for _, month in month_cols],
                    f"{fuel}_output": [clean_number(row[col]) for col, _ in month_cols],
                }
            )
            rows.append(part)

        if not rows:
            continue

        out = rows[0]
        for part in rows[1:]:
            out = out.merge(part, on="month", how="outer")
        out = out.sort_values("month").reset_index(drop=True)
        return out

    if last_error:
        raise last_error
    return None


def output_column_candidates(df: pd.DataFrame, fuel: str) -> list[str]:
    fuel_keys = {
        "gasoline": ["gasoline", "汽油"],
        "diesel": ["diesel", "柴油"],
    }[fuel]
    output_keys = ["output", "产量", "生产量"]
    cols: list[str] = []
    for col in df.columns:
        lower = col.lower()
        if any(key in lower for key in fuel_keys) and any(key in lower for key in output_keys):
            cols.append(col)
    if not cols:
        for col in df.columns:
            lower = col.lower()
            if any(key in lower for key in fuel_keys):
                cols.append(col)
    return cols


def discover_output_files(data_dir: Path) -> list[Path]:
    if not data_dir.exists():
        return []
    files: list[Path] = []
    for path in data_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in {".csv", ".xlsx", ".xls"}:
            continue
        name = path.name.lower()
        if "trade" in name or "apparent" in name:
            continue
        if any(k in name for k in ["output", "产量", "汽油", "柴油", "gasoline", "diesel"]):
            files.append(path)
    return sorted(files)


def build_output_table(data_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    files = discover_output_files(data_dir)
    notes: list[str] = []
    frames: list[pd.DataFrame] = []

    for path in files:
        try:
            wide = parse_nbs_wide_output(path)
            if wide is not None:
                frames.append(wide)
                notes.append(f"{path.name}: 按国家统计局宽表解析，产量单位为万吨")
                continue

            df = read_table(path)
            month = parse_month(df)
            file_frame = pd.DataFrame({"month": month})
            found_any = False
            for fuel in ["gasoline", "diesel"]:
                candidates = output_column_candidates(df, fuel)
                if candidates:
                    col = candidates[0]
                    factor = infer_unit_factor(col, path.name)
                    file_frame[f"{fuel}_output"] = df[col].map(clean_number) * factor
                    notes.append(f"{path.name}: 使用 {col} 作为 {fuel}_output，换算系数 {factor}")
                    found_any = True
            if found_any:
                frames.append(file_frame)
        except Exception as exc:  # noqa: BLE001 - continue scanning other files.
            notes.append(f"{path.name}: 读取失败，{exc}")

    if not frames:
        return pd.DataFrame(columns=["month", "gasoline_output", "diesel_output"]), notes

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.groupby("month", as_index=False).agg(
        gasoline_output=("gasoline_output", "first"),
        diesel_output=("diesel_output", "first"),
    )
    return merged, notes


def run(args: argparse.Namespace) -> None:
    if not args.trade_csv.exists():
        raise SystemExit(f"缺少贸易数据：{args.trade_csv}。请先运行 build_gasoline_diesel_trade.py")

    trade = pd.read_csv(args.trade_csv)
    output, notes = build_output_table(args.data_dir)

    if output.empty or output[["gasoline_output", "diesel_output"]].dropna(how="all").empty:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            "未找到可识别的国家统计局汽油/柴油产量文件，未生成表观消费量。\n"
            "请把产量文件放入 data/task2/，文件名或列名包含 汽油/柴油 与 产量。\n\n"
            + "\n".join(notes)
            + "\n",
            encoding="utf-8",
        )
        print(f"未找到产量文件，已输出检查报告：{args.report}")
        return

    merged = output.merge(trade, on="month", how="outer").sort_values("month")
    for col in [
        "gasoline_output",
        "gasoline_import",
        "gasoline_export",
        "diesel_output",
        "diesel_import",
        "diesel_export",
    ]:
        if col not in merged.columns:
            merged[col] = np.nan
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    fill_cols = [
        "gasoline_output",
        "gasoline_import",
        "gasoline_export",
        "diesel_output",
        "diesel_import",
        "diesel_export",
    ]
    merged[fill_cols] = merged[fill_cols].interpolate(
        method="linear",
        limit_direction="both",
    )

    merged["gasoline_apparent_consumption"] = (
        merged["gasoline_output"] + merged["gasoline_import"] - merged["gasoline_export"]
    )
    merged["diesel_apparent_consumption"] = (
        merged["diesel_output"] + merged["diesel_import"] - merged["diesel_export"]
    )

    final_cols = [
        "month",
        "gasoline_output",
        "gasoline_import",
        "gasoline_export",
        "gasoline_apparent_consumption",
        "diesel_output",
        "diesel_import",
        "diesel_export",
        "diesel_apparent_consumption",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged[final_cols].to_csv(args.output, index=False, encoding="utf-8-sig")

    report_lines = ["表观消费量合并检查报告", "=" * 40, f"输出：{args.output}", ""]
    report_lines.append("产量文件识别说明：")
    report_lines.extend(f"- {note}" for note in notes)
    report_lines.append("")
    report_lines.append("缺失处理说明：")
    report_lines.append("- 产量、进口、出口月度缺失值使用线性插值。")
    report_lines.append("- 序列两端缺失用最近有效值延伸，以保证表观消费量可计算。")
    report_lines.append("")
    report_lines.append("缺失值统计：")
    report_lines.append(merged[final_cols].isna().sum().to_string())
    report_lines.append("")
    report_lines.append("描述统计：")
    report_lines.append(merged[final_cols[1:]].describe().to_string())
    args.report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"已输出：{args.output}")
    print(f"已输出：{args.report}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--trade-csv", type=Path, default=TRADE_CSV)
    parser.add_argument("--output", type=Path, default=OUT_CSV)
    parser.add_argument("--report", type=Path, default=OUT_REPORT)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args(sys.argv[1:]))
