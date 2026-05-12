from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

import task1_price_mechanism as base


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "task1_teammate"


def teammate_phi(ma_usd: float, config: base.MechanismConfig) -> float:
    if ma_usd <= config.floor_usd_per_bbl:
        return 0.0
    if ma_usd >= config.ceiling_usd_per_bbl:
        return config.ceiling_up_factor
    return 1.0


def teammate_window(
    panel: pd.DataFrame,
    prev_date: pd.Timestamp,
    date: pd.Timestamp,
    price_col: str,
    config: base.MechanismConfig,
) -> pd.DataFrame:
    end = date - pd.Timedelta(days=config.pricing_lag_days)
    window = panel[(panel["date"] > prev_date) & (panel["date"] <= end)].copy()
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
    rows: list[dict[str, object]] = []
    carry_gasoline = 0.0
    carry_diesel = 0.0

    for i in range(1, len(domestic)):
        prev_date = domestic.loc[i - 1, "adjust_date"]
        date = domestic.loc[i, "adjust_date"]
        window = teammate_window(panel, prev_date, date, price_col, config)
        if window.empty:
            continue

        start_usd = float(window[price_col].iloc[0])
        ma_usd = float(window[price_col].mean())
        fx_avg = float(window["usd_cny"].mean())
        if not np.isfinite(start_usd) or start_usd == 0:
            continue

        change_rate = (ma_usd - start_usd) / start_usd
        phi = teammate_phi(ma_usd, config)
        gasoline_raw = ma_usd * change_rate * config.gasoline_bbl_per_ton * fx_avg * config.tax_factor
        diesel_raw = ma_usd * change_rate * config.diesel_bbl_per_ton * fx_avg * config.tax_factor

        gasoline_total = gasoline_raw * phi + carry_gasoline
        diesel_total = diesel_raw * phi + carry_diesel
        gasoline_theory = gasoline_total if abs(gasoline_total) >= config.threshold_yuan_per_ton else 0.0
        diesel_theory = diesel_total if abs(diesel_total) >= config.threshold_yuan_per_ton else 0.0
        carry_gasoline = 0.0 if gasoline_theory else gasoline_total
        carry_diesel = 0.0 if diesel_theory else diesel_total

        rows.append(
            {
                "adjust_date": date,
                "prev_adjust_date": prev_date,
                "oil_index": price_col,
                "window_observations": len(window),
                "window_start": window["date"].min(),
                "window_end": window["date"].max(),
                "start_usd_per_bbl": start_usd,
                "ma_usd_per_bbl": ma_usd,
                "avg_usd_cny": fx_avg,
                "oil_change_rate": change_rate,
                "phi": phi,
                "zone": base.zone_for_price(ma_usd, config),
                "gasoline_formula_delta": gasoline_raw,
                "diesel_formula_delta": diesel_raw,
                "gasoline_theory_delta": gasoline_theory,
                "diesel_theory_delta": diesel_theory,
                "gasoline_carry_after": carry_gasoline,
                "diesel_carry_after": carry_diesel,
                "gasoline_actual_delta": domestic.loc[i, "gasoline_actual_delta"],
                "diesel_actual_delta": domestic.loc[i, "diesel_actual_delta"],
                "gasoline_price": domestic.loc[i, "gasoline_price"],
                "diesel_price": domestic.loc[i, "diesel_price"],
                "ref_ma_usd_per_bbl": ma_usd,
            }
        )

    return pd.DataFrame(rows)


def build_summary(df: pd.DataFrame) -> dict[str, object]:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run teammate's original Task 1 pricing-control formula.")
    parser.add_argument("--start-date", default=base.MechanismConfig.start_date)
    parser.add_argument("--window-size", type=int, default=base.MechanismConfig.window_size)
    parser.add_argument("--pricing-lag-days", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = base.MechanismConfig(
        start_date=args.start_date,
        window_size=args.window_size,
        pricing_lag_days=args.pricing_lag_days,
    )
    panel = base.read_oil_panel()
    domestic = base.read_domestic_adjustments(config.start_date)

    outputs: dict[str, pd.DataFrame] = {}
    summaries: dict[str, object] = {}
    for name, price_col in {
        "fixed_basket": "basket_usd",
        "pca_kalman": "kalman_index_usd",
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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in outputs.items():
        df.to_csv(OUTPUT_DIR / f"teammate_validation_{name}.csv", index=False, encoding="utf-8-sig")
    with (OUTPUT_DIR / "summary_metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"config": asdict(config), "summaries": summaries}, f, ensure_ascii=False, indent=2)

    print("Teammate original-formula validation")
    for name, summary in summaries.items():
        print(f"\n[{name}]")
        print("PCA weights:", summary["pca_weights"])
        print("Gasoline:", summary["metrics"]["gasoline"])
        print("Diesel:", summary["metrics"]["diesel"])
        print("NARDL-like gasoline:", summary["nardl_like"]["gasoline"])
        print("NARDL-like diesel:", summary["nardl_like"]["diesel"])
    print(f"\nOutputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
