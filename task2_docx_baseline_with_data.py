"""Task 2 docx-style baseline with the cleaned data inputs.

This script is a comparison baseline, not the improved main model. It keeps the
five-part welfare loss and all cleaned data inputs from ``task2_dynamic_pricing_dp``,
but solves each pricing window myopically: choose the ratio action that minimizes
the current-period loss only.

Outputs are written under outputs/task2_dynamic_pricing/docx_baseline/.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import task2_dynamic_pricing_dp as improved


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs" / "task2_dynamic_pricing" / "docx_baseline"


def myopic_single_period_strategy(
    df: pd.DataFrame,
    dp: improved.DPResult,
    fuel: improved.Fuel,
) -> pd.DataFrame:
    rows = []
    price_opt = float(df.loc[0, "domestic_price"])
    price_opt_prev2 = price_opt
    opt_action_prev = 0.0
    actual_action_prev = 0.0
    actual_price_prev2 = float(df.loc[0, "domestic_price"] - df.loc[0, "actual_delta"])
    mechanism_action_prev = 0.0
    mechanism_price_prev2 = actual_price_prev2
    cpi_prev = float(df.loc[0, "cpi_yoy"])
    carry_over = 0.0

    for _, row in df.iterrows():
        current_prev_price = float(row["domestic_price"] - row["actual_delta"])
        cur_loss, cur_parts, cpi_cur = improved.normalized_single_period_loss(
            price_prev=current_prev_price,
            action=float(row["actual_delta"]),
            action_prev=actual_action_prev,
            price_prev2=actual_price_prev2,
            oil_price=float(row["oil_price_usd_bbl"]),
            inventory_cost=float(row["inventory_cost"]),
            cpi_prev=cpi_prev,
            quantity=float(row["quantity_tonnes"]),
            calib=dp.calibration,
        )

        mechanism_delta = float(row["current_mechanism_delta"])
        mech_loss, mech_parts, cpi_mech = improved.normalized_single_period_loss(
            price_prev=current_prev_price,
            action=mechanism_delta,
            action_prev=mechanism_action_prev,
            price_prev2=mechanism_price_prev2,
            oil_price=float(row["oil_price_usd_bbl"]),
            inventory_cost=float(row["inventory_cost"]),
            cpi_prev=cpi_prev,
            quantity=float(row["quantity_tonnes"]),
            calib=dp.calibration,
        )

        mechanism_signal = mechanism_delta + carry_over
        best: dict[str, object] | None = None
        for ratio in dp.calibration.ratio_grid:
            action, next_carry, raw_action = improved.action_from_ratio(mechanism_signal, ratio)
            opt_loss, opt_parts, cpi_opt = improved.normalized_single_period_loss(
                price_prev=price_opt,
                action=action,
                action_prev=opt_action_prev,
                price_prev2=price_opt_prev2,
                oil_price=float(row["oil_price_usd_bbl"]),
                inventory_cost=float(row["inventory_cost"]),
                cpi_prev=cpi_prev,
                quantity=float(row["quantity_tonnes"]),
                calib=dp.calibration,
            )
            opt_loss += 0.20 * (next_carry / 500.0) ** 2
            if best is None or opt_loss < float(best["loss"]):
                best = {
                    "ratio": ratio,
                    "raw_action": raw_action,
                    "action": action,
                    "carry": next_carry,
                    "price": max(price_opt + action, 1.0),
                    "loss": opt_loss,
                    "parts": opt_parts,
                    "cpi": cpi_opt,
                }

        if best is None:
            raise RuntimeError("myopic baseline failed to find an action")

        action = float(best["action"])
        opt_parts = best["parts"]
        price_before_action = price_opt
        price_opt = float(best["price"])
        price_opt_prev2 = price_before_action
        opt_action_prev = action
        carry_over = float(best["carry"])
        cpi_prev = float(best["cpi"])

        actual_price_prev2 = current_prev_price
        actual_action_prev = float(row["actual_delta"])
        mechanism_price_prev2 = current_prev_price
        mechanism_action_prev = mechanism_delta

        rows.append(
            {
                "adjust_date": row["adjust_date"],
                "fuel": fuel,
                "scenario": "docx_myopic_with_data",
                "oil_price_usd_bbl": row["oil_price_usd_bbl"],
                "inventory_cost": row["inventory_cost"],
                "quantity_tonnes": row["quantity_tonnes"],
                "transport_share": row["transport_share"],
                "actual_delta": row["actual_delta"],
                "mechanism_delta": mechanism_delta,
                "mechanism_signal_with_carry": mechanism_signal,
                "optimal_ratio": float(best["ratio"]),
                "raw_optimal_delta": float(best["raw_action"]),
                "optimal_delta": action,
                "carry_over_next": carry_over,
                "actual_price": row["domestic_price"],
                "optimal_price": price_opt,
                "actual_loss": cur_loss,
                "mechanism_loss": mech_loss,
                "optimal_loss": float(best["loss"]),
                "actual_cpi_pred": cpi_cur,
                "mechanism_cpi_pred": cpi_mech,
                "optimal_cpi_pred": cpi_prev,
                **{f"actual_loss_{k}": v for k, v in cur_parts.items()},
                **{f"mechanism_loss_{k}": v for k, v in mech_parts.items()},
                **{f"optimal_loss_{k}": v for k, v in opt_parts.items()},
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sims = []
    summaries = []

    for fuel in ["gasoline", "diesel"]:
        df = improved.make_model_data(fuel)  # type: ignore[arg-type]
        dp = improved.initialize_rolling_horizon_model(
            df,
            fuel,  # type: ignore[arg-type]
            scenario="baseline",
            horizon=1,
        )
        sim = myopic_single_period_strategy(df, dp, fuel)  # type: ignore[arg-type]
        sim.to_csv(OUT_DIR / f"task2_docx_myopic_{fuel}_strategy.csv", index=False, encoding="utf-8-sig")
        sims.append(sim)

        summary = improved.summarize(sim)
        summary["scenario"] = "docx_myopic_with_data"
        summary["fuel"] = fuel
        summaries.append(summary)

        calib_payload = {
            "scenario": "docx_myopic_with_data",
            "solver": "single_period_myopic_ratio_search",
            "weights": dp.calibration.weights,
            "ratio_grid": dp.calibration.ratio_grid,
            "transport_share_mean": dp.calibration.transport_share_mean,
            "theta0_cpi": dp.calibration.theta0_cpi,
            "theta1_cpi": dp.calibration.theta1_cpi,
            "security_penalty_mild": dp.calibration.security_penalty_mild,
            "security_penalty_severe": dp.calibration.security_penalty_severe,
        }
        (OUT_DIR / f"task2_docx_myopic_{fuel}_calibration.json").write_text(
            json.dumps(calib_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OUT_DIR / "task2_docx_myopic_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(sims, ignore_index=True).to_csv(
        OUT_DIR / "task2_docx_myopic_all_fuels_strategy.csv", index=False, encoding="utf-8-sig"
    )
    print(summary_df.to_string(index=False))
    print(f"\n已输出目录: {OUT_DIR}")


if __name__ == "__main__":
    main()
