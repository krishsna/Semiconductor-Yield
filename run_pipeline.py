"""
run_pipeline.py
================
Driver script that ties together:
  - wafer_data_generator.py  (Step 1)
  - wafer_spc_analysis.py    (Step 2)
  - wafer_early_warning.py   (Step 4)

Put all four files in the same folder and run:
    python3 run_pipeline.py
"""

from wafer_data_generator import generate_lot
from wafer_spc_analysis import (
    compute_yield_per_wafer,
    compute_yield_per_lot,
    compute_spc_summary,
    plot_wafer_map,
    plot_spc_trends,
    generate_process_health_report,
)
from wafer_early_warning import generate_early_warning_report


def main():
    # 1. Generate synthetic lot data
    df = generate_lot(lot_id="LOT001", num_wafers=8, grid_size=32, seed=42)
    print(f"Generated {len(df)} dies across {df['wafer_id'].nunique()} wafers\n")

    # 2. Yield + SPC summary
    yield_per_wafer = compute_yield_per_wafer(df)
    lot_yield = compute_yield_per_lot(df)
    summary = compute_spc_summary(df)

    print("Yield per wafer:\n", yield_per_wafer, "\n")
    print("Lot yield:", lot_yield, "\n")
    print("SPC summary:\n", summary[["wafer_id", "mean", "std", "Cp", "Cpk", "yield"]], "\n")

    # 3. Wafer maps (worst + best yield)
    worst = summary.loc[summary["yield"].idxmin(), "wafer_id"]
    best = summary.loc[summary["yield"].idxmax(), "wafer_id"]
    plot_wafer_map(df, worst, summary, save_path=f"wafer_map_{worst}.png")
    plot_wafer_map(df, best, summary, save_path=f"wafer_map_{best}.png")
    print(f"Saved wafer maps: wafer_map_{worst}.png, wafer_map_{best}.png\n")

    # 4. SPC trend charts
    plot_spc_trends(summary, save_path_prefix="lot001_spc")
    print("Saved SPC trend charts: lot001_spc_vth_trend.png, lot001_spc_cpk_trend.png\n")

    # 5. Process health report (Step 2)
    print(generate_process_health_report(df, summary))
    print()

    # 6. Early warning report (Step 4)
    print(generate_early_warning_report(df, summary))


if __name__ == "__main__":
    main()
