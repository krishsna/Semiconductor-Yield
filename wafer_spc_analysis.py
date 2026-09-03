"""
Wafer Yield / SPC / Wafer-Map Analysis Layer
=============================================

Takes the die-level DataFrame produced by `wafer_data_generator.generate_lot()`
and produces:
  1. Yield metrics (per-wafer, per-lot, distribution across wafers)
  2. SPC / capability metrics (mean, std, Cp, Cpk per wafer)
  3. Wafer map visualizations (pass/fail scatter)
  4. SPC trend charts across wafers (Vth mean, Cpk) with control limits
  5. A rule-based "Process Health Report" diagnostic layer

This module does NOT touch data generation and does NOT use any ML --
every metric here is a closed-form statistical calculation.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe backend; figures are saved to disk
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# 1. Yield analysis
# ----------------------------------------------------------------------

def compute_yield_per_wafer(df: pd.DataFrame) -> pd.DataFrame:
    """Per-wafer die count, pass count, and yield percentage."""
    grouped = df.groupby("wafer_id", sort=True)
    out = grouped["pass_fail"].agg(
        n_dies="count",
        n_pass=lambda s: (s == "PASS").sum(),
    ).reset_index()
    out["yield_pct"] = 100 * out["n_pass"] / out["n_dies"]
    return out


def compute_yield_per_lot(df: pd.DataFrame) -> dict:
    """Overall lot-level yield (all wafers combined)."""
    n_dies = len(df)
    n_pass = (df["pass_fail"] == "PASS").sum()
    return {
        "lot_id": df["lot_id"].iloc[0],
        "n_dies": int(n_dies),
        "n_pass": int(n_pass),
        "yield_pct": 100 * n_pass / n_dies,
    }


def compute_yield_distribution(yield_per_wafer_df: pd.DataFrame) -> dict:
    """Spread of yield across wafers -- flags whether yield is consistent
    wafer-to-wafer or volatile."""
    y = yield_per_wafer_df["yield_pct"]
    return {
        "mean_yield": y.mean(),
        "std_yield": y.std(ddof=1),
        "min_yield": y.min(),
        "max_yield": y.max(),
        "range_yield": y.max() - y.min(),
    }


# ----------------------------------------------------------------------
# 2. SPC / capability metrics
# ----------------------------------------------------------------------

def compute_cp_cpk(mean: float, std: float, lsl: float, usl: float) -> tuple:
    """Standard process-capability formulas."""
    if std == 0:
        return np.inf, np.inf
    cp = (usl - lsl) / (6 * std)
    cpk = min((usl - mean) / (3 * std), (mean - lsl) / (3 * std))
    return cp, cpk


def compute_spc_summary(
    df: pd.DataFrame, lsl: float = 0.95, usl: float = 1.05
) -> pd.DataFrame:
    """
    Build the required summary table:
        wafer_id | mean | std | Cp | Cpk | yield
    one row per wafer, based on each wafer's Vth distribution.
    """
    yield_df = compute_yield_per_wafer(df)

    rows = []
    for wafer_id, g in df.groupby("wafer_id", sort=True):
        mean = g["Vth"].mean()
        std = g["Vth"].std(ddof=1)
        cp, cpk = compute_cp_cpk(mean, std, lsl, usl)
        rows.append({"wafer_id": wafer_id, "mean": mean, "std": std, "Cp": cp, "Cpk": cpk})

    summary = pd.DataFrame(rows)
    summary = summary.merge(yield_df[["wafer_id", "yield_pct"]], on="wafer_id")
    summary = summary.rename(columns={"yield_pct": "yield"})
    summary = summary.sort_values("wafer_id").reset_index(drop=True)
    summary["wafer_index"] = range(1, len(summary) + 1)
    return summary


# ----------------------------------------------------------------------
# 3. Wafer map visualization
# ----------------------------------------------------------------------

def plot_wafer_map(
    df: pd.DataFrame,
    wafer_id: str,
    summary_df: pd.DataFrame = None,
    lsl: float = 0.95,
    usl: float = 1.05,
    save_path: str = None,
):
    """
    Scatter plot of one wafer's dies at their (x, y) positions:
    green = PASS, red = FAIL. Title reports wafer yield + Cpk.
    """
    wafer_df = df[df["wafer_id"] == wafer_id]
    if wafer_df.empty:
        raise ValueError(f"No data found for wafer_id={wafer_id!r}")

    # Pull yield/Cpk from the summary table if given, else compute on the fly
    if summary_df is not None and wafer_id in summary_df["wafer_id"].values:
        row = summary_df[summary_df["wafer_id"] == wafer_id].iloc[0]
        yield_pct, cpk = row["yield"], row["Cpk"]
    else:
        yield_pct = 100 * (wafer_df["pass_fail"] == "PASS").mean()
        _, cpk = compute_cp_cpk(wafer_df["Vth"].mean(), wafer_df["Vth"].std(ddof=1), lsl, usl)

    colors = np.where(wafer_df["pass_fail"] == "PASS", "green", "red")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(wafer_df["x"], wafer_df["y"], c=colors, s=18, edgecolors="none")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Wafer {wafer_id} | Yield: {yield_pct:.1f}% | Cpk: {cpk:.2f}")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path
    return fig


# ----------------------------------------------------------------------
# 4. SPC trend across wafers
# ----------------------------------------------------------------------

def _control_limits(values: pd.Series) -> tuple:
    """Classic Xbar-style control limits: center +/- 3 sigma, using the
    lot's own wafer-to-wafer values as the baseline."""
    center = values.mean()
    sigma = values.std(ddof=1)
    return center, center + 3 * sigma, center - 3 * sigma


def plot_spc_trends(summary_df: pd.DataFrame, save_path_prefix: str = None):
    """
    Two trend charts across wafers in the lot:
      - mean Vth vs wafer index, with UCL/LCL from lot baseline
      - Cpk vs wafer index, with UCL/LCL from lot baseline
    Returns (fig_vth, fig_cpk) or saved file paths if save_path_prefix given.
    """
    x = summary_df["wafer_index"]

    # --- Vth mean trend ---
    center_v, ucl_v, lcl_v = _control_limits(summary_df["mean"])
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(x, summary_df["mean"], "o-", color="steelblue", label="Wafer mean Vth")
    ax1.axhline(center_v, color="gray", linestyle="-", label="Center line")
    ax1.axhline(ucl_v, color="red", linestyle="--", label="UCL")
    ax1.axhline(lcl_v, color="red", linestyle="--", label="LCL")
    ax1.set_xlabel("Wafer index")
    ax1.set_ylabel("Mean Vth (V)")
    ax1.set_title("SPC Trend: Mean Vth Across Wafers")
    ax1.legend(fontsize=8)

    # --- Cpk trend ---
    center_c, ucl_c, lcl_c = _control_limits(summary_df["Cpk"])
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.plot(x, summary_df["Cpk"], "o-", color="darkorange", label="Wafer Cpk")
    ax2.axhline(center_c, color="gray", linestyle="-", label="Center line")
    ax2.axhline(ucl_c, color="red", linestyle="--", label="UCL")
    ax2.axhline(lcl_c, color="red", linestyle="--", label="LCL")
    ax2.axhline(1.33, color="green", linestyle=":", label="Cpk=1.33 (typical target)")
    ax2.set_xlabel("Wafer index")
    ax2.set_ylabel("Cpk")
    ax2.set_title("SPC Trend: Cpk Across Wafers")
    ax2.legend(fontsize=8)

    if save_path_prefix:
        path1 = f"{save_path_prefix}_vth_trend.png"
        path2 = f"{save_path_prefix}_cpk_trend.png"
        fig1.savefig(path1, dpi=150, bbox_inches="tight")
        fig2.savefig(path2, dpi=150, bbox_inches="tight")
        plt.close(fig1)
        plt.close(fig2)
        return path1, path2
    return fig1, fig2


# ----------------------------------------------------------------------
# 5. Process insight / diagnostic layer
# ----------------------------------------------------------------------

def add_edge_distance(df: pd.DataFrame, edge_band_fraction: float = 0.15) -> pd.DataFrame:
    """
    Tag each die as edge/interior based on its distance from its own
    wafer's center, inferred from that wafer's x/y extent (no dependency
    on generator internals).
    """
    df = df.copy()
    dist = pd.Series(index=df.index, dtype=float)
    is_edge = pd.Series(index=df.index, dtype=bool)

    for wafer_id, g in df.groupby("wafer_id"):
        cx = (g["x"].min() + g["x"].max()) / 2
        cy = (g["y"].min() + g["y"].max()) / 2
        d = np.sqrt((g["x"] - cx) ** 2 + (g["y"] - cy) ** 2)
        radius = d.max()
        dist.loc[g.index] = d
        is_edge.loc[g.index] = d >= radius * (1 - edge_band_fraction)

    df["dist_from_center"] = dist
    df["is_edge"] = is_edge
    return df


def diagnose_edge_failures(df: pd.DataFrame, edge_band_fraction: float = 0.15, factor: float = 1.5) -> str:
    """Flag if failures are disproportionately concentrated at the wafer edge."""
    tagged = add_edge_distance(df, edge_band_fraction)
    edge_fail_rate = 1 - (tagged.loc[tagged["is_edge"], "pass_fail"] == "PASS").mean()
    interior_fail_rate = 1 - (tagged.loc[~tagged["is_edge"], "pass_fail"] == "PASS").mean()

    if interior_fail_rate == 0 and edge_fail_rate > 0:
        flagged = True
    else:
        flagged = edge_fail_rate > factor * interior_fail_rate and edge_fail_rate > 0.01

    if flagged:
        return (
            f"EDGE EFFECT DETECTED: edge-die fail rate ({edge_fail_rate:.1%}) is notably "
            f"higher than interior fail rate ({interior_fail_rate:.1%}). "
            "Possible edge etch / deposition non-uniformity -- inspect edge-ring process steps."
        )
    return None


def diagnose_cpk_drift(summary_df: pd.DataFrame, slope_threshold: float = -0.02) -> str:
    """Flag a consistent downward trend in Cpk across wafers (process drift)."""
    if len(summary_df) < 3:
        return None  # not enough points for a meaningful trend

    slope, _ = np.polyfit(summary_df["wafer_index"], summary_df["Cpk"], 1)
    if slope <= slope_threshold:
        return (
            f"CPK DRIFT DETECTED: Cpk is declining across the lot at a rate of "
            f"{slope:.3f} per wafer. Process drift suspected -- check equipment "
            "calibration / consumable wear over the lot run."
        )
    return None


def diagnose_leakage_variance(df: pd.DataFrame, factor: float = 2.0) -> str:
    """Flag wafers whose leakage spread is unusually large vs the lot's median."""
    leak_std = df.groupby("wafer_id")["leakage"].std(ddof=1)
    median_std = leak_std.median()
    flagged = leak_std[leak_std > factor * median_std]

    if not flagged.empty:
        wafers = ", ".join(flagged.index.tolist())
        return (
            f"HIGH LEAKAGE VARIANCE DETECTED on wafer(s) {wafers} "
            f"(std > {factor:.0f}x lot median of {median_std:.2f} nA). "
            "Contamination or random point-defect increase suspected."
        )
    return None


def generate_process_health_report(df: pd.DataFrame, summary_df: pd.DataFrame) -> str:
    """Combine all diagnostics into one readable per-lot report."""
    lot_yield = compute_yield_per_lot(df)
    yield_dist = compute_yield_distribution(compute_yield_per_wafer(df))

    findings = [
        diagnose_edge_failures(df),
        diagnose_cpk_drift(summary_df),
        diagnose_leakage_variance(df),
    ]
    findings = [f for f in findings if f]

    lines = []
    lines.append("=" * 60)
    lines.append(f"PROCESS HEALTH REPORT -- Lot {lot_yield['lot_id']}")
    lines.append("=" * 60)
    lines.append(f"Lot yield: {lot_yield['yield_pct']:.2f}% "
                  f"({lot_yield['n_pass']}/{lot_yield['n_dies']} dies)")
    lines.append(f"Wafer yield spread: {yield_dist['min_yield']:.1f}% - "
                  f"{yield_dist['max_yield']:.1f}% "
                  f"(std {yield_dist['std_yield']:.2f} pts)")
    lines.append(f"Lot avg Cpk: {summary_df['Cpk'].mean():.2f} "
                  f"(min {summary_df['Cpk'].min():.2f}, max {summary_df['Cpk'].max():.2f})")
    lines.append("-" * 60)

    if findings:
        lines.append("FLAGGED ISSUES:")
        for f in findings:
            lines.append(f"  - {f}")
    else:
        lines.append("No significant process anomalies detected. Lot appears stable.")

    lines.append("=" * 60)
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------

if __name__ == "__main__":
    from wafer_data_generator import generate_lot

    df = generate_lot(lot_id="LOT001", num_wafers=8, grid_size=32, seed=42)

    # 1. Yield
    yield_per_wafer = compute_yield_per_wafer(df)
    lot_yield = compute_yield_per_lot(df)
    print("Yield per wafer:\n", yield_per_wafer, "\n")
    print("Lot yield:", lot_yield, "\n")

    # 2. SPC / capability summary table
    summary = compute_spc_summary(df)
    print("SPC summary table:\n", summary[["wafer_id", "mean", "std", "Cp", "Cpk", "yield"]], "\n")

    # 3. Wafer maps (worst- and best-yield wafers)
    worst_wafer = summary.loc[summary["yield"].idxmin(), "wafer_id"]
    best_wafer = summary.loc[summary["yield"].idxmax(), "wafer_id"]
    plot_wafer_map(df, worst_wafer, summary, save_path=f"wafer_map_{worst_wafer}.png")
    plot_wafer_map(df, best_wafer, summary, save_path=f"wafer_map_{best_wafer}.png")
    print(f"Saved wafer maps for {worst_wafer} (worst yield) and {best_wafer} (best yield)\n")

    # 4. SPC trends
    plot_spc_trends(summary, save_path_prefix="lot001_spc")
    print("Saved SPC trend charts\n")

    # 5. Process health report
    report = generate_process_health_report(df, summary)
    print(report)
