"""
Process Early Warning System
=============================

Sits on top of the existing yield/SPC analytics (wafer_data_generator.py,
wafer_spc_analysis.py) WITHOUT modifying either. Its job is to catch
*leading* indicators of process degradation -- trends across wafers --
rather than just reporting the current snapshot.

Every signal here is a transparent, rule-based check on a measurable
trend (a slope and/or a run of consecutive moves in one direction).
There is no ML, no neural net, and no opaque scoring: the final risk
level is a simple, documented sum of which rules fired.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass

from wafer_spc_analysis import add_edge_distance  # reused, not modified


# ----------------------------------------------------------------------
# Configuration: every threshold used by the rules lives here, so the
# logic is auditable and tunable without touching the detection code.
# ----------------------------------------------------------------------

@dataclass
class EarlyWarningConfig:
    lsl: float = 0.95
    usl: float = 1.05

    # 1. Cpk degradation
    cpk_slope_threshold: float = -0.03     # Cpk points lost per wafer -> flag
    cpk_consecutive_min: int = 3           # consecutive declining wafers -> flag
    cpk_severe_slope: float = -0.08        # steeper than this -> "severe"

    # 2. Mean Vth drift toward a spec limit (tracked via shrinking margin
    #    to the nearer of LSL/USL)
    margin_slope_threshold: float = -0.0008   # volts of margin lost per wafer
    margin_consecutive_min: int = 3
    margin_severe_fraction: float = 0.15      # margin < 15% of (USL-LSL) -> "severe"

    # 3. Variance (std) growth
    std_slope_threshold: float = 0.0004    # volts of std gained per wafer
    std_consecutive_min: int = 3

    # 4. Edge failure escalation (rate among edge-zone dies only)
    edge_rate_slope_threshold: float = 0.004   # fraction-points per wafer
    edge_consecutive_min: int = 3
    edge_band_fraction: float = 0.15

    # 5. Yield-vs-capability mismatch
    yield_stable_min: float = 90.0         # yield must stay above this to call it "high"
    yield_slope_tolerance: float = -0.3    # yield can drop this much/wafer and still count as "not yet declining"


# ----------------------------------------------------------------------
# Small trend helpers (the only math primitives the rules need)
# ----------------------------------------------------------------------

def linear_trend_slope(values: np.ndarray) -> float:
    """Least-squares slope of a sequence vs its index (per-wafer rate of change)."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return 0.0
    slope, _ = np.polyfit(np.arange(len(values)), values, 1)
    return slope


def longest_trailing_run(values: np.ndarray, direction: str) -> int:
    """
    Length of the most recent unbroken run of step-to-step moves in the
    given direction ('increase' or 'decrease'). Used to catch a clear,
    ongoing trend even when the overall slope is diluted by earlier data.
    """
    values = np.asarray(values, dtype=float)
    diffs = np.diff(values)
    cond = diffs < 0 if direction == "decrease" else diffs > 0
    run = 0
    for d in cond[::-1]:
        if d:
            run += 1
        else:
            break
    return run


# ----------------------------------------------------------------------
# Signal 1: Cpk degradation
# ----------------------------------------------------------------------

def detect_cpk_degradation(summary_df: pd.DataFrame, cfg: EarlyWarningConfig) -> dict:
    s = summary_df.sort_values("wafer_index")
    cpk = s["Cpk"].values
    slope = linear_trend_slope(cpk)
    run = longest_trailing_run(cpk, "decrease")

    flagged = (slope <= cfg.cpk_slope_threshold) or (run >= cfg.cpk_consecutive_min)
    severity = None
    message = None
    if flagged:
        severity = "severe" if slope <= cfg.cpk_severe_slope else "moderate"
        message = (
            f"Cpk decreasing across wafers (slope {slope:+.3f}/wafer, {run} "
            f"consecutive declines, latest Cpk {cpk[-1]:.2f}). Capability "
            "degradation detected. Process moving toward specification limits."
        )

    return {"name": "cpk_degradation", "flagged": flagged, "severity": severity,
            "slope": slope, "run": run, "message": message}


# ----------------------------------------------------------------------
# Signal 2: Mean Vth drift toward a spec limit
# ----------------------------------------------------------------------

def detect_mean_drift(summary_df: pd.DataFrame, cfg: EarlyWarningConfig) -> dict:
    s = summary_df.sort_values("wafer_index")
    means = s["mean"].values
    # distance from each wafer's mean to the NEARER spec limit
    margin = np.minimum(cfg.usl - means, means - cfg.lsl)

    slope = linear_trend_slope(margin)        # negative = margin shrinking = drifting toward a limit
    run = longest_trailing_run(margin, "decrease")

    flagged = (slope <= cfg.margin_slope_threshold) or (run >= cfg.margin_consecutive_min)
    severity = None
    message = None
    if flagged:
        nearer = "USL" if (cfg.usl - means[-1]) < (means[-1] - cfg.lsl) else "LSL"
        spec_width = cfg.usl - cfg.lsl
        severity = "severe" if margin[-1] < cfg.margin_severe_fraction * spec_width else "moderate"
        message = (
            f"Process mean Vth drifting toward {nearer} (margin shrinking at "
            f"{slope:+.4f} V/wafer, {run} consecutive shrinks, current margin "
            f"{margin[-1]:.3f} V). Process mean drifting toward specification boundary."
        )

    return {"name": "mean_drift", "flagged": flagged, "severity": severity,
            "slope": slope, "run": run, "message": message}


# ----------------------------------------------------------------------
# Signal 3: Variance growth
# ----------------------------------------------------------------------

def detect_variance_growth(summary_df: pd.DataFrame, cfg: EarlyWarningConfig) -> dict:
    s = summary_df.sort_values("wafer_index")
    std = s["std"].values
    slope = linear_trend_slope(std)
    run = longest_trailing_run(std, "increase")

    flagged = (slope >= cfg.std_slope_threshold) or (run >= cfg.std_consecutive_min)
    severity = None
    message = None
    if flagged:
        severity = "severe" if slope >= 2 * cfg.std_slope_threshold else "moderate"
        message = (
            f"Vth standard deviation increasing across wafers (slope {slope:+.4f} "
            f"V/wafer, {run} consecutive increases, latest std {std[-1]:.4f} V). "
            "Process variability increasing. Future capability reduction likely."
        )

    return {"name": "variance_growth", "flagged": flagged, "severity": severity,
            "slope": slope, "run": run, "message": message}


# ----------------------------------------------------------------------
# Signal 4: Edge failure escalation
# ----------------------------------------------------------------------

def _edge_fail_rate_series(df: pd.DataFrame, summary_df: pd.DataFrame, cfg: EarlyWarningConfig) -> np.ndarray:
    """Per-wafer fail rate computed ONLY among edge-zone dies, ordered to match summary_df."""
    tagged = add_edge_distance(df, edge_band_fraction=cfg.edge_band_fraction)
    edge_only = tagged[tagged["is_edge"]]
    rates = edge_only.groupby("wafer_id")["pass_fail"].apply(lambda s: (s == "FAIL").mean())

    ordered = summary_df.sort_values("wafer_index").set_index("wafer_id")
    ordered = ordered.join(rates.rename("edge_fail_rate"))["edge_fail_rate"].fillna(0.0)
    return ordered.values


def detect_edge_escalation(df: pd.DataFrame, summary_df: pd.DataFrame, cfg: EarlyWarningConfig) -> dict:
    rates = _edge_fail_rate_series(df, summary_df, cfg)
    slope = linear_trend_slope(rates)
    run = longest_trailing_run(rates, "increase")

    flagged = (slope >= cfg.edge_rate_slope_threshold) or (run >= cfg.edge_consecutive_min)
    severity = None
    message = None
    if flagged:
        severity = "severe" if slope >= 2 * cfg.edge_rate_slope_threshold else "moderate"
        message = (
            f"Edge-zone fail rate increasing across wafers (slope {slope:+.3%}/wafer, "
            f"{run} consecutive increases, latest edge fail rate {rates[-1]:.1%}). "
            "Edge-related failure growth detected. Possible process non-uniformity."
        )

    return {"name": "edge_escalation", "flagged": flagged, "severity": severity,
            "slope": slope, "run": run, "message": message}


# ----------------------------------------------------------------------
# Signal 5: Yield-vs-capability mismatch (the most important diagnostic)
# ----------------------------------------------------------------------

def detect_yield_capability_mismatch(summary_df: pd.DataFrame, cfg: EarlyWarningConfig,
                                      cpk_signal: dict) -> dict:
    s = summary_df.sort_values("wafer_index")
    yields = s["yield"].values
    yield_slope = linear_trend_slope(yields)

    yield_high = yields.min() >= cfg.yield_stable_min
    yield_not_yet_declining = yield_slope >= cfg.yield_slope_tolerance

    flagged = cpk_signal["flagged"] and yield_high and yield_not_yet_declining
    message = None
    if flagged:
        message = (
            f"Yield remains high (min {yields.min():.1f}%, trend {yield_slope:+.2f} "
            "pts/wafer) while Cpk is deteriorating. Yield currently acceptable, but "
            "capability trend suggests elevated future risk."
        )

    return {"name": "yield_capability_mismatch", "flagged": flagged, "severity": "severe" if flagged else None,
            "yield_slope": yield_slope, "yield_min": yields.min(), "message": message}


# ----------------------------------------------------------------------
# Deterministic risk assessment
# ----------------------------------------------------------------------
#
# Risk score (fully transparent, no randomness, no learned weights):
#   +1 for each of the 4 core signals that fires (cpk, mean drift,
#       variance growth, edge escalation)
#   +1 extra for each of those that fires at "severe" strength
#   +1 if the yield/capability mismatch fires (it is weighted because it
#       represents *hidden* risk that yield numbers alone would miss)
#
#   score 0      -> LOW
#   score 1-2    -> MODERATE
#   score 3-4    -> HIGH
#   score 5+     -> CRITICAL
# ----------------------------------------------------------------------

CORE_SIGNAL_NAMES = ["cpk_degradation", "mean_drift", "variance_growth", "edge_escalation"]


def assess_process_risk(signals: dict) -> dict:
    score = 0
    for name in CORE_SIGNAL_NAMES:
        sig = signals[name]
        if sig["flagged"]:
            score += 1
            if sig.get("severity") == "severe":
                score += 1

    if signals["yield_capability_mismatch"]["flagged"]:
        score += 1

    if score == 0:
        level = "LOW"
    elif score <= 2:
        level = "MODERATE"
    elif score <= 4:
        level = "HIGH"
    else:
        level = "CRITICAL"

    return {"score": score, "level": level}


# ----------------------------------------------------------------------
# Top-level entry point
# ----------------------------------------------------------------------

ASSESSMENT_TEXT = {
    "LOW": "No significant early-warning signals detected. Process appears stable across the monitored wafers.",
    "MODERATE": "Early indicators of process change detected. Recommend continued monitoring; no immediate action required.",
    "HIGH": "Process remains functional but shows evidence of degradation. Investigation recommended before yield impact becomes significant.",
    "CRITICAL": "Multiple compounding warning signals detected. Immediate investigation and corrective action recommended to prevent significant yield loss.",
}


def run_early_warning_checks(df: pd.DataFrame, summary_df: pd.DataFrame,
                              cfg: EarlyWarningConfig = None) -> dict:
    """Run all 5 signals and the risk assessment; returns a structured dict."""
    cfg = cfg or EarlyWarningConfig()

    cpk_signal = detect_cpk_degradation(summary_df, cfg)
    mean_signal = detect_mean_drift(summary_df, cfg)
    var_signal = detect_variance_growth(summary_df, cfg)
    edge_signal = detect_edge_escalation(df, summary_df, cfg)
    mismatch_signal = detect_yield_capability_mismatch(summary_df, cfg, cpk_signal)

    signals = {
        "cpk_degradation": cpk_signal,
        "mean_drift": mean_signal,
        "variance_growth": var_signal,
        "edge_escalation": edge_signal,
        "yield_capability_mismatch": mismatch_signal,
    }
    risk = assess_process_risk(signals)
    return {"signals": signals, "risk": risk}


def generate_early_warning_report(df: pd.DataFrame, summary_df: pd.DataFrame,
                                   cfg: EarlyWarningConfig = None) -> str:
    """Build the human-readable 'PROCESS EARLY WARNING REPORT' text."""
    result = run_early_warning_checks(df, summary_df, cfg)
    signals, risk = result["signals"], result["risk"]

    observed = [s["message"] for s in signals.values() if s["message"]]

    lines = []
    lines.append("=" * 60)
    lines.append("PROCESS EARLY WARNING REPORT")
    lines.append("=" * 60)
    lines.append(f"Risk Level: {risk['level']}  (rule score: {risk['score']})")
    lines.append("-" * 60)

    if observed:
        lines.append("Observed Signals:")
        for msg in observed:
            lines.append(f"  - {msg}")
    else:
        lines.append("Observed Signals: none")

    lines.append("-" * 60)
    lines.append("Assessment:")
    lines.append(ASSESSMENT_TEXT[risk["level"]])
    lines.append("=" * 60)

    return "\n".join(lines)


# ----------------------------------------------------------------------
# Demonstration
# ----------------------------------------------------------------------

if __name__ == "__main__":
    from wafer_data_generator import generate_lot
    from wafer_spc_analysis import compute_spc_summary

    # --- Case A: the normal, unmodified lot from Step 1/2 ---
    print("CASE A: real generated lot (LOT001, seed=42)\n")
    df_normal = generate_lot(lot_id="LOT001", num_wafers=8, grid_size=32, seed=42)
    summary_normal = compute_spc_summary(df_normal)
    print(generate_early_warning_report(df_normal, summary_normal))

    # --- Case B: a synthetic stress-test lot with deliberate, worsening
    # drift -- built independently of the generator, purely to verify the
    # warning system correctly escalates risk when a real drift exists.
    print("\n\nCASE B: synthetic stress-test lot (deliberate drift)\n")

    rng = np.random.default_rng(7)
    grid_size = 20
    radius = grid_size / 2
    coords = [(x, y) for x in range(grid_size) for y in range(grid_size)]
    base_grid = pd.DataFrame(coords, columns=["x", "y"])
    base_grid["dist"] = np.sqrt((base_grid.x - radius) ** 2 + (base_grid.y - radius) ** 2)
    base_grid = base_grid[base_grid["dist"] <= radius].reset_index(drop=True)
    is_edge_mask = base_grid["dist"] >= radius * 0.85

    LSL, USL = 0.95, 1.05
    stress_wafers = []
    for i in range(1, 9):
        wafer_id = f"STRESS-W{i:02d}"
        n = len(base_grid)
        mean_shift = 0.004 * i               # mean walks steadily toward USL
        std_i = 0.018 + 0.0025 * i           # std grows each wafer
        vth = rng.normal(1.0 + mean_shift, std_i, size=n)

        # edge dies get an increasing extra fail probability each wafer
        edge_extra_fail_prob = 0.02 * i
        rand_fail = rng.random(n) < np.where(is_edge_mask, edge_extra_fail_prob, 0.01)
        vth_fail = (vth < LSL) | (vth > USL)
        fail = vth_fail | rand_fail

        wdf = base_grid[["x", "y"]].copy()
        wdf["lot_id"] = "STRESS"
        wdf["wafer_id"] = wafer_id
        wdf["Vth"] = vth
        wdf["leakage"] = rng.lognormal(np.log(5.0), 0.5, size=n)
        wdf["delay"] = 100.0
        wdf["pass_fail"] = np.where(fail, "FAIL", "PASS")
        wdf["bin"] = np.where(fail, 7, 1)
        stress_wafers.append(wdf)

    df_stress = pd.concat(stress_wafers, ignore_index=True)
    df_stress = df_stress[["lot_id", "wafer_id", "x", "y", "Vth", "leakage", "delay", "pass_fail", "bin"]]

    summary_stress = compute_spc_summary(df_stress)
    print(summary_stress[["wafer_id", "mean", "std", "Cp", "Cpk", "yield"]])
    print()
    print(generate_early_warning_report(df_stress, summary_stress))
