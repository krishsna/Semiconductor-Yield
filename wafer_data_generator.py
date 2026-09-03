"""
Synthetic Semiconductor Wafer Test Data Generator
==================================================

Generates realistic die-level wafer probe test data with:
  - Lot -> Wafer -> Die hierarchy
  - Wafer-to-wafer mean shift (process drift between wafers)
  - Die-level random noise
  - Simple edge-effect defect behavior on leakage
  - Spec-limit-based pass/fail labeling and binning

No plotting / dashboards / SPC here -- pure data generation.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


# ----------------------------------------------------------------------
# Configuration: all distribution params and spec limits live here so
# they're easy to tweak without touching the generation logic below.
# ----------------------------------------------------------------------

@dataclass
class WaferTestConfig:
    # Base die-level distributions (before any wafer-level shift)
    vth_mean: float = 1.0          # volts
    vth_std: float = 0.02          # volts
    leakage_mean_log: float = np.log(5.0)   # lognormal -> typical ~5 nA
    leakage_sigma_log: float = 0.5
    delay_mean: float = 100.0      # ps
    delay_std: float = 5.0         # ps
    delay_vth_sensitivity: float = 50.0  # ps shift per volt of Vth deviation

    # Wafer-to-wafer variation: each wafer's Vth mean is shifted randomly
    wafer_shift_std: float = 0.01  # volts

    # Spec limits used for pass/fail labeling
    vth_lsl: float = 0.95
    vth_usl: float = 1.05
    leakage_max_na: float = 50.0

    # Simple spatial defect behavior: dies near the wafer edge tend to
    # show elevated leakage (a common real-world wafer effect)
    edge_leakage_multiplier: float = 1.8
    edge_band_fraction: float = 0.15  # outer 15% of radius counts as "edge"

    # Simple random point-defect behavior: a small fraction of dies get
    # a large leakage spike (e.g. particle contamination), independent
    # of wafer position -- this is what actually drives leakage failures
    defect_rate: float = 0.015          # fraction of dies hit by a defect
    defect_leakage_multiplier_range: tuple = (8.0, 30.0)


# ----------------------------------------------------------------------
# Spatial grid generation
# ----------------------------------------------------------------------

def generate_die_grid(grid_size: int = 32, circular: bool = True) -> pd.DataFrame:
    """
    Build (x, y) die coordinates for one wafer.

    grid_size : dies along each side of the bounding square grid.
    circular  : if True, drop dies outside the wafer's circular boundary
                so the layout resembles a real wafer map; if False,
                keep the full square grid.
    """
    radius = grid_size / 2
    coords = [(x, y) for x in range(grid_size) for y in range(grid_size)]
    df = pd.DataFrame(coords, columns=["x", "y"])

    # distance from wafer center -- used for the circular mask AND
    # later to decide which dies count as "edge" dies
    df["_dist_from_center"] = np.sqrt((df.x - radius) ** 2 + (df.y - radius) ** 2)

    if circular:
        df = df[df["_dist_from_center"] <= radius].reset_index(drop=True)

    return df


# ----------------------------------------------------------------------
# Wafer-level variation
# ----------------------------------------------------------------------

def sample_wafer_mean_shift(rng: np.random.Generator, cfg: WaferTestConfig) -> float:
    """Random per-wafer Vth mean shift, simulating wafer-to-wafer process drift."""
    return rng.normal(0.0, cfg.wafer_shift_std)


# ----------------------------------------------------------------------
# Die-level parameter generation
# ----------------------------------------------------------------------

def generate_die_parameters(
    die_grid: pd.DataFrame,
    wafer_vth_shift: float,
    rng: np.random.Generator,
    cfg: WaferTestConfig,
    radius: float,
) -> pd.DataFrame:
    """
    Populate Vth, leakage, and delay for every die on one wafer, given
    that wafer's random mean shift.
    """
    n = len(die_grid)
    df = die_grid.copy()

    # Vth = global mean + wafer-level shift + die-level Gaussian noise
    df["Vth"] = rng.normal(cfg.vth_mean + wafer_vth_shift, cfg.vth_std, size=n)

    # Leakage: lognormal base (always positive, right-skewed like real
    # leakage current), inflated for dies sitting in the outer edge ring
    base_leakage = rng.lognormal(cfg.leakage_mean_log, cfg.leakage_sigma_log, size=n)
    edge_threshold = radius * (1 - cfg.edge_band_fraction)
    is_edge = df["_dist_from_center"] >= edge_threshold
    multiplier = np.where(is_edge, cfg.edge_leakage_multiplier, 1.0)

    # Random point defects (e.g. particle contamination): a small
    # fraction of dies, anywhere on the wafer, get a large extra spike
    is_defect = rng.random(n) < cfg.defect_rate
    defect_multiplier = rng.uniform(*cfg.defect_leakage_multiplier_range, size=n)
    multiplier = np.where(is_defect, multiplier * defect_multiplier, multiplier)

    df["leakage"] = base_leakage * multiplier  # nA

    # Delay (optional): weakly tied to Vth deviation (higher Vth -> slower)
    # plus its own independent noise
    vth_dev = df["Vth"] - cfg.vth_mean
    df["delay"] = (
        cfg.delay_mean
        + cfg.delay_vth_sensitivity * vth_dev
        + rng.normal(0, cfg.delay_std, size=n)
    )

    return df.drop(columns="_dist_from_center")


# ----------------------------------------------------------------------
# Spec-limit labeling and binning
# ----------------------------------------------------------------------

def apply_spec_and_binning(df: pd.DataFrame, cfg: WaferTestConfig) -> pd.DataFrame:
    """
    Label each die pass/fail against spec limits and assign a bin:
      Bin 1 -> all specs pass
      Bin 7 -> Vth-only failure
      Bin 9 -> leakage-only failure
      Bin 0 -> general fail (multiple specs violated)
    """
    df = df.copy()

    vth_fail = (df["Vth"] < cfg.vth_lsl) | (df["Vth"] > cfg.vth_usl)
    leak_fail = df["leakage"] > cfg.leakage_max_na

    conditions = [
        vth_fail & leak_fail,   # both bad -> general fail
        vth_fail & ~leak_fail,  # Vth only
        ~vth_fail & leak_fail,  # leakage only
    ]
    choices = [0, 7, 9]
    df["bin"] = np.select(conditions, choices, default=1)
    df["pass_fail"] = np.where(df["bin"] == 1, "PASS", "FAIL")

    return df


# ----------------------------------------------------------------------
# Single wafer assembly
# ----------------------------------------------------------------------

def generate_wafer(
    lot_id: str,
    wafer_id: str,
    rng: np.random.Generator,
    cfg: WaferTestConfig,
    grid_size: int = 32,
    circular: bool = True,
) -> pd.DataFrame:
    """Generate one full wafer's worth of die-level test data."""
    grid = generate_die_grid(grid_size, circular)
    wafer_shift = sample_wafer_mean_shift(rng, cfg)
    radius = grid_size / 2

    df = generate_die_parameters(grid, wafer_shift, rng, cfg, radius)
    df = apply_spec_and_binning(df, cfg)

    df.insert(0, "wafer_id", wafer_id)
    df.insert(0, "lot_id", lot_id)
    return df


# ----------------------------------------------------------------------
# Full lot assembly (the main entry point)
# ----------------------------------------------------------------------

def generate_lot(
    lot_id: str = "LOT001",
    num_wafers: int = 8,
    grid_size: int = 32,
    circular: bool = True,
    seed: Optional[int] = 42,
    cfg: Optional[WaferTestConfig] = None,
) -> pd.DataFrame:
    """
    Generate a complete synthetic lot: `num_wafers` wafers, each with a
    grid of dies. Reproducible given the same `seed`.
    """
    cfg = cfg or WaferTestConfig()
    rng = np.random.default_rng(seed)  # single shared generator -> reproducible across whole lot

    wafers = []
    for i in range(1, num_wafers + 1):
        wafer_id = f"{lot_id}-W{i:02d}"
        wafer_df = generate_wafer(lot_id, wafer_id, rng, cfg, grid_size, circular)
        wafers.append(wafer_df)

    lot_df = pd.concat(wafers, ignore_index=True)
    return lot_df[
        ["lot_id", "wafer_id", "x", "y", "Vth", "leakage", "delay", "pass_fail", "bin"]
    ]


# ----------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------

if __name__ == "__main__":
    data = generate_lot(lot_id="LOT001", num_wafers=8, grid_size=32, seed=42)

    print(f"Generated {len(data)} dies across {data['wafer_id'].nunique()} wafers")
    print(data.head())
    print("\nBin distribution:")
    print(data["bin"].value_counts().sort_index())
    print(f"\nOverall yield: {(data['pass_fail'] == 'PASS').mean():.2%}")

    data.to_csv("synthetic_wafer_data.csv", index=False)
    print("\nSaved to synthetic_wafer_data.csv")
