"""Typed boundary shared by every engine and adapter.

These dataclasses are the *stable public surface*. Internals (the grid solver, the
CVaR LP, the existing dicts) can change freely as long as they still map to these.
No Streamlit, no I/O — pure data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np


# ── Inputs ────────────────────────────────────────────────────────────────────
@dataclass
class AssetUniverse:
    """Annualised market inputs for the securities (before any derivatives)."""
    names: list[str]
    means: np.ndarray          # expected annual returns, shape (n,)
    sigmas: np.ndarray         # annual volatilities, shape (n,)
    corr: np.ndarray           # correlation matrix, shape (n, n)

    def __post_init__(self) -> None:
        self.means = np.asarray(self.means, dtype=float)
        self.sigmas = np.asarray(self.sigmas, dtype=float)
        self.corr = np.asarray(self.corr, dtype=float)

    @property
    def n(self) -> int:
        return len(self.names)

    @property
    def cov(self) -> np.ndarray:
        """Covariance implied by sigmas and corr (matches app's corr_to_cov)."""
        d = np.diag(self.sigmas)
        return d @ self.corr @ d


@dataclass
class DerivativeSpec:
    """One derivative / structured-product overlay on an underlying in the universe."""
    type: str                  # 'protective_put' | 'collar' | 'cgn' | 'straddle' | ...
    underlying_idx: int
    params: dict[str, float] = field(default_factory=dict)   # e.g. {'strike': 0.95}
    T: float = 1.0             # option life (years)
    r: float = 0.03            # risk-free rate
    label: str = ""
    vol_override: float | None = None


@dataclass
class Constraint:
    """The downside goal. kind selects the engine's constraint mode."""
    kind: str = "var"          # 'var' | 'es_thesis' | 'es_rigorous'
    H: float = -0.15           # loss threshold (e.g. -0.15 = -15%)
    alpha: float = 0.05        # max shortfall probability (VaR) / CVaR tail fraction
    L: float | None = None     # ES / CVaR floor (required for es_* and scenario engine)

    def __post_init__(self) -> None:
        valid = {"var", "es_thesis", "es_rigorous"}
        if self.kind not in valid:
            raise ValueError(f"Constraint.kind must be one of {sorted(valid)}, got {self.kind!r}")
        if self.kind in {"es_thesis", "es_rigorous"} and self.L is None:
            raise ValueError(f"Constraint.kind={self.kind!r} requires an ES floor L")


# ── Outputs ─────────────────────────────────────────────────────────────────--
@dataclass
class PortfolioResult:
    weights: np.ndarray
    labels: list[str]
    expected_return: float
    std_dev: float
    shortfall_stat: float          # P(r<H) for VaR, or E[r|r<H] for ES/CVaR
    feasible: bool = True
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_grid(cls, d: dict, labels: list[str], feasible: bool = True) -> "PortfolioResult":
        """Map the grid solver's dict (optimize_portfolio / run_opt) onto the boundary."""
        return cls(
            weights=np.asarray(d["weights"], dtype=float),
            labels=list(labels),
            expected_return=float(d["expected_return"]),
            std_dev=float(d["std_dev"]),
            shortfall_stat=float(d["shortfall_stat"]),
            feasible=feasible,
            meta={k: d[k] for k in
                  ("skewness", "excess_kurtosis", "eligible_count", "method_used")
                  if k in d},
        )

    @classmethod
    def from_scenario(cls, weights, labels, expected_return, realised_cvar,
                      std_dev, skewness=None, feasible=True, **extra) -> "PortfolioResult":
        """Map the MC + CVaR engine's outputs (w, er, es, ...) onto the boundary."""
        meta = {"realised_cvar": float(realised_cvar)}
        if skewness is not None:
            meta["skewness"] = float(skewness)
        meta.update(extra)
        return cls(
            weights=np.asarray(weights, dtype=float),
            labels=list(labels),
            expected_return=float(expected_return),
            std_dev=float(std_dev),
            shortfall_stat=float(realised_cvar),
            feasible=feasible,
            meta=meta,
        )

    def to_dict(self) -> dict:
        """JSON-friendly form for an API / MCP response."""
        return {
            "labels": list(self.labels),
            "weights": [float(x) for x in self.weights],
            "expected_return": self.expected_return,
            "std_dev": self.std_dev,
            "shortfall_stat": self.shortfall_stat,
            "feasible": self.feasible,
            "meta": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                     for k, v in self.meta.items()},
        }


@dataclass
class FrontierPoint:
    x: float                   # risk axis (std-dev % for grid, CVaR floor % for scenario)
    y: float                   # max expected return (%)
    label: str = ""
    feasible: bool = True

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "label": self.label, "feasible": self.feasible}
