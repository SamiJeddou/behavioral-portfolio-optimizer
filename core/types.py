# © 2026 Sami Jeddou. All rights reserved.
# Published publicly for demonstration and evaluation only — no license is granted.
# Copying, modification, redistribution, or reuse (in whole or in part) without the
# author's prior written permission is prohibited.
"""Typed boundary shared by every engine and adapter.

These dataclasses are the *stable public surface*. Internals (the grid solver, the
CVaR LP, the existing dicts) can change freely as long as they still map to these.
No Streamlit, no I/O — pure data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import math
import numpy as np


def _jsonable(x):
    """JSON-safe scalar: non-finite floats (NaN/inf) -> None, numpy floats -> float."""
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return x if math.isfinite(x) else None
    if isinstance(x, (int, np.integer)):
        return int(x)
    return x


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
            "weights": [_jsonable(x) for x in self.weights],
            "expected_return": _jsonable(self.expected_return),
            "std_dev": _jsonable(self.std_dev),
            "shortfall_stat": _jsonable(self.shortfall_stat),
            "feasible": bool(self.feasible),
            "meta": {k: _jsonable(v) for k, v in self.meta.items()},
        }


@dataclass
class FrontierPoint:
    x: float                   # risk axis (std-dev % for grid, CVaR floor % for scenario)
    y: float                   # max expected return (%)
    label: str = ""
    feasible: bool = True

    def to_dict(self) -> dict:
        return {"x": _jsonable(self.x), "y": _jsonable(self.y),
                "label": self.label, "feasible": bool(self.feasible)}


@dataclass
class BacktestResult:
    p1: PortfolioResult                       # no-derivative (construction optimum)
    p2: PortfolioResult                       # with-derivative
    underlying: str                           # derivative underlying name
    derivative_weight: float                  # optimal weight on the derivative in P2
    realised: dict                            # expected vs realised return/vol, breach flags
    paths: dict                               # {'dates':[...], 'pv1':[...], 'pv2':[...]}
    verdict: list                             # plain-language summary lines
    alpha_beta: dict | None = None            # per-portfolio & per-security beta/alpha/r2 vs benchmark

    def to_dict(self) -> dict:
        return {
            "p1": self.p1.to_dict(),
            "p2": self.p2.to_dict(),
            "underlying": self.underlying,
            "derivative_weight": _jsonable(self.derivative_weight),
            "realised": {k: {kk: _jsonable(vv) for kk, vv in v.items()}
                         if isinstance(v, dict) else _jsonable(v)
                         for k, v in self.realised.items()},
            "paths": {"dates": list(self.paths.get("dates", [])),
                      "pv1": [_jsonable(x) for x in self.paths.get("pv1", [])],
                      "pv2": [_jsonable(x) for x in self.paths.get("pv2", [])]},
            "verdict": list(self.verdict),
            "alpha_beta": self.alpha_beta,
        }
