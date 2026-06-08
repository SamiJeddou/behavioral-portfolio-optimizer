# © 2026 Sami Jeddou. All rights reserved.
# Published publicly for demonstration and evaluation only — no license is granted.
# Copying, modification, redistribution, or reuse (in whole or in part) without the
# author's prior written permission is prohibited.
"""High-level typed API — the stable public surface that adapters (API/MCP) call.

Each wrapper takes one typed input (AssetUniverse + Constraint [+ DerivativeSpec]) and
returns one typed result (PortfolioResult / FrontierPoint). Internals delegate to the
existing, validated engines in core.grid and core.scenario — no maths is changed here.
"""
from __future__ import annotations

import numpy as np

from core.types import (AssetUniverse, DerivativeSpec, Constraint,
                        PortfolioResult, FrontierPoint)
from core.grid import run_opt
from core.scenario import (mc_generate_scenarios, mc_build_matrix,
                          mc_max_return_cvar, mc_frontier)
from core.pricing import build_der_config

_RES = {"fast": (21, 15), "standard": (35, 50), "high": (51, 99), "turbo": (51, "turbo")}
_KIND = {"var": "var", "es_thesis": "es", "es_rigorous": "es_rigorous"}


def _der_specs(derivatives, sigmas):
    out = []
    for d in derivatives or []:
        out.append({"der_type": d.type, "params": d.params,
                    "underlying_idx": d.underlying_idx, "T": d.T, "r": d.r,
                    "vol_override": d.vol_override, "label": d.label or d.type})
    return out


def optimise_scenario(universe: AssetUniverse, constraint: Constraint,
                      derivatives: list[DerivativeSpec] | None = None,
                      scenarios: int = 10_000, copula: str = "gaussian",
                      dof: int = 5, w_max: float | None = None,
                      seed: int = 0) -> PortfolioResult:
    """Scalable MC + CVaR optimum. The CVaR floor is constraint.L (falls back to H)."""
    floor = constraint.L if constraint.L is not None else constraint.H
    R = mc_generate_scenarios(universe.means, universe.sigmas, universe.corr,
                              S=scenarios, copula=copula, dof=dof, seed=seed)
    Rf, labels, errs = mc_build_matrix(R, _der_specs(derivatives, universe.sigmas),
                                       universe.sigmas, universe.names)
    w, er, es, res = mc_max_return_cvar(Rf, constraint.alpha, floor, w_max=w_max)
    if w is None:
        return PortfolioResult(weights=np.array([]), labels=labels,
                               expected_return=float("nan"), std_dev=float("nan"),
                               shortfall_stat=float("nan"), feasible=False,
                               meta={"reason": "no feasible portfolio for this floor",
                                     "errors": errs})
    port = Rf @ w
    sig = float(port.std()); mn = float(port.mean())
    skew = float((((port - mn) / sig) ** 3).mean()) if sig > 0 else 0.0
    feas = round(es * 100, 2) >= round(floor * 100, 2)
    return PortfolioResult.from_scenario(w, labels, er, es, sig, skewness=skew,
                                         feasible=feas, scenarios=int(scenarios),
                                         copula=copula, floor=floor,
                                         alpha=constraint.alpha, errors=errs)


def scenario_frontier(universe: AssetUniverse, constraint: Constraint,
                      derivatives: list[DerivativeSpec] | None = None,
                      floors: list[float] | None = None, scenarios: int = 10_000,
                      copula: str = "gaussian", w_max: float | None = None,
                      seed: int = 0) -> list[FrontierPoint]:
    """Trace the return / tail-risk frontier (one max-return solve per CVaR floor)."""
    if floors is None:
        floors = [-0.30, -0.25, -0.20, -0.15, -0.10, -0.05]
    R = mc_generate_scenarios(universe.means, universe.sigmas, universe.corr,
                              S=scenarios, copula=copula, seed=seed)
    Rf, _labels, _ = mc_build_matrix(R, _der_specs(derivatives, universe.sigmas),
                                     universe.sigmas, universe.names)
    rows = mc_frontier(Rf, constraint.alpha, floors, w_max=w_max)
    return [FrontierPoint(x=r["L"] * 100,
                          y=(r["er"] * 100 if r["er"] is not None else float("nan")),
                          label=f"L={r['L']:.0%}", feasible=bool(r["ok"]))
            for r in rows]


def optimise_grid(universe: AssetUniverse, constraint: Constraint,
                  derivatives: list[DerivativeSpec] | None = None,
                  resolution: str = "standard") -> PortfolioResult:
    """Exact grid optimum (thesis engine). Uses the first derivative if any are given."""
    m, mp = _RES.get(resolution, _RES["standard"])
    der_config = None
    labels = list(universe.names)
    if derivatives:
        d = derivatives[0]
        params = dict(d.params)
        params.setdefault("vol", d.vol_override if d.vol_override is not None
                          else float(universe.sigmas[d.underlying_idx]))
        params.setdefault("r", d.r); params.setdefault("T", d.T)
        der_config = build_der_config(d.type, params, universe.sigmas, d.underlying_idx)
        labels.append(d.label or d.type)
    res, n = run_opt(universe.means, universe.sigmas, universe.cov, der_config,
                     constraint.H, constraint.alpha, m, mp,
                     constraint_type=_KIND.get(constraint.kind, "var"), L=constraint.L)
    return PortfolioResult.from_grid(res, labels[:len(res["weights"])], feasible=True)


__all__ = ["optimise_scenario", "scenario_frontier", "optimise_grid"]
