"""REST adapter — exposes the Beyond Mean-Variance engines over HTTP.

A thin translation layer: JSON request -> typed core dataclasses -> engine wrapper ->
JSON response. No optimisation logic lives here. Run locally with:

    uvicorn api.main:app --reload

Interactive docs (request/response schemas) are auto-published at /docs.
Set BMV_API_KEY in the environment to require an 'x-api-key' header on every call;
leave it unset for an open demo.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from core.types import AssetUniverse, DerivativeSpec, Constraint
from core.optimise import optimise_scenario, scenario_frontier, optimise_grid

app = FastAPI(
    title="Beyond Mean-Variance Portfolio Optimiser API",
    version="1.0",
    description="Goal-based portfolio optimisation with derivatives — exact grid and "
                "scalable Monte-Carlo + CVaR engines. Research/educational; not investment advice.",
)


# ── Request schemas (mirror the core dataclasses) ─────────────────────────────
class UniverseIn(BaseModel):
    names: list[str]
    means: list[float] = Field(..., description="Annualised expected returns")
    sigmas: list[float] = Field(..., description="Annualised volatilities")
    corr: list[list[float]] = Field(..., description="Correlation matrix")


class DerivativeIn(BaseModel):
    type: str = Field(..., examples=["put", "safety_collar", "cgn_capped"])
    underlying_idx: int
    params: dict = Field(default_factory=dict, examples=[{"strike": 0.9}])
    T: float = 1.0
    r: float = 0.03
    label: str = ""
    vol_override: Optional[float] = None


class ConstraintIn(BaseModel):
    kind: str = Field("var", examples=["var", "es_thesis", "es_rigorous"])
    H: float = Field(-0.15, description="Loss threshold")
    alpha: float = Field(0.05, description="Shortfall probability / CVaR tail")
    L: Optional[float] = Field(None, description="ES / CVaR floor (required for es_* and scenario)")


class ScenarioRequest(BaseModel):
    universe: UniverseIn
    constraint: ConstraintIn
    derivatives: list[DerivativeIn] = []
    scenarios: int = 10_000
    copula: str = "gaussian"
    dof: int = 5
    w_max: Optional[float] = None
    seed: int = 0


class FrontierRequest(BaseModel):
    universe: UniverseIn
    constraint: ConstraintIn
    derivatives: list[DerivativeIn] = []
    floors: Optional[list[float]] = None
    scenarios: int = 10_000
    copula: str = "gaussian"
    w_max: Optional[float] = None
    seed: int = 0


class GridRequest(BaseModel):
    universe: UniverseIn
    constraint: ConstraintIn
    derivatives: list[DerivativeIn] = []
    resolution: str = "standard"


# ── Helpers ───────────────────────────────────────────────────────────────────
def _require_key(x_api_key: Optional[str]) -> None:
    expected = os.environ.get("BMV_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing x-api-key")


def _universe(u: UniverseIn) -> AssetUniverse:
    return AssetUniverse(names=u.names, means=np.array(u.means),
                         sigmas=np.array(u.sigmas), corr=np.array(u.corr))


def _constraint(c: ConstraintIn) -> Constraint:
    return Constraint(kind=c.kind, H=c.H, alpha=c.alpha, L=c.L)


def _derivatives(ds: list[DerivativeIn]) -> list[DerivativeSpec]:
    return [DerivativeSpec(type=d.type, underlying_idx=d.underlying_idx, params=d.params,
                           T=d.T, r=d.r, label=d.label, vol_override=d.vol_override)
            for d in ds]


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "engines": ["grid", "scenario"]}


@app.post("/optimise/scenario")
def optimise_scenario_endpoint(req: ScenarioRequest, x_api_key: Optional[str] = Header(None)):
    _require_key(x_api_key)
    try:
        res = optimise_scenario(_universe(req.universe), _constraint(req.constraint),
                                derivatives=_derivatives(req.derivatives),
                                scenarios=req.scenarios, copula=req.copula, dof=req.dof,
                                w_max=req.w_max, seed=req.seed)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return res.to_dict()


@app.post("/optimise/frontier")
def frontier_endpoint(req: FrontierRequest, x_api_key: Optional[str] = Header(None)):
    _require_key(x_api_key)
    try:
        pts = scenario_frontier(_universe(req.universe), _constraint(req.constraint),
                                derivatives=_derivatives(req.derivatives), floors=req.floors,
                                scenarios=req.scenarios, copula=req.copula,
                                w_max=req.w_max, seed=req.seed)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"frontier": [p.to_dict() for p in pts]}


@app.post("/optimise/grid")
def grid_endpoint(req: GridRequest, x_api_key: Optional[str] = Header(None)):
    _require_key(x_api_key)
    try:
        res = optimise_grid(_universe(req.universe), _constraint(req.constraint),
                            derivatives=_derivatives(req.derivatives), resolution=req.resolution)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return res.to_dict()
