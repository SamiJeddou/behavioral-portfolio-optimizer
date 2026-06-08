"""MCP server — exposes the Beyond Mean-Variance engines as tools for AI agents.

Lets Claude / ChatGPT / any MCP client call the optimiser in natural language. Each
function below becomes a tool; its docstring is the description the model sees.

This file can live anywhere in the repo (e.g. an Extras/ folder); the bootstrap below
locates the repo root (the directory containing core/) and puts it on the import path,
so `from core...` works regardless of where the script is placed or run from.

Run:
    pip install mcp
    python Extras/mcp_server.py          # stdio transport

Add to an MCP client (e.g. Claude Desktop) config:
    {
      "mcpServers": {
        "beyond-mean-variance": {
          "command": "python",
          "args": ["/abs/path/to/Extras/mcp_server.py"]
        }
      }
    }
"""
from __future__ import annotations

import os
import sys

# ── Locate the repo root (the folder containing core/) and add it to sys.path ──
_d = os.path.dirname(os.path.abspath(__file__))
while _d != os.path.dirname(_d):
    if os.path.isdir(os.path.join(_d, "core")):
        if _d not in sys.path:
            sys.path.insert(0, _d)
        break
    _d = os.path.dirname(_d)

from typing import Optional

import numpy as np
from mcp.server.fastmcp import FastMCP

from core.types import AssetUniverse, Constraint, DerivativeSpec
from core.optimise import optimise_scenario, scenario_frontier

mcp = FastMCP("beyond-mean-variance")


def _universe(names, means, sigmas, corr) -> AssetUniverse:
    return AssetUniverse(names=names, means=np.array(means, float),
                         sigmas=np.array(sigmas, float), corr=np.array(corr, float))


def _derivatives(derivatives) -> list[DerivativeSpec]:
    out = []
    for d in derivatives or []:
        out.append(DerivativeSpec(
            type=d["type"], underlying_idx=int(d["underlying_idx"]),
            params=d.get("params", {}), T=float(d.get("T", 1.0)),
            r=float(d.get("r", 0.03)), label=d.get("label", ""),
            vol_override=d.get("vol_override")))
    return out


@mcp.tool()
def optimise_scenario_tool(
    names: list[str], means: list[float], sigmas: list[float], corr: list[list[float]],
    floor: float, alpha: float = 0.05, derivatives: Optional[list[dict]] = None,
    scenarios: int = 10_000, copula: str = "gaussian", w_max: Optional[float] = None,
) -> dict:
    """Find the portfolio that maximises expected return subject to a tail-risk (CVaR) floor,
    using Monte-Carlo scenarios. Scales to many assets and can include derivative overlays.

    names/means/sigmas: per-asset name, annualised expected return and volatility.
    corr: correlation matrix. floor: the worst-tail average return you will accept
    (e.g. -0.20). alpha: tail fraction (default 5%). copula: 'gaussian' or 'student_t'
    (t adds joint crash risk). w_max: optional cap per asset. derivatives: list of
    {type, underlying_idx, params, [T, r, vol_override, label]}.

    Returns weights, expected return, realised CVaR, volatility and feasibility.
    """
    u = _universe(names, means, sigmas, corr)
    con = Constraint(kind="es_thesis", H=floor, alpha=alpha, L=floor)
    return optimise_scenario(u, con, derivatives=_derivatives(derivatives),
                             scenarios=scenarios, copula=copula, w_max=w_max).to_dict()


@mcp.tool()
def trace_frontier_tool(
    names: list[str], means: list[float], sigmas: list[float], corr: list[list[float]],
    alpha: float = 0.05, floors: Optional[list[float]] = None,
    scenarios: int = 10_000, copula: str = "gaussian", w_max: Optional[float] = None,
) -> dict:
    """Trace the return / tail-risk frontier: the maximum expected return achievable at each
    CVaR floor. floors defaults to -30% .. -5%. Returns a list of {x: floor%, y: max return%,
    feasible} points (infeasible floors have y = null).
    """
    u = _universe(names, means, sigmas, corr)
    con = Constraint(kind="es_thesis", H=(floors[0] if floors else -0.20), alpha=alpha,
                     L=(floors[0] if floors else -0.20))
    pts = scenario_frontier(u, con, floors=floors, scenarios=scenarios,
                            copula=copula, w_max=w_max)
    return {"frontier": [p.to_dict() for p in pts]}


@mcp.tool()
def backtest_tool(
    tickers: list[str], construction: list[str], evaluation: list[str], derivative: dict,
    floor: float = -0.15, alpha: float = 0.05, kind: str = "es_thesis",
    benchmark: Optional[str] = None, freq: str = "Daily", rf: float = 0.0,
    resolution: str = "standard",
) -> dict:
    """Out-of-sample backtest: build the optimum on a construction window, hold it through an
    evaluation window (the derivative is marked to market), and compare a no-derivative (P1) and
    with-derivative (P2) portfolio — realised vs expected return/volatility, loss-threshold
    breach, and realised alpha/beta vs an optional benchmark.

    tickers: symbols. construction/evaluation: [start, end] ISO dates. derivative: required dict
    {type, params, [underlying_idx, T, r, vol_override, label]} (underlying_idx -1 = auto-pick the
    highest-volatility security). floor/alpha/kind: the downside constraint used to build the
    portfolios. benchmark: symbol for alpha/beta. resolution: fast | standard | high.
    """
    from core.backtest import run_backtest
    con = Constraint(kind=kind, H=floor, alpha=alpha, L=floor)
    der = _derivatives([derivative])[0]
    return run_backtest(tickers=tickers, construction=(construction[0], construction[1]),
                        evaluation=(evaluation[0], evaluation[1]), constraint=con, derivative=der,
                        benchmark=benchmark, freq=freq, rf=rf, resolution=resolution).to_dict()


if __name__ == "__main__":
    mcp.run()
