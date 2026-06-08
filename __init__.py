"""Beyond Mean-Variance — UI-free compute core.

Importing from `core` must never pull in Streamlit. The Streamlit app, a REST API,
an MCP server, or a notebook all import the same engines from here.
"""
from .types import (
    AssetUniverse, DerivativeSpec, Constraint,
    PortfolioResult, FrontierPoint,
)

__all__ = [
    "AssetUniverse", "DerivativeSpec", "Constraint",
    "PortfolioResult", "FrontierPoint",
]
