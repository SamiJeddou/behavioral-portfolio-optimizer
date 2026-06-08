"""Instrument pricing and payoff functions — UI-free.

Black-Scholes leg valuation, derivative configuration, and per-scenario / mark-to-market
payoffs. Re-exports bs_call, bs_put, compute_structured_payoff so callers have one import
for everything pricing-related. No Streamlit.
"""
import numpy as np
from scipy.stats import norm as _norm
from behavioral_portfolio_optimizer import bs_call, bs_put, compute_structured_payoff

__all__ = [
    "bs_call", "bs_put", "compute_structured_payoff",
    "preset_components", "build_der_config", "_bt_legs", "_leg_value",
    "mtm_gross_path", "_mc_leg_intrinsic", "_mc_leg_value_vec", "mc_der_returns",
    "_SYN_UNDERLYING", "_COMPONENT_PRESETS",
]


_SYN_UNDERLYING = [
    {"type": "long_call", "strike": 1.0},
    {"type": "short_put",  "strike": 1.0},
    {"type": "zcb",        "notional": 1.0},
]

_COMPONENT_PRESETS = {
    "bull_call_spread", "bear_put_spread", "butterfly_call", "condor_call",
    "reverse_convertible", "discount_certificate", "outperformance_certificate",
}

def preset_components(der_type, p):
    """Map a preset instrument + its params to a list of pricer components."""
    if der_type == "bull_call_spread":
        return [{"type": "long_call", "strike": p["k1"]},
                {"type": "short_call", "strike": p["k2"]}]
    if der_type == "bear_put_spread":
        return [{"type": "long_put", "strike": p["k1"]},
                {"type": "short_put", "strike": p["k2"]}]
    if der_type == "butterfly_call":
        c, w = p["center"], p["width"]
        return [{"type": "long_call", "strike": c - w},
                {"type": "short_call", "strike": c},
                {"type": "short_call", "strike": c},
                {"type": "long_call", "strike": c + w}]
    if der_type == "condor_call":
        c, wi, wo = p["center"], p["w_in"], p["w_out"]
        return [{"type": "long_call", "strike": c - wo},
                {"type": "short_call", "strike": c - wi},
                {"type": "short_call", "strike": c + wi},
                {"type": "long_call", "strike": c + wo}]
    if der_type == "reverse_convertible":
        return [{"type": "zcb", "notional": 1.0},
                {"type": "short_put", "strike": p["kp"]}]
    if der_type == "discount_certificate":
        return _SYN_UNDERLYING + [{"type": "short_call", "strike": p["kc"]}]
    if der_type == "outperformance_certificate":
        return _SYN_UNDERLYING + [{"type": "long_call", "strike": p["k"]}]
    return []

def build_der_config(der_type, der_params, sigs, underlying_idx):
    # Defaults follow the thesis (vol = underlying's std dev, r = 3%, T = 1y);
    # the sidebar sliders can override each of them.
    base = {"underlying_index": underlying_idx,
            "vol": der_params.get("vol", sigs[underlying_idx]),
            "S0": 1.0,
            "r":  der_params.get("r", 0.03),
            "T":  der_params.get("T", 1.0)}
    if der_type == "put":
        return {**base, "type":"put", "strike":der_params["strike"]}
    elif der_type == "call":
        return {**base, "type":"call", "strike":der_params["strike"]}
    elif der_type == "straddle":
        return {**base, "type":"straddle", "strike":der_params["strike"]}
    elif der_type == "safety_collar":
        return {**base, "type":"safety_collar",
                "strike_p":der_params["strike_p"],"strike_c":der_params["strike_c"]}
    elif der_type == "aggressive_collar":
        return {**base, "type":"aggressive_collar",
                "strike_p":der_params["strike_p"],"strike_c":der_params["strike_c"]}
    elif der_type == "strangle":
        return {**base, "type":"strangle",
                "strike_kp":der_params["strike_kp"],"strike_kc":der_params["strike_kc"]}
    elif der_type == "cgn_uncapped":
        return {**base, "type":"cgn","floor":der_params["floor"],
                "participation":der_params["participation"],
                "cap":None,"cgn_premium":der_params["premium"]}
    elif der_type == "cgn_capped":
        return {**base, "type":"cgn","floor":der_params["floor"],
                "participation":der_params["participation"],
                "cap":der_params["cap"],"cgn_premium":der_params["premium"]}
    elif der_type == "barrier_m":
        return {**base, "type":"barrier_m",
                "M":der_params["M"],"premium_bm":der_params["premium_bm"]}
    elif der_type in _COMPONENT_PRESETS:
        return {**base, "type": "custom",
                "components": preset_components(der_type, der_params),
                "premium": der_params.get("premium", 0.0)}
    elif der_type == "custom":
        return {**base, "type":"custom","components":der_params["components"]}
    return None

def _bt_legs(der_type, der_params):
    """Return (legs, norm_mode, premium) for the backtest, or (None, None, None) if
    unsupported. Strikes are fractions of the entry spot (S0 = 1). norm_mode is
    'gross' for collars (engine divides the return by P0+C0) else 'net'."""
    prem = float(der_params.get("premium", 0.0))
    if der_type == "call":
        return [{"type": "long_call", "strike": der_params["strike"]}], "net", 0.0
    if der_type == "put":
        return [{"type": "long_put", "strike": der_params["strike"]}], "net", 0.0
    if der_type == "straddle":
        K = der_params["strike"]
        return [{"type": "long_call", "strike": K}, {"type": "long_put", "strike": K}], "net", 0.0
    if der_type == "strangle":
        return [{"type": "long_put", "strike": der_params["strike_kp"]},
                {"type": "long_call", "strike": der_params["strike_kc"]}], "net", 0.0
    if der_type == "safety_collar":
        return [{"type": "long_put", "strike": der_params["strike_p"]},
                {"type": "short_call", "strike": der_params["strike_c"]}], "gross", 0.0
    if der_type == "aggressive_collar":
        return [{"type": "long_call", "strike": der_params["strike_c"]},
                {"type": "short_put", "strike": der_params["strike_p"]}], "gross", 0.0
    if der_type in ("cgn_uncapped", "cgn_capped"):
        f = der_params["floor"]; y = der_params["participation"]
        legs = [{"type": "zcb", "notional": 1.0 + f},
                {"type": "long_call", "strike": 1.0 + f, "qty": y}]
        if der_type == "cgn_capped":
            legs.append({"type": "short_call", "strike": 1.0 + der_params["cap"], "qty": y})
        return legs, "net", float(der_params.get("premium", 0.0))
    if der_type in _COMPONENT_PRESETS:
        return preset_components(der_type, der_params), "net", prem
    return None, None, None

def _leg_value(leg, s, remT, vol, r):
    """Signed Black-Scholes value of one leg at spot s with remaining maturity remT.
    Honours an optional 'qty' multiplier (e.g. CGN participation rate)."""
    q = float(leg.get("qty", 1.0))
    t = leg["type"]
    if t == "zcb":
        return q * float(leg.get("notional", 1.0)) * np.exp(-r * remT)
    K = leg["strike"]
    if t == "long_call":  return  q * bs_call(vol, s, r, remT, K)
    if t == "short_call": return -q * bs_call(vol, s, r, remT, K)
    if t == "long_put":   return  q * bs_put(vol, s, r, remT, K)
    if t == "short_put":  return -q * bs_put(vol, s, r, remT, K)
    return 0.0

def mtm_gross_path(legs, norm_mode, prem, spot_path, T, vol, r):
    """Mark-to-market gross-return path of the derivative over a realised underlying
    price path. Entry spot normalised to 1.0; remaining maturity shrinks linearly
    (option entered at the start, expiring at the end). V_t = signed Black-Scholes
    value of the legs. Consistent with the engine's per-instrument return definition:
        g_t = 1 + (V_t - paid) / normalizer
    where paid = fair value * (1 + premium), and normalizer = gross premium (P0+C0)
    for collars, else paid. For net instruments this reduces to V_t / paid."""
    spot_path = np.asarray(spot_path, dtype=float)
    s_rel = spot_path / float(spot_path[0])
    n = len(s_rel)
    leg_v0 = [_leg_value(lg, 1.0, T, vol, r) for lg in legs]
    V0 = float(np.sum(leg_v0))
    gross = float(np.sum(np.abs(leg_v0)))
    paid = V0 * (1.0 + prem)
    normalizer = gross if norm_mode == "gross" else paid
    if abs(normalizer) < 1e-9:
        return None
    remT = np.maximum(T * (1.0 - np.linspace(0.0, 1.0, n)), 1e-6)
    V = np.array([sum(_leg_value(lg, s_rel[i], remT[i], vol, r) for lg in legs)
                  for i in range(n)])
    return 1.0 + (V - paid) / normalizer

def _mc_leg_intrinsic(leg, spot_T):
    """Terminal intrinsic value of a leg at terminal spot (array)."""
    q = float(leg.get("qty", 1.0)); t = leg["type"]
    if t == "zcb":        return q * float(leg.get("notional", 1.0)) * np.ones_like(spot_T)
    K = leg["strike"]
    if t == "long_call":  return  q * np.maximum(spot_T - K, 0.0)
    if t == "short_call": return -q * np.maximum(spot_T - K, 0.0)
    if t == "long_put":   return  q * np.maximum(K - spot_T, 0.0)
    if t == "short_put":  return -q * np.maximum(K - spot_T, 0.0)
    return np.zeros_like(spot_T)

def _mc_leg_value_vec(leg, s, remT, vol, r):
    """Vectorised signed Black-Scholes value of one leg at the horizon: spot array s,
    scalar remaining maturity remT (>0). Mirrors the engine's BS formula so a
    marked-to-market option at the horizon is priced consistently with inception."""
    q = float(leg.get("qty", 1.0)); t = leg["type"]
    s = np.asarray(s, float)
    if t == "zcb":
        return q * float(leg.get("notional", 1.0)) * np.exp(-r * remT) * np.ones_like(s)
    K = leg["strike"]
    if vol <= 0 or remT <= 0:
        return _mc_leg_intrinsic(leg, s)
    s_safe = np.maximum(s, 1e-12)
    sq = vol * np.sqrt(remT)
    d1 = (np.log(s_safe / K) + (r + 0.5 * vol * vol) * remT) / sq
    d2 = d1 - sq
    call = s_safe * _norm.cdf(d1) - K * np.exp(-r * remT) * _norm.cdf(d2)
    put  = K * np.exp(-r * remT) * _norm.cdf(-d2) - s_safe * _norm.cdf(-d1)
    if t == "long_call":  return  q * call
    if t == "short_call": return -q * call
    if t == "long_put":   return  q * put
    if t == "short_put":  return -q * put
    return np.zeros_like(s)

def mc_der_returns(der_type, der_params, und_ret, vol, r=0.03, T=1.0, horizon=1.0):
    """Per-scenario derivative return, priced from its Black-Scholes legs. The option
    is bought at inception (full maturity T) and valued at the optimisation horizon:
    intrinsic if it expires at/before the horizon (T<=horizon), otherwise a
    Black-Scholes mark-to-market with the remaining maturity (T-horizon). Same return
    definition as the engine/backtest."""
    legs, norm_mode, prem = _bt_legs(der_type, der_params)
    if legs is None:
        return None
    leg_v0 = [_leg_value(lg, 1.0, T, vol, r) for lg in legs]
    V0 = float(np.sum(leg_v0)); gross = float(np.sum(np.abs(leg_v0)))
    paid = V0 * (1.0 + prem)
    normalizer = gross if norm_mode == "gross" else paid
    if abs(normalizer) < 1e-9:
        return None
    spot_T = 1.0 + np.asarray(und_ret, float)
    remT = max(float(T) - float(horizon), 0.0)
    if remT <= 1e-9:
        payoff = np.sum([_mc_leg_intrinsic(lg, spot_T) for lg in legs], axis=0)
    else:
        payoff = np.sum([_mc_leg_value_vec(lg, spot_T, remT, vol, r) for lg in legs], axis=0)
    return (payoff - paid) / normalizer
