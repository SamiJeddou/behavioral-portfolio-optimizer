# © 2026 Sami Jeddou. All rights reserved.
# Published publicly for demonstration and evaluation only — no license is granted.
"""Risk-profiling module — UI-free.

Implements the validated Grable & Lytton (1999) 13-item financial risk-tolerance
scale, plus a transparent, documented mapping from the resulting tolerance band to
this app's simulation parameters (threshold H and shortfall probability alpha).

  Grable, J. E., & Lytton, R. H. (1999). Financial risk tolerance revisited: the
  development of a risk assessment instrument. Financial Services Review, 8, 163-181.

IMPORTANT — this is NOT investment advice and NOT a regulated suitability assessment.
The band -> (H, alpha) mapping is a heuristic that sets the *simulation's* risk
parameters only. It is intentionally monotonic (greater risk tolerance => deeper
loss floor H and higher tolerated shortfall probability alpha => lower implied
risk-aversion lambda) and is bounded to the optimiser's own slider ranges
(H in [-40, -1] %, alpha in [1, 15] %). No Streamlit, no I/O.
"""

# Each item: the verbatim question and a list of (answer_text, points) options,
# in the published order. Scoring per Grable & Lytton (1999).
GL_ITEMS = [
    {
        "q": "In general, how would your best friend describe you as a risk taker?",
        "options": [
            ("A real gambler", 4),
            ("Willing to take risks after completing adequate research", 3),
            ("Cautious", 2),
            ("A real risk avoider", 1),
        ],
    },
    {
        "q": "You are on a TV game show and can choose one of the following; which would you take?",
        "options": [
            ("$1,000 in cash", 1),
            ("A 50% chance at winning $5,000", 2),
            ("A 25% chance at winning $10,000", 3),
            ("A 5% chance at winning $100,000", 4),
        ],
    },
    {
        "q": ("You have just finished saving for a “once-in-a-lifetime” vacation. "
              "Three weeks before you plan to leave, you lose your job. You would:"),
        "options": [
            ("Cancel the vacation", 1),
            ("Take a much more modest vacation", 2),
            ("Go as scheduled, reasoning that you need the time to prepare for a job search", 3),
            ("Extend your vacation, because this might be your last chance to go first-class", 4),
        ],
    },
    {
        "q": "If you unexpectedly received $20,000 to invest, what would you do?",
        "options": [
            ("Deposit it in a bank account, money market account, or insured CD", 1),
            ("Invest it in safe high-quality bonds or bond mutual funds", 2),
            ("Invest it in stocks or stock mutual funds", 3),
        ],
    },
    {
        "q": "In terms of experience, how comfortable are you investing in stocks or stock mutual funds?",
        "options": [
            ("Not at all comfortable", 1),
            ("Somewhat comfortable", 2),
            ("Very comfortable", 3),
        ],
    },
    {
        "q": "When you think of the word “risk,” which of the following words comes to mind first?",
        "options": [
            ("Loss", 1),
            ("Uncertainty", 2),
            ("Opportunity", 3),
            ("Thrill", 4),
        ],
    },
    {
        "q": ("Some experts are predicting prices of assets such as gold, jewels, collectibles, and real "
              "estate (hard assets) to increase in value; bond prices may fall, however, experts tend to "
              "agree that government bonds are relatively safe. Most of your investment assets are now in "
              "high-interest government bonds. What would you do?"),
        "options": [
            ("Hold the bonds", 1),
            ("Sell the bonds, put half the proceeds into money market accounts, and the other half into hard assets", 2),
            ("Sell the bonds and put the total proceeds into hard assets", 3),
            ("Sell the bonds, put all the money into hard assets, and borrow additional money to buy more", 4),
        ],
    },
    {
        "q": "Given the best and worst case returns of the four investment choices below, which would you prefer?",
        "options": [
            ("$200 gain best case; $0 gain/loss worst case", 1),
            ("$800 gain best case; $200 loss worst case", 2),
            ("$2,600 gain best case; $800 loss worst case", 3),
            ("$4,800 gain best case; $2,400 loss worst case", 4),
        ],
    },
    {
        "q": ("In addition to whatever you own, you have been given $1,000. You are now asked to choose between:"),
        "options": [
            ("A sure gain of $500", 1),
            ("A 50% chance to gain $1,000 and a 50% chance to gain nothing", 3),
        ],
    },
    {
        "q": ("In addition to whatever you own, you have been given $2,000. You are now asked to choose between:"),
        "options": [
            ("A sure loss of $500", 1),
            ("A 50% chance to lose $1,000 and a 50% chance to lose nothing", 3),
        ],
    },
    {
        "q": ("Suppose a relative left you an inheritance of $100,000, stipulating in the will that you invest "
              "ALL the money in ONE of the following choices. Which one would you select?"),
        "options": [
            ("A savings account or money market mutual fund", 1),
            ("A mutual fund that owns stocks and bonds", 2),
            ("A portfolio of 15 common stocks", 3),
            ("Commodities like gold, silver, and oil", 4),
        ],
    },
    {
        "q": "If you had to invest $20,000, which of the following investment choices would you find most appealing?",
        "options": [
            ("60% in low-risk investments, 30% in medium-risk investments, 10% in high-risk investments", 1),
            ("30% in low-risk investments, 40% in medium-risk investments, 30% in high-risk investments", 2),
            ("10% in low-risk investments, 40% in medium-risk investments, 50% in high-risk investments", 3),
        ],
    },
    {
        "q": ("Your trusted friend and neighbor, an experienced geologist, is putting together a group of "
              "investors to fund an exploratory gold mining venture. The venture could pay back 50 to 100 "
              "times the investment if successful. If the mine is a bust, the entire investment is worthless. "
              "Your friend estimates the chance of success is only 20%. If you had the money, how much would "
              "you invest?"),
        "options": [
            ("Nothing", 1),
            ("One month's salary", 2),
            ("Three months' salary", 3),
            ("Six months' salary", 4),
        ],
    },
]

# Published score bands (Grable & Lytton). Direct-sum version (the widely used
# Rutgers / University of Missouri online form): any score >= 33 is "High".
GL_BANDS = [
    (33, 99, "High"),
    (29, 32, "Above-average"),
    (23, 28, "Average / moderate"),
    (19, 22, "Below-average"),
    (0, 18, "Low"),
]

# Documented, monotonic mapping band -> (H %, alpha %), bounded to the optimiser's
# slider ranges. "Average / moderate" sits on the app's existing default (-10 / 5
# region) so the questionnaire feels continuous with manual use.
BAND_TO_HALPHA = {
    "Low":               (-5,  2),
    "Below-average":     (-8,  3),
    "Average / moderate": (-12, 5),
    "Above-average":     (-20, 8),
    "High":              (-30, 12),
}

# Scalable (Monte-Carlo + CVaR) engine uses alpha + a CVaR floor L instead of H.
# L is the tail-average floor, so it sits deeper than the corresponding VaR
# threshold H (CVaR <= VaR). Same alpha as above (shared). Bounded to the
# scalable L slider range [-40, 0] %, with "Average" on its default (-20).
BAND_TO_L = {
    "Low":               -10,
    "Below-average":     -15,
    "Average / moderate": -20,
    "Above-average":     -30,
    "High":              -40,
}

# Short, plain-language descriptors for each band (framing: simulation parameters).
BAND_BLURB = {
    "Low": "You prefer to protect capital. Your simulation uses a shallow loss floor and a strict breach probability.",
    "Below-average": "You lean cautious. Your simulation tolerates only modest downside before the constraint binds.",
    "Average / moderate": "You sit near the middle — the app's base case. A balanced loss floor and shortfall probability.",
    "Above-average": "You are comfortable with meaningful downside in pursuit of return. A deeper floor, looser breach probability.",
    "High": "You tolerate large drawdowns for upside. The deepest floor and highest tolerated shortfall probability in the scale.",
}

N_ITEMS = len(GL_ITEMS)  # 13


def score_responses(choices):
    """Sum the points for a list of chosen answer texts (one per item, in order).

    `choices[i]` must be the selected answer_text for GL_ITEMS[i], or None.
    Raises ValueError if any item is unanswered or a choice is not a valid option."""
    if len(choices) != N_ITEMS:
        raise ValueError(f"expected {N_ITEMS} responses, got {len(choices)}")
    total = 0
    for i, choice in enumerate(choices):
        if choice is None:
            raise ValueError(f"item {i + 1} is unanswered")
        lookup = {text: pts for text, pts in GL_ITEMS[i]["options"]}
        if choice not in lookup:
            raise ValueError(f"item {i + 1}: '{choice}' is not a valid option")
        total += lookup[choice]
    return total


def classify(score):
    """Return the tolerance band label for a summed score."""
    for lo, hi, label in GL_BANDS:
        if lo <= score <= hi:
            return label
    return "Low" if score < 19 else "High"


def profile(choices):
    """Full result for a set of responses: score, band, mapped (H, alpha) as fractions
    and as integer percents, plus the band blurb. H/alpha fractions are negative/positive
    decimals ready for the optimiser (e.g. -0.12, 0.05)."""
    score = score_responses(choices)
    band = classify(score)
    h_pct, a_pct = BAND_TO_HALPHA[band]
    l_pct = BAND_TO_L[band]
    return {
        "score": score,
        "band": band,
        "H_pct": h_pct,
        "alpha_pct": a_pct,
        "L_pct": l_pct,
        "H": h_pct / 100.0,
        "alpha": a_pct / 100.0,
        "L": l_pct / 100.0,
        "blurb": BAND_BLURB[band],
    }


# Theoretical score range, for tests / display.
def _score_bounds():
    lo = sum(min(p for _, p in it["options"]) for it in GL_ITEMS)
    hi = sum(max(p for _, p in it["options"]) for it in GL_ITEMS)
    return lo, hi
