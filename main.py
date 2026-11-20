#!/usr/bin/env python3
"""Forecast GTA VI trophy earn rates (Exophase, PS5) at a case target.

Reads every ``*.in`` case in $MAKEFUN_INPUT_DIR (else /input) and writes one
forecast file per case to $MAKEFUN_OUTPUT_DIR (else /output).  One output row
per trophy: five quantiles (5%, 25%, 50%, 75%, 95%) of the trophy's earn
rate, in percent, at the case target.

Model: earner-count decomposition with hierarchical shrinkage
-------------------------------------------------------------
An Exophase earn rate is earners / players-tracked, and both move after a
launch: earners only grow as players progress, while the tracked-player
count explodes and dilutes every rate.  Modelling the rate directly
confuses the two.  So for each trophy we reconstruct approximate earner
counts E = rate * N / 100 from the history, fit log E against log-time
x = ln(1 + hours since launch), shrink each trophy's growth slope toward a
type prior (story chapters grow fast, grinds grow slowly), anchor the
forecast at the last observed count, forecast N the same way, and take
rate(T) = E(T) / N(T).  Story chapters are sequential, so their forecasted
earner counts are forced non-increasing along the chapter order, and the
platinum is forced no larger than any other trophy's count.

Quantiles are normal on the logit scale around the median, with a spread
combining the fit's prediction error, an extrapolation term that grows with
the horizon, and a refresh-shock term: Exophase refreshes move all rates
at once, which a per-trophy random walk cannot see.

Standard library only, no network, deterministic: the same case always
produces the same bytes.
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timezone

LAUNCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Standard-normal quantiles for the five levels, by bisection on Phi.
# (math.erf is exact; no tabulated constants are needed.)


def norm_ppf(p):
    lo, hi = -10.0, 10.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0))) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def logit_pct(rate):
    """logit of an earn rate given in percent, clamped to [0.01, 99.99]."""
    r = min(99.99, max(0.01, rate))
    p = r / 100.0
    return math.log(p / (1.0 - p))


def pct_from_logit(l):
    """Inverse: a logit value back to an earn rate in percent."""
    if l >= 0:
        s = 1.0 / (1.0 + math.exp(-l))
    else:
        e = math.exp(l)
        s = e / (1.0 + e)
    return 100.0 * s


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def x_of(moment):
    """Log-time since launch: ln(1 + hours since 2026-11-19T00:00Z)."""
    hours = (moment - LAUNCH).total_seconds() / 3600.0
    return math.log1p(max(0.0, hours))


def wls(xs, ys, ws):
    """Weighted least squares of y on x.

    Returns (slope, intercept, resid_sd, n, xbar, sxx), or None values when
    the slope cannot be identified (fewer than 2 distinct x).
    """
    n = len(xs)
    if n < 2:
        return None, None, None, n, None, None
    sw = sum(ws)
    xbar = sum(w * x for w, x in zip(ws, xs)) / sw
    ybar = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - xbar) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return None, None, None, n, xbar, sxx
    sxy = sum(w * (x - xbar) * (y - ybar) for w, x, y in zip(ws, xs, ys))
    slope = sxy / sxx
    intercept = ybar - slope * xbar
    if n > 2:
        resid_sd = math.sqrt(
            sum(w * (y - (intercept + slope * x)) ** 2 for w, x, y in zip(ws, xs, ys))
            / (sw * (n - 2) / n)
        )
    else:
        resid_sd = None
    return slope, intercept, resid_sd, n, xbar, sxx


# --- Trophy type priors -------------------------------------------------
# Growth slopes are d ln(earners) / d ln(1 + hours): how fast a trophy's
# earner count grows in log-time.  Calibrated from the launch-night reads
# (638 players tracked) against the live Exophase page ~38 h after launch,
# when the tracked base had grown roughly a hundred-fold and the early
# story and activity trophies had multiplied their earners while the
# grinds had barely moved.  Medians are week-one plateau guesses in the
# same spirit (GTA V's trophy profile a week after launch), used only
# when a case carries no usable history at all.

def classify(trophy):
    name = (trophy.get("name") or "").lower()
    desc = (trophy.get("description") or "").lower()
    tier = (trophy.get("tier") or "").lower()
    if "gone fishing" in desc or "gone fishing" in name:
        return "intro"
    m = re.search(r"complete chapter (\d+)", desc)
    if m:
        return "story_ch%d" % int(m.group(1))
    if "complete the final mission" in desc:
        return "story_final"
    if tier == "platinum":
        return "platinum"
    if "100% completion" in desc or "attain 100%" in desc:
        return "completion"
    if "fishing" in desc or "bowling" in desc or "arcade" in desc or "race" in desc:
        return "activity"
    if "stunt" in desc or "challenge" in desc or "collectible card" in desc or "waink" in desc:
        return "grind"
    if "mission for" in desc or "mission strand" in desc:
        return "side_mission"
    if desc.startswith("rob ") or "bank vault" in desc or "armored truck" in desc:
        return "heist"
    if "earn $" in desc or "spend $" in desc or "laundering" in desc or "purchase" in desc or "register" in desc:
        return "money"
    if "meditate" in desc or "hunt" in desc or "explore" in desc or "discover" in desc:
        return "explore"
    return "misc"


SLOPE_PRIOR = {
    "intro": 0.0,
    "story_ch1": 1.2,
    "story_ch2": 0.7,
    "story_ch3": 0.45,
    "story_ch4": 0.4,
    "story_final": 0.35,
    "platinum": 0.15,
    "completion": 0.15,
    "activity": 0.9,
    "grind": 0.25,
    "side_mission": 0.45,
    "heist": 0.45,
    "money": 0.3,
    "explore": 0.5,
    "misc": 0.45,
}

MEDIAN_PRIOR = {
    "intro": 100.0,
    "story_ch1": 35.0,
    "story_ch2": 18.0,
    "story_ch3": 8.0,
    "story_ch4": 5.0,
    "story_final": 3.0,
    "platinum": 0.3,
    "completion": 0.4,
    "activity": 8.0,
    "grind": 0.1,
    "side_mission": 2.0,
    "heist": 1.0,
    "money": 0.5,
    "explore": 2.0,
    "misc": 1.0,
}

SHRINK_K = 3.0        # plateau-count units of shrinkage toward the type prior
N_SLOPE_PRIOR = 0.7   # tracked players keep growing, but the launch spike fades
N_SHRINK_K = 2.0
SHOCK_SD = 0.6        # logit-scale refresh shock, common to every trophy
EXTRA_PER_DX = 0.26   # extrapolation uncertainty per unit of log-time horizon
NUNC_PER_DX = 0.21    # uncertainty in the tracked-player forecast, same units
FIT_FLOOR = 0.5       # spread floor when the fit has too few points to judge
S_MIN = 0.45
S_MAX = 2.2


def plateaus(history):
    """Collapse consecutive identical reads into plateaus.

    Exophase only changes on its own refresh, so a history is a step
    function: long runs of identical reads.  Each plateau is anchored at
    its last read: the values are known to hold at least until then, and
    the last read is the freshest observation of the current state.
    """
    out = []
    prev_key = None
    for read in history:
        rates = read.get("rates") or []
        key = (read.get("players_tracked"), tuple(rates))
        try:
            moment = parse_time(read["at"])
        except (KeyError, TypeError, ValueError):
            continue
        if key == prev_key:
            out[-1]["t"] = moment
            continue
        prev_key = key
        out.append({"t": moment, "n": read.get("players_tracked"), "rates": rates})
    return out


def forecast_case(case):
    trophies = case.get("trophies") or []
    if not trophies:
        return {"trophies": []}
    levels = case.get("levels") or [0.05, 0.25, 0.5, 0.75, 0.95]
    zvals = [norm_ppf(level) for level in levels]
    try:
        target = parse_time(case["target"])
    except (KeyError, TypeError, ValueError):
        target = LAUNCH
    x_target = x_of(target)

    cats = [classify(t) for t in trophies]
    hist = plateaus(case.get("history") or [])

    # No usable history at all: fall back to the type priors, wide spread.
    if not hist:
        rows = []
        for trophy, cat in zip(trophies, cats):
            med = MEDIAN_PRIOR.get(cat, 1.0)
            rows.append((trophy.get("id"), med, 1.5))
        return {"trophies": [quantile_row(tid, med, s, zvals) for tid, med, s in rows]}

    x_last = x_of(hist[-1]["t"])
    dx = max(0.0, x_target - x_last)

    # --- Forecast the tracked-player count N(T). ---
    n_pts = [(x_of(p["t"]), math.log(p["n"])) for p in hist
             if isinstance(p["n"], (int, float)) and p["n"] > 0]
    n_last = hist[-1]["n"]
    if not isinstance(n_last, (int, float)) or n_last <= 0:
        for p in reversed(hist):
            if isinstance(p["n"], (int, float)) and p["n"] > 0:
                n_last = p["n"]
                break
    n_target = None
    if isinstance(n_last, (int, float)) and n_last > 0 and len(n_pts) >= 2:
        xs = [p[0] for p in n_pts]
        ys = [p[1] for p in n_pts]
        ws = [1.0] * len(xs)
        slope, _, _, n, _, _ = wls(xs, ys, ws)
        if slope is None:
            b_n = N_SLOPE_PRIOR
        else:
            b_n = (n * slope + N_SHRINK_K * N_SLOPE_PRIOR) / (n + N_SHRINK_K)
        b_n = min(1.8, max(0.0, b_n))
        n_target = n_last * math.exp(b_n * dx)
        n_target = min(n_target, n_last * 50.0)
    elif isinstance(n_last, (int, float)) and n_last > 0:
        n_target = n_last * math.exp(N_SLOPE_PRIOR * dx)

    # --- Per-trophy earner forecasts. ---
    medians = [None] * len(trophies)
    spreads = [None] * len(trophies)
    e_forecast = [None] * len(trophies)
    for idx, trophy in enumerate(trophies):
        cat = cats[idx]
        prior_b = SLOPE_PRIOR.get(cat, 0.45)
        pts = []
        rate_pts = []
        e_last = None
        last_rate = None
        for p in hist:
            rates = p["rates"]
            if idx >= len(rates):
                continue
            rate = rates[idx]
            if rate is None:
                continue
            rate_pts.append((x_of(p["t"]), logit_pct(rate)))
            if not isinstance(p["n"], (int, float)) or p["n"] <= 0:
                continue
            e_last = max(rate * p["n"] / 100.0, 0.5)
            last_rate = rate
            pts.append((x_of(p["t"]), math.log(e_last)))
        if e_last is None:
            # Trophy never observed: prior median, expressed via N(T).
            med = MEDIAN_PRIOR.get(cat, 1.0)
            medians[idx] = med
            spreads[idx] = 1.5
            e_forecast[idx] = (med / 100.0 * n_target) if n_target else None
            continue
        if cat == "intro" and last_rate is not None and last_rate >= 99.5:
            # Essentially every tracked player holds this one, in every read
            # so far: its earners are the tracked players themselves, so no
            # dilution applies to it.  Forecast it at the top, tight.
            medians[idx] = 99.99
            spreads[idx] = 0.25
            e_forecast[idx] = n_target if n_target else e_last
            continue
        if len(pts) >= 2:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ws = [1.0] * len(xs)
            slope, _, resid, n, xbar, sxx = wls(xs, ys, ws)
            if slope is None:
                b = prior_b
                s_fit = FIT_FLOOR
            else:
                b = (n * slope + SHRINK_K * prior_b) / (n + SHRINK_K)
                b = min(2.5, max(-0.5, b))
                rsd = resid if resid is not None else FIT_FLOOR
                rsd = max(rsd, 0.25)
                if sxx and sxx > 1e-12:
                    se = rsd * math.sqrt(1.0 + 1.0 / n + (x_target - xbar) ** 2 / sxx)
                else:
                    se = rsd
                s_fit = se
        else:
            b = prior_b
            s_fit = FIT_FLOOR
        if last_rate == 0:
            s_fit = max(s_fit, 0.9)  # anchor is a rounding floor, not a count
        e_t = e_last * math.exp(b * dx)
        e_forecast[idx] = e_t
        if n_target:
            med = 100.0 * e_t / n_target
        else:
            # No player counts at all: fall back to a logit-rate trend.
            med = rate_trend_median(rate_pts, x_target, prior_b)
        medians[idx] = min(99.99, max(0.01, med))
        spreads[idx] = s_fit

    # --- Structural constraints on the earner counts. ---
    if n_target and all(e is not None for e in e_forecast):
        # Story chapters are sequential: later chapters cannot have more
        # earners than earlier ones.  Enforce a cumulative minimum.
        order = []
        for idx, trophy in enumerate(trophies):
            desc = (trophy.get("description") or "").lower()
            m = re.search(r"complete chapter (\d+)", desc)
            if m:
                order.append((int(m.group(1)), idx))
            elif "complete the final mission" in desc:
                order.append((99, idx))
        order.sort()
        run_min = None
        for _, idx in order:
            if run_min is None:
                run_min = e_forecast[idx]
            else:
                run_min = min(run_min, e_forecast[idx])
                e_forecast[idx] = run_min
        # The platinum needs every other trophy.
        plat = [i for i, t in enumerate(trophies)
                if (t.get("tier") or "").lower() == "platinum"]
        if plat:
            cap = min(e_forecast)
            e_forecast[plat[0]] = min(e_forecast[plat[0]], cap)
        for idx in range(len(trophies)):
            if cats[idx] == "intro":
                continue
            medians[idx] = min(99.99, max(0.01, 100.0 * e_forecast[idx] / n_target))

    # --- Quantiles on the logit scale. ---
    rows = []
    for idx, trophy in enumerate(trophies):
        med = medians[idx]
        s = spreads[idx]
        if s is None:
            s = FIT_FLOOR
        extra = math.sqrt((EXTRA_PER_DX * dx) ** 2 + (NUNC_PER_DX * dx) ** 2
                          + SHOCK_SD ** 2)
        s = math.sqrt(s * s + extra * extra)
        s = min(S_MAX, max(S_MIN, s))
        if cats[idx] == "intro" and med >= 99.5:
            # "Complete Gone Fishing" sits at 100% in every read so far:
            # a tight band just under the top.  On the logit scale any
            # quantile far below 100% pays dearly if the rate stays 100%,
            # so the band stays within a logit or two of the top.
            rows.append((trophy.get("id"),
                         [99.9, 99.95, 99.99, 99.99, 99.99]))
            continue
        rows.append((trophy.get("id"), quantile_row_vals(med, s, zvals)))
    return {"trophies": [{"id": tid, "q": qs} for tid, qs in rows]}


def rate_trend_median(pts, x_target, prior_b):
    """Fallback when no player counts exist: logit-rate trend, shrunk.

    pts are (log-time, logit(rate)) points.  The prior slope is the
    earner-growth prior minus the player-growth prior, since for small
    rates d logit(rate)/dx is about d ln E/dx - d ln N/dx.
    """
    if len(pts) < 2:
        return 1.0
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ws = [1.0] * len(xs)
    slope, intercept, _, n, _, _ = wls(xs, ys, ws)
    if slope is None:
        return 1.0
    b = (n * slope + SHRINK_K * (prior_b - N_SLOPE_PRIOR)) / (n + SHRINK_K)
    last_y = ys[-1]
    x_last = xs[-1]
    return min(99.99, max(0.01,
                           pct_from_logit(last_y + b * (x_target - x_last))))


def quantile_row_vals(median, spread, zvals):
    center = logit_pct(median)
    qs = []
    for z in zvals:
        q = pct_from_logit(center + z * spread)
        q = round(min(99.99, max(0.01, q)), 4)
        qs.append(q)
    # Rounding can break monotonicity at the 4th decimal; repair it.
    for i in range(1, len(qs)):
        if qs[i] < qs[i - 1]:
            qs[i] = qs[i - 1]
    return qs


def quantile_row(tid, median, spread, zvals):
    return {"id": tid, "q": quantile_row_vals(median, spread, zvals)}


def process_case(case):
    try:
        return forecast_case(case)
    except Exception:
        # Never let one bad case kill the run: emit a wide prior forecast.
        rows = []
        for trophy in case.get("trophies") or []:
            cat = classify(trophy)
            med = MEDIAN_PRIOR.get(cat, 1.0)
            rows.append({"id": trophy.get("id"),
                         "q": quantile_row_vals(med, 1.5,
                                                 [norm_ppf(l) for l in
                                                  (case.get("levels") or
                                                   [0.05, 0.25, 0.5, 0.75, 0.95])])})
        return {"trophies": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    for name in names:
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        forecast = process_case(case)
        out_name = name[:-len(".in")] + ".out"
        with open(os.path.join(out_dir, out_name), "w", encoding="utf-8") as fh:
            fh.write(json.dumps(forecast, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
