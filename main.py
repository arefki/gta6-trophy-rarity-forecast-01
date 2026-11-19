"""GTA VI trophy-rarity forecast (make.fun gta6-trophy-rarity-forecast-01).

Forecasts, for every PS5 trophy of Grand Theft Auto VI, its earn rate on
Exophase (percent of tracked players) at the case's target instant, as five
quantiles. One run reads every <id>.in in the input directory and writes one
<id>.out per case. Python 3.13, standard library only, no network, no wall
clock, fully deterministic.

Model (per trophy, fitted fresh on the case's own history):
  * Work on the logit scale, logit(p) = ln(p / (1 - p)), p = rate / 100,
    rates clamped to [0.01, 99.99] before the logit.
  * Fit a robust straight line of logit(rate) against x = ln(1 + hours since
    the game's release, 2026-11-19T00:00Z), over the last 24 hours of reads
    (Huber reweighting, so one Exophase refresh step cannot drag the line).
    The log-time axis is the shape of a launch curve: early progress is fast,
    then it flattens as the tracked base grows and the easy trophies saturate.
  * Shrink each trophy's slope lightly toward its tier's pooled slope and
    damp the extrapolation slightly (x0.95), because a straight line in
    log-time can overshoot over a 4.5-day horizon. The median is the damped
    line read at the target, anchored mostly on the fitted line (the last
    observed rate may be a stale held value).
  * Near-certain trophies (last rate >= 90%) are diluted, not extrapolated,
    but ONLY when their fitted trend is flat or falling: new tracked players
    join between the checkpoint and the target, and they have not all earned
    the trophy yet. The players-tracked count is itself forecast on the same
    log-time axis, and the median mixes the old holders with the newcomers
    at a newcomers-earn rate of 0.80-0.90. A near-certain trophy whose rate
    is still climbing steeply is being earned by newcomers almost at once,
    so it keeps its trend forecast instead.
  * A page that has not moved at all since launch is stale, not settled:
    Exophase refreshes in batches, and the next refresh folds in the whole
    launch-night influx, which dilutes every rate. When a trophy's history is
    perfectly flat while the tracked count is also frozen, the median is
    shifted down (0.55 logit) and the spread is widened a lot. This branch
    fires on the practice cases and (by construction) not on a final case
    with two days of real movement.
  * Spreads are honest and wide: random-walk growth with the horizon, the
    fit's residual scatter, an extrapolation penalty that grows with the
    distance past the last read, a staleness term, and a base model term.
    Pinball on the logit scale punishes narrow bands far more than wide ones,
    so the 5%-95% band is built to cover roughly 90% of plausible outcomes,
    especially for rare trophies (platinum / 100%) whose week-one rates are
    dominated by a handful of extreme players.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Damping applied to every fitted slope before extrapolation. Kept near 1:
# the log-time axis already flattens the launch curve, and heavy damping
# biases the median toward persistence (diagnosed on synthetic finals).
SLOPE_DAMP = 0.95
# Weight of a trophy's own slope vs its tier's pooled slope.
OWN_SLOPE_W = 0.85
# Down-shift (logit) for a trophy whose whole history is one frozen page.
STALE_SHIFT = 0.55
# Newcomers' earn rate for a near-certain trophy, as a fraction.
NEWCOMER_RATE_HIGH = 0.90
NEWCOMER_RATE_MID = 0.80


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_release(moment):
    return (moment - RELEASE).total_seconds() / 3600.0


def time_axis(moment):
    """Log-time axis: ln(1 + hours since release). Defined from launch night
    on (reads just before release give a small negative hour count, fine)."""
    return math.log1p(hours_since_release(moment))


def clamp_rate(pct):
    return min(99.99, max(0.01, pct))


def log_odds(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_log_odds(y):
    if y >= 0:
        e = math.exp(-y) if y < 700 else 0.0
        return 100.0 / (1.0 + e)
    e = math.exp(y) if y > -700 else 0.0
    return 100.0 * e / (1.0 + e)


def round4_half_up(x):
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def median(values):
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def ols(points):
    """Ordinary least squares y ~ a + b*x over [(x, y, w)] (w ignored here)."""
    n = len(points)
    sx = sum(p[0] for p in points)
    sy = sum(p[1] for p in points)
    sxx = sum(p[0] * p[0] for p in points)
    sxy = sum(p[0] * p[1] for p in points)
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        return sy / n, 0.0
    slope = (n * sxy - sx * sy) / denom
    return (sy - slope * sx) / n, slope


def huber_line(points):
    """Robust line fit: OLS, then two Huber reweighting passes. Returns
    (intercept, slope, residual_sd). Deterministic."""
    if len(points) < 3:
        return median([p[1] for p in points]), 0.0, 0.0
    intercept, slope = ols(points)
    for _ in range(2):
        resid = [y - (intercept + slope * x) for x, y, _w in points]
        mad = median([abs(r - median(resid)) for r in resid])
        scale = max(1.4826 * mad, 0.05)
        cut = 1.345 * scale
        weighted = []
        for (x, y, _w), r in zip(points, resid):
            w = 1.0 if abs(r) <= cut else cut / abs(r)
            weighted.append((x, y, w))
        n = len(weighted)
        sw = sum(w for _x, _y, w in weighted)
        sx = sum(w * x for x, _y, w in weighted)
        sy = sum(w * y for _x, y, w in weighted)
        sxx = sum(w * x * x for x, _y, w in weighted)
        sxy = sum(w * x * y for x, y, w in weighted)
        denom = sw * sxx - sx * sx
        if abs(denom) < 1e-12 or sw <= 0:
            break
        new_slope = (sw * sxy - sx * sy) / denom
        new_intercept = (sy - new_slope * sx) / sw
        if abs(new_slope - slope) < 1e-12 and abs(new_intercept - intercept) < 1e-12:
            intercept, slope = new_intercept, new_slope
            break
        intercept, slope = new_intercept, new_slope
    resid = [y - (intercept + slope * x) for x, y, _w in points]
    sd = math.sqrt(sum(r * r for r in resid) / len(resid)) if resid else 0.0
    return intercept, slope, sd


def subsample(points, limit):
    """Evenly thin a time-ordered point list to at most `limit` points,
    always keeping the first and the last. Deterministic."""
    if len(points) <= limit:
        return points
    step = (len(points) - 1) / (limit - 1)
    picked = []
    seen = set()
    for k in range(limit):
        idx = min(len(points) - 1, int(round(k * step)))
        if idx not in seen:
            seen.add(idx)
            picked.append(points[idx])
    return picked


def fit_players(history, x_target):
    """Forecast the players-tracked count at the target on the log-time axis
    (same robust line, on ln(players)); never forecasts a decrease."""
    pts = []
    for read in history:
        count = read.get("players_tracked")
        if count and count > 0:
            pts.append((time_axis(parse_time(read["at"])), math.log(count), 1.0))
    if not pts:
        return None
    pts = subsample(pts, 80)
    intercept, slope, _sd = huber_line(pts)
    slope = max(0.0, slope)
    last_x = pts[-1][0]
    last_log = math.log(history[-1]["players_tracked"])
    pred_log = last_log + slope * (x_target - last_x)
    return math.exp(pred_log)


def forecast_case(case):
    trophies = case["trophies"]
    history = case["history"]
    target = parse_time(case["target"])
    horizon_days = case["horizon_minutes"] / 1440.0
    x_target = time_axis(target)

    # Clock hours per read, for the 48h window.
    read_hours = [hours_since_release(parse_time(r["at"])) for r in history]
    read_x = [time_axis(parse_time(r["at"])) for r in history]
    last_hour = read_hours[-1] if read_hours else 0.0

    players_last = history[-1].get("players_tracked") if history else None
    players_frozen = bool(history) and all(
        (r.get("players_tracked") == players_last) for r in history
    )
    players_target = fit_players(history, x_target) if history else None

    # Per-trophy robust fits (first pass: own slope only).
    fits = []
    for ti in range(len(trophies)):
        series = []
        for ri, read in enumerate(history):
            rate = read["rates"][ti] if ti < len(read["rates"]) else None
            series.append((read_x[ri], read_hours[ri], rate))
        fits.append(fit_one(series, last_hour, x_target))

    # Tier-pooled slopes for shrinkage (median of the trophies in the tier).
    pooled = {}
    for tier in ("platinum", "gold", "silver", "bronze"):
        slopes = [f["slope"] for f, t in zip(fits, trophies) if t["tier"] == tier]
        if slopes:
            pooled[tier] = median(slopes)

    rows = []
    for ti, trophy in enumerate(trophies):
        f = fits[ti]
        slope = f["slope"]
        pool = pooled.get(trophy["tier"], slope)
        slope = OWN_SLOPE_W * slope + (1.0 - OWN_SLOPE_W) * pool
        slope *= SLOPE_DAMP

        dx = x_target - f["x_last"]
        y_fit_last = f["intercept"] + f["slope_raw"] * f["x_last"]
        # Anchor mostly on the fitted line: the last observed rate may be a
        # stale held value from before Exophase's latest refresh.
        anchor = 0.75 * y_fit_last + 0.25 * f["y_last"]
        y_med = anchor + slope * dx
        # Never let the median run away from the last observed rate.
        y_med = min(max(y_med, f["y_last"] - 3.0), f["y_last"] + 3.0)

        sigma2 = 0.0
        sigma2 += (0.5 * math.sqrt(horizon_days)) ** 2  # random walk
        sigma2 += min(max(f["resid_sd"], 0.15), 1.2) ** 2  # fit scatter
        sigma2 += (0.10 * abs(slope) * max(dx, 0.0)) ** 2  # extrapolation
        sigma2 += 0.25 ** 2  # model term (flat: the random-walk term already grows with horizon)

        if f["flat"] and players_frozen:
            # One frozen page since launch: the next refresh dilutes rates.
            y_med -= STALE_SHIFT
            sigma2 += 1.0 ** 2
        elif f["flat"]:
            sigma2 += 0.5 ** 2
        elif f["stale_hours"] is not None and f["stale_hours"] >= 2.0:
            sigma2 += 0.3 ** 2

        med_rate = inv_log_odds(y_med)

        # Dilution for near-certain trophies: newcomers have not all earned it.
        # Gate it on the observed trend: dilution only applies when the rate
        # is NOT still rising steeply. A trophy at 99.9% whose rate is still
        # climbing is being earned by newcomers almost at once; diluting it
        # would drag the median far below the truth (one such trophy cost
        # ~0.09 of mean loss in a synthetic final).
        if (
            players_target is not None
            and players_last
            and players_target > players_last * 1.02
            and f["last_rate"] >= 90.0
            and slope <= 0.05
        ):
            newcomer = NEWCOMER_RATE_HIGH if f["last_rate"] >= 99.0 else NEWCOMER_RATE_MID
            diluted = (
                players_last * f["last_rate"]
                + (players_target - players_last) * newcomer * 100.0
            ) / players_target
            med_rate = 0.5 * med_rate + 0.5 * diluted
            sigma2 += 0.4 ** 2

        sigma = min(max(math.sqrt(sigma2), 0.35), 2.5)
        y_med = log_odds(med_rate)
        qs = []
        for z in Z:
            q = inv_log_odds(y_med + z * sigma)
            qs.append(round4_half_up(min(99.99, max(0.01, q))))
        # Enforce non-decreasing quantiles after rounding.
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def fit_one(series, last_hour, x_target):
    """Fit one trophy's logit-rate line. series: [(x, clock_hour, rate|None)].
    Returns a dict of fit facts. No crash on degenerate input."""
    observed = [(x, h, r) for x, h, r in series if r is not None]
    if not observed:
        return {
            "slope": 0.0,
            "slope_raw": 0.0,
            "intercept": log_odds(1.0),
            "resid_sd": 0.3,
            "x_last": time_axis(datetime(2026, 11, 19, tzinfo=timezone.utc)),
            "y_last": log_odds(1.0),
            "last_rate": 1.0,
            "flat": True,
            "stale_hours": None,
        }
    last_x, _lh, last_rate = observed[-1]
    y_last = log_odds(last_rate)
    flat = all(r == observed[0][2] for _x, _h, r in observed)

    # Hours since the last change of this trophy's rate (staleness).
    stale_hours = None
    for k in range(len(observed) - 2, -1, -1):
        if observed[k][2] != last_rate:
            stale_hours = observed[-1][1] - observed[k][1]
            break
    if stale_hours is None:
        stale_hours = observed[-1][1] - observed[0][1]

    # Recent window: reads from the last 24 clock hours (the launch regime
    # changes fast; older reads mix in a different slope); fall back to all.
    recent = [(x, y) for x, h, r in observed for y in [log_odds(r)] if h >= last_hour - 24.0]
    if len({x for x, _y in recent}) < 3:
        recent = [(x, log_odds(r)) for x, _h, r in observed]
    points = subsample([(x, y, 1.0) for x, y in recent], 80)
    if len({x for x, _y, _w in points}) < 2 or flat:
        intercept, slope, resid_sd = y_last, 0.0, 0.30
    else:
        intercept, slope, resid_sd = huber_line(points)
    return {
        "slope": slope,
        "slope_raw": slope,
        "intercept": intercept,
        "resid_sd": resid_sd,
        "x_last": last_x,
        "y_last": y_last,
        "last_rate": last_rate,
        "flat": flat,
        "stale_hours": stale_hours,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out = OUTPUT_DIR / (path.stem + ".out")
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
