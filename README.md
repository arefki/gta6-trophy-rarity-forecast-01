# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/arefki/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/arefki/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts, for every PlayStation 5 trophy of Grand Theft Auto VI, its earn rate on Exophase (percent of tracked players) at 2026-11-26T00:00:00Z — one week after launch — as five quantiles per trophy, from the house's recorded reads of the Exophase trophy page up to the deadline.

The model in `main.py` (Python 3.13, standard library only) fits, per trophy, a robust straight line of the logit earn rate against log-time since release, over the last 24 hours of reads (Huber reweighting, so one Exophase refresh step cannot drag the line), shrinks each slope lightly toward its tier's pooled slope, and reads the damped line at the target. Near-certain trophies are diluted for the newcomers who join between the checkpoint and the target — but only when their trend is flat or falling, never while their rate is still climbing. A page that has not moved at all since launch is treated as stale, not settled: the median shifts down and the spread widens a lot, because the next refresh folds in the whole launch-night influx. Spreads combine random-walk growth with the horizon, fit scatter, extrapolation distance, staleness, and a base model term, so the 5%–95% band is built to cover roughly 90% of plausible outcomes. No network, no wall clock, deterministic: two runs produce byte-identical output.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
