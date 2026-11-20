# GTA VI Trophy Rarity Forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ibrahimdiaz0x/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ibrahimdiaz0x/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Entry for [gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01):
a program that forecasts every Grand Theft Auto VI PlayStation 5 trophy's
earn rate on Exophase at 2026-11-26 00:00 UTC, as five quantiles
(5%, 25%, 50%, 75%, 95%) per trophy. Scored by mean pinball loss on the
logit scale; a prize needs a lower loss than the best house entry.

## How it works

An Exophase earn rate is earners / players-tracked, and both move after a
launch: earners only grow as players progress, while the tracked-player
count explodes and dilutes every rate. Modelling the rate directly confuses
the two, so the program (`main.py`, Python 3.13, standard library only,
no network) reconstructs approximate earner counts `E = rate * N / 100`
from the case history and:

1. Fits log E against log-time `x = ln(1 + hours since launch)` per trophy,
   shrinking each trophy's growth slope toward a type prior (story chapters
   grow fast, grinds grow slowly), anchored at the last observed count.
2. Forecasts the tracked-player count N the same way and takes
   `rate(T) = E(T) / N(T)`, so dilution is modelled, not mistaken for
   players abandoning trophies.
3. Enforces structure the data must obey: story chapters are sequential,
   so their forecasted earner counts are non-increasing along the chapter
   order, and the platinum can hold no more earners than any other trophy.
   "Complete Gone Fishing" sits at 100% in every read, so it is forecast
   in a tight band just under the top.
4. Puts quantiles on the logit scale around the median, with a spread
   combining the fit's prediction error, an extrapolation term that grows
   with the horizon, and a refresh-shock term: Exophase refreshes move all
   rates at once, which a per-trophy random walk cannot see.

The program reads every `*.in` case in `$MAKEFUN_INPUT_DIR` (else `/input`)
and writes `/output/<id>.out` for each. It is deterministic: the same case
always produces the same bytes.

## Run it locally

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

where `input/` holds the kit's practice cases (`p00.in`, `p01.in`).

## Local verification

- Practice cases: both run clean in ~0.1 s total, outputs valid (55 rows,
  five ordered quantiles in [0, 100] each, ~2.8 KB), byte-identical across
  runs.
- Synthetic final-case check: a simulated week-one world calibrated to the
  launch-night reads and the live Exophase page ~38 h after launch
  (saturating earner curves; two tracked-player caps; plus a log-linear
  world as a stress case). Mean pinball loss over the three worlds:
  this program 0.12, house trend baseline 0.15, house flat baseline 0.19.
  The simulation is a sanity check built from my own assumptions, not a
  guarantee about the real final.

## License

MIT. See `LICENSE`.
