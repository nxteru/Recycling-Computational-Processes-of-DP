# Recycling-Compuational-Processes-of-DP

Code for the subset-sum problem (SSP) experiments in:

> Sora Todaka, Akihiro Yamamoto, Nozomi Akashi.
> *Recycling computational processes of dynamic programming for
> combinatorial optimization problems: a reservoir computing approach.*

## What it does

The subset sum problem (SSP) asks, for a set of numbers and a target n,
whether some subset sums to n (DECSSP), and if so the max/min number of
elements such a subset can have (MAXSSP/MINSSP). Rather than solving
MAXSSP and MINSSP independently, this code recycles the DP table computed
for the simpler DECSSP subproblem as a reservoir, then trains a linear
readout on it to approximate MAXSSP/MINSSP — both their optimal values and
the actual solutions — at a fraction of the cost of solving them from
scratch. INPUT/ELM/NG-RC reservoirs are included as baselines.

## Running an experiment

```
python -m pipeline.runner run configs/campaigns/<name>.yaml --jobs 4
```

`configs/campaigns/` contains the experiment grids used in the paper;
`pipeline/plot.py` renders the resulting figures.

## Layout

```
src/00-input .. 50-visualize   pipeline stages: input -> target ->
                                reservoir -> regression -> experiment -> visualize
pipeline/                      orchestration (runner, grid, store, aggregate, plot)
configs/campaigns/             experiment definitions
configs/plots/                 figure definitions
```
