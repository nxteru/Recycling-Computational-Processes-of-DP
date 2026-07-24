"""campaign YAML (+ defaults.yaml) -> deduplicated per-stage task lists."""
import hashlib
import itertools
import json
from dataclasses import dataclass
from math import comb
from pathlib import Path

import yaml

from pipeline.constants import EXISTENCE_DEP, STAGE_ORDER
from pipeline.store import REPO_ROOT

DEFAULTS_PATH = REPO_ROOT / "configs" / "defaults.yaml"


def canonical_json(d: dict) -> str:
    return json.dumps(d, sort_keys=True, separators=(",", ":"))


def task_id(params: dict) -> str:
    return hashlib.sha1(canonical_json(params).encode()).hexdigest()[:10]


@dataclass(frozen=True)
class Task:
    stage: str
    params: dict
    task_id: str


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_campaign(path: Path) -> dict:
    defaults = _load_yaml(DEFAULTS_PATH) if DEFAULTS_PATH.exists() else {}
    campaign = _load_yaml(Path(path))

    fixed = {**defaults.get("fixed", {}), **campaign.get("fixed", {})}
    stages = campaign.get("stages", list(STAGE_ORDER))
    unknown = set(stages) - set(STAGE_ORDER)
    if unknown:
        raise ValueError(f"unknown stage(s) in campaign: {unknown}")

    return {
        "name": campaign["name"],
        "description": campaign.get("description", ""),
        "input_method": campaign.get("input_method", defaults.get("input_method", "UNIFORM")),
        "stages": stages,
        "fixed": fixed,
        "grid": campaign.get("grid", {}),
        "overrides": campaign.get("overrides", {}),
        "visualize": campaign.get("visualize", []),
    }


def _apply_overrides(leaf: dict, overrides: dict) -> list[dict]:
    """overrides: {grid_key: {value: {extra_grid_key: [values...]}}}.

    Lets a campaign give one grid dimension (e.g. reservoir_method) its own
    independent sweep of another parameter (e.g. reservoir_size), instead of
    every value in `grid` being cross-producted the same way for every
    method. If a leaf's value for `grid_key` isn't listed in `overrides`,
    the leaf passes through unchanged (falls back to whatever `grid`/`fixed`
    already gave it, e.g. reservoir_size left at its auto-detected width).
    """
    extra_grid: dict = {}
    for key, by_value in overrides.items():
        if key in leaf and leaf[key] in by_value:
            extra_grid.update(by_value[leaf[key]])
    if not extra_grid:
        return [leaf]
    extra_keys = list(extra_grid.keys())
    combos = itertools.product(*[extra_grid[k] for k in extra_keys])
    return [{**leaf, **dict(zip(extra_keys, combo))} for combo in combos]


def _decssp_feature_count(U: int, D: int) -> int:
    """DECSSP's feature count: a constant term plus U prefix-sum cells per
    D position (matches src/20-reservoir/DECSSP/main.py). Computed
    analytically so ELM/NGRC directives can resolve against it without
    DECSSP itself having been generated, or even being in the grid."""
    return 1 + U * D * (D + 1) // 2


def _input_feature_count(D: int) -> int:
    """INPUT's feature count is a pure function of D: a constant term plus
    the D raw input values (src/20-reservoir/INPUT/main.py)."""
    return D + 1


def _ngrc_cumulative_count(D: int, degree: int) -> int:
    """Cumulative NGRC feature count (constant term + degrees 1..degree),
    computed via the stars-and-bars combinations-with-repetition count
    (matches the dfs-based counting in src/20-reservoir/NGRC/main.py,
    which only depends on D and the degree, not on U or the sampled
    values)."""
    return 1 + sum(comb(D + d - 1, d) for d in range(1, degree + 1))


def _ngrc_degree_first_exceeding(D: int, target_count: int) -> int:
    """Smallest NGRC degree whose cumulative feature count is strictly
    greater than target_count."""
    cumulative = 1
    degree = 0
    while cumulative <= target_count:
        degree += 1
        cumulative += comb(D + degree - 1, degree)
    return degree


def _ngrc_degree_nearest(D: int, target_count: int) -> int:
    """NGRC degree whose cumulative feature count is closest to
    target_count (ties broken toward the lower degree)."""
    cumulative = 1
    degree = 0
    best_degree, best_diff = degree, abs(cumulative - target_count)
    while cumulative <= target_count:
        degree += 1
        cumulative += comb(D + degree - 1, degree)
        diff = abs(cumulative - target_count)
        if diff < best_diff:
            best_degree, best_diff = degree, diff
    return best_degree


_RESERVOIR_SIZE_DEFAULT = {
    # Each method's reservoir_size, when a campaign doesn't request one:
    "DECSSP": _decssp_feature_count,  # its own natural width
    "INPUT": lambda U, D: _input_feature_count(D),  # its own natural width
    "ELM": lambda U, D: {"match_reservoir": "DECSSP"},  # same width as DECSSP
    "NGRC": lambda U, D: {"ngrc_degree_exceeding": "DECSSP"},  # lowest degree exceeding DECSSP's width
}


def _apply_reservoir_size_default(leaf: dict) -> dict:
    """Fill in leaf["reservoir_size"] from _RESERVOIR_SIZE_DEFAULT when a
    campaign leaves it unset. A no-op for leaves with no (known)
    reservoir_method, e.g. input-only campaigns."""
    if leaf.get("reservoir_size") is not None:
        return leaf
    default_fn = _RESERVOIR_SIZE_DEFAULT.get(leaf.get("reservoir_method"))
    if default_fn is None:
        return leaf
    return {**leaf, "reservoir_size": default_fn(leaf["U"], leaf["D"])}


def _resolve_static_reservoir_size_directive(leaf: dict) -> dict:
    """Directives whose target width is a pure function of (U, D) / degree
    -- {ngrc_degree_exceeding: "DECSSP"}, {ngrc_degree_nearest: "DECSSP"},
    and {match_reservoir: "DECSSP"[, times: T]} -- are resolved once here
    at grid-expansion time, so no generation needs to have happened yet."""
    rs = leaf.get("reservoir_size")
    if not isinstance(rs, dict):
        return leaf

    if "ngrc_degree_exceeding" in rs:
        if rs["ngrc_degree_exceeding"] != "DECSSP":
            raise ValueError("ngrc_degree_exceeding currently only supports matching DECSSP")
        target_count = _decssp_feature_count(leaf["U"], leaf["D"])
        degree = _ngrc_degree_first_exceeding(leaf["D"], target_count)
        return {**leaf, "reservoir_size": {"ngrc_degree": degree}}

    if "ngrc_degree_nearest" in rs:
        if rs["ngrc_degree_nearest"] != "DECSSP":
            raise ValueError("ngrc_degree_nearest currently only supports matching DECSSP")
        target_count = _decssp_feature_count(leaf["U"], leaf["D"])
        degree = _ngrc_degree_nearest(leaf["D"], target_count)
        return {**leaf, "reservoir_size": {"ngrc_degree": degree}}

    if "match_reservoir" in rs:
        if rs["match_reservoir"] != "DECSSP":
            raise ValueError("match_reservoir currently only supports matching DECSSP")
        times = rs.get("times", 1)
        return {**leaf, "reservoir_size": times * _decssp_feature_count(leaf["U"], leaf["D"])}

    return leaf


def _resolved_reservoir_feature_count(leaf: dict) -> int:
    """reservoir_size as an int, whether it's already one or still NGRC's
    {ngrc_degree: N} form (kept as a degree elsewhere since regression/
    experiment need the degree itself to slice columns, not just its
    width)."""
    rs = leaf.get("reservoir_size")
    if rs is None:
        raise ValueError("train_pool_size: {times_reservoir_size: ...} requires reservoir_method to be set")
    if isinstance(rs, dict):
        return _ngrc_cumulative_count(leaf["D"], rs["ngrc_degree"])
    return rs


def _max_resolved_reservoir_size_by_ud(leaves: list[dict]) -> dict[tuple, int]:
    """The largest feature count any reservoir_method leaf actually
    resolves to, grouped by (U, D), honoring per-campaign reservoir_size
    overrides (not just each method's un-overridden default) -- so a
    shared training pool sized off this is guaranteed wide enough for
    whatever every leaf at that (U, D) actually asks for. `leaves` must
    already have reservoir_size resolved (defaults and overrides
    applied)."""
    max_by_ud: dict[tuple, int] = {}
    for leaf in leaves:
        if leaf.get("reservoir_method") is None:
            continue
        key = (leaf["U"], leaf["D"])
        count = _resolved_reservoir_feature_count(leaf)
        max_by_ud[key] = max(max_by_ud.get(key, 0), count)
    return max_by_ud


def _resolve_train_pool_size_directive(leaf: dict, max_by_ud: dict[tuple, int]) -> dict:
    """train_pool_size may be a plain int, {times_reservoir_size: N}
    (N x the leaf's own reservoir_size), or {times_max_reservoir_size: N}
    (N x the largest feature count any reservoir_method leaf actually
    reaches at this (U, D), independent of the leaf's own
    reservoir_method -- so every method at a given (U, D) can share one
    training pool). `max_by_ud` comes from
    _max_resolved_reservoir_size_by_ud, computed over every leaf with
    reservoir_size already resolved."""
    tp = leaf.get("train_pool_size")
    if not isinstance(tp, dict):
        return leaf
    if "times_reservoir_size" in tp:
        count = _resolved_reservoir_feature_count(leaf)
        mult = tp["times_reservoir_size"]
    elif "times_max_reservoir_size" in tp:
        key = (leaf["U"], leaf["D"])
        if key not in max_by_ud:
            raise ValueError("train_pool_size: {times_max_reservoir_size: ...} requires reservoir_method to be set")
        count = max_by_ud[key]
        mult = tp["times_max_reservoir_size"]
    else:
        raise ValueError(f"unknown train_pool_size directive: {tp}")
    return {**leaf, "train_pool_size": round(mult * count)}


def _resolve_test_size_directive(leaf: dict) -> dict:
    """test_size may be a plain int or {times_train_pool_size: N}, N x the
    leaf's own train_pool_size. Must run after
    _resolve_train_pool_size_directive."""
    ts = leaf.get("test_size")
    if not isinstance(ts, dict):
        return leaf
    if "times_train_pool_size" not in ts:
        raise ValueError(f"unknown test_size directive: {ts}")
    return {**leaf, "test_size": round(ts["times_train_pool_size"] * leaf["train_pool_size"])}


def _resolve_use_size_directive(leaf: dict) -> dict | None:
    """use_size may be a plain int or {times_reservoir_size: N} (N can be
    fractional, e.g. 1/16), N x the leaf's own reservoir_size. Must run
    after _resolve_static_reservoir_size_directive and
    _resolve_train_pool_size_directive.

    Returns None (leaf dropped) when the resolved use_size would exceed
    train_pool_size: a use_size sweep applied uniformly across a
    reservoir_size sweep will produce some combinations that ask for more
    training rows than exist, which aren't valid operating points, not a
    configuration error worth failing the whole run over."""
    us = leaf.get("use_size")
    if us is None:
        return leaf
    if isinstance(us, dict):
        if "times_reservoir_size" not in us:
            raise ValueError(f"unknown use_size directive: {us}")
        resolved = round(us["times_reservoir_size"] * _resolved_reservoir_feature_count(leaf))
    else:
        resolved = us
    if resolved > leaf["train_pool_size"]:
        return None
    return {**leaf, "use_size": resolved}


def expand_leaves(campaign: dict) -> list[dict]:
    grid = campaign["grid"]
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys])) if keys else [()]
    overrides = campaign.get("overrides", {})

    # First pass: apply overrides and resolve each leaf's own reservoir_size
    # (default or override). Needed before train_pool_size can be resolved,
    # since {times_max_reservoir_size: ...} looks across every leaf sharing
    # a (U, D), including any reservoir_size overrides.
    leaves = []
    for combo in combos:
        leaf = dict(campaign["fixed"])
        leaf.update(dict(zip(keys, combo)))
        leaf["input_method"] = campaign["input_method"]
        for expanded_leaf in _apply_overrides(leaf, overrides):
            expanded_leaf = _apply_reservoir_size_default(expanded_leaf)
            expanded_leaf = _resolve_static_reservoir_size_directive(expanded_leaf)
            leaves.append(expanded_leaf)

    max_by_ud = _max_resolved_reservoir_size_by_ud(leaves)

    resolved_leaves = []
    for leaf in leaves:
        leaf = _resolve_train_pool_size_directive(leaf, max_by_ud)
        leaf = _resolve_test_size_directive(leaf)
        leaf = _resolve_use_size_directive(leaf)
        if leaf is not None:
            resolved_leaves.append(leaf)
    return resolved_leaves


def reservoir_extra_params(reservoir_method: str, reservoir_size: int | None, ngrc_max_degree: int | None = None) -> dict:
    """Extra CLI params for reservoir generation.

    ELM generates exactly the leaf's own reservoir_size (num_features =
    reservoir_size - 1, excluding the constant "1" term every method's
    output starts with). Its RNG draws weights row-major, so a wider
    generation's columns aren't a superset of a narrower one's -- leaves
    wanting different widths need separate generations, unlike NGRC below.

    NGRC generates once per (U, D) at the largest degree any leaf there
    asks for (ngrc_max_degree, computed by build_tasks): since each
    degree's monomials are appended after the previous degree's,
    regression/experiment can slice a prefix for smaller degrees.

    DECSSP's width is a pure function of (U, D) and INPUT is a fixed-width
    passthrough -- neither needs an extra parameter.
    """
    if reservoir_method == "ELM":
        assert isinstance(reservoir_size, int)  # build_tasks always resolves ELM's own reservoir_size first
        return {"num_features": reservoir_size - 1}
    if reservoir_method == "NGRC":
        assert ngrc_max_degree is not None  # build_tasks always computes a default
        return {"max_degree": ngrc_max_degree}
    return {}


def _reservoir_size_directive(leaf: dict) -> dict | None:
    """By the time build_tasks runs (after expand_leaves), reservoir_size is
    always either a plain int or NGRC's {ngrc_degree: N} -- every other
    directive (ngrc_degree_exceeding, ngrc_degree_nearest, match_reservoir) and the unset/None
    case are already resolved in expand_leaves."""
    rs = leaf.get("reservoir_size")
    return rs if isinstance(rs, dict) else None


def build_tasks(campaign: dict) -> dict[str, list[Task]]:
    """Expand a campaign into deduplicated task lists, one list per stage.

    Tasks are deduplicated by their canonical parameter dict, so e.g. many
    leaves that share the same (U, D) input only produce one 'input' task.
    """
    leaves = expand_leaves(campaign)
    stages_enabled = set(campaign["stages"])

    tasks: dict[str, dict[str, Task]] = {stage: {} for stage in STAGE_ORDER}

    def add(stage: str, params: dict) -> None:
        tid = task_id(params)
        tasks[stage][tid] = Task(stage, params, tid)

    def add_train_and_test(stage: str, base_params: dict, train_pool: int, test_size: int) -> None:
        for split, size in (("train", train_pool), ("test", test_size)):
            add(stage, {**base_params, "size": size, "split": split})

    # Pre-scan: NGRC's reservoir stage generates once per (U, D) at the
    # largest degree any leaf there requests (see reservoir_extra_params).
    # ELM needs no such pre-scan since it generates each leaf's own width.
    ngrc_max_degree: dict[tuple, int] = {}
    for leaf in leaves:
        directive = _reservoir_size_directive(leaf)
        if not directive:
            continue
        if "ngrc_degree" not in directive:
            raise ValueError(f"unknown reservoir_size directive: {directive}")
        if leaf.get("reservoir_method") != "NGRC":
            raise ValueError("reservoir_size: {ngrc_degree: ...} is only valid when reservoir_method is NGRC")
        key = (leaf["U"], leaf["D"])
        ngrc_max_degree[key] = max(ngrc_max_degree.get(key, 0), directive["ngrc_degree"])

    for leaf in leaves:
        U, D = leaf["U"], leaf["D"]
        input_method = leaf["input_method"]
        train_pool = leaf["train_pool_size"]
        test_size = leaf["test_size"]

        if "input" in stages_enabled:
            add_train_and_test("input", {"input_method": input_method, "U": U, "D": D}, train_pool, test_size)

        target_method = leaf.get("target_method")
        if "target" in stages_enabled and target_method:
            add_train_and_test(
                "target",
                {"input_method": input_method, "U": U, "D": D, "target_method": target_method},
                train_pool,
                test_size,
            )

        reservoir_method = leaf.get("reservoir_method")
        use_size = leaf.get("use_size")
        reservoir_size = leaf.get("reservoir_size")

        if "reservoir" in stages_enabled and reservoir_method:
            reservoir_extra = reservoir_extra_params(reservoir_method, reservoir_size, ngrc_max_degree.get((U, D)))
            add_train_and_test(
                "reservoir",
                {"input_method": input_method, "U": U, "D": D, "reservoir_method": reservoir_method, **reservoir_extra},
                train_pool,
                test_size,
            )

        def regression_params(tm):
            return {
                "input_method": input_method,
                "U": U,
                "D": D,
                "train_pool_size": train_pool,
                "target_method": tm,
                "reservoir_method": reservoir_method,
                "use_size": use_size,
                "reservoir_size": reservoir_size,
            }

        if "regression" in stages_enabled and target_method and reservoir_method and use_size is not None:
            add("regression", regression_params(target_method))
            existence_tm = EXISTENCE_DEP.get(target_method)
            if existence_tm:
                add("regression", regression_params(existence_tm))

        if "experiment" in stages_enabled and target_method and reservoir_method and use_size is not None:
            add(
                "experiment",
                {
                    "input_method": input_method,
                    "U": U,
                    "D": D,
                    "train_pool_size": train_pool,
                    "test_size": test_size,
                    "target_method": target_method,
                    "reservoir_method": reservoir_method,
                    "use_size": use_size,
                    "reservoir_size": reservoir_size,
                    "existence_target_method": EXISTENCE_DEP.get(target_method),
                },
            )

    return {stage: list(d.values()) for stage, d in tasks.items()}
