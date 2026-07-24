"""CLI: python -m pipeline.runner run <campaign.yaml> [--jobs N] [--stages a,b,c] [--force]

Executes a campaign's tasks stage by stage (input -> target -> reservoir ->
regression -> experiment) by invoking each stage/method's src/<stage>/<method>/
{gen,main}.py as a subprocess, skipping tasks whose output + meta.json already
match the requested parameters, running independent tasks within a stage
concurrently, and isolating failures so one bad task doesn't stop the rest.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from pathlib import Path

from pipeline import store
from pipeline.constants import STAGE_ORDER
from pipeline.grid import Task, build_tasks, load_campaign

sys.path.insert(0, str(store.REPO_ROOT / "src"))
import _binstore

DEFAULT_JOBS = {"input": 4, "target": 4, "reservoir": 4, "regression": 1, "experiment": 4}


def _default_python() -> str:
    """Prefer the repo's own .venv interpreter over sys.executable, so
    subprocesses still get numpy/torch/etc. even if the pipeline itself
    was invoked with a bare `python3` that never activated .venv -- a
    ModuleNotFoundError buried in one subprocess's log per task is a
    confusing way to discover that."""
    venv_python = store.REPO_ROOT / ".venv" / "bin" / "python3"
    return str(venv_python) if venv_python.exists() else sys.executable


PYTHON = _default_python()


class TaskFailed(Exception):
    pass


def run_cmd(cmd: list, log_path: Path, stdin_path: Path | None = None, stdout_path: Path | None = None) -> None:
    store.ensure_parent(log_path)
    with ExitStack() as stack:
        log_f = stack.enter_context(open(log_path, "wb"))
        stdin_f = stack.enter_context(open(stdin_path, "rb")) if stdin_path else None
        if stdout_path:
            stdout_f = stack.enter_context(open(stdout_path, "wb"))
            stderr_target = log_f
        else:
            stdout_f = log_f
            stderr_target = subprocess.STDOUT
        proc = subprocess.run(
            [str(c) for c in cmd], cwd=store.REPO_ROOT, stdin=stdin_f, stdout=stdout_f, stderr=stderr_target
        )
    if proc.returncode != 0:
        raise TaskFailed(f"exit {proc.returncode}: {' '.join(str(c) for c in cmd)} (see {log_path})")


def first_row_width(path: Path) -> int:
    return _binstore.row_width(path, _binstore.RESERVOIR_DTYPE)


def parse_stage_timing(log_path: Path) -> dict[str, float] | None:
    """Extract the io_sec/compute_sec breakdown a stage script reports on
    its last stderr line (see src/_stage_timing.py's StageTiming.report()),
    routed into the task's log file by run_cmd. Returns None if the script
    didn't report one (e.g. it failed before getting there)."""
    for line in reversed(log_path.read_text(errors="replace").splitlines()):
        if line.startswith("TIMING "):
            fields = dict(kv.split("=", 1) for kv in line[len("TIMING "):].split())
            return {k: float(v) for k, v in fields.items()}
    return None


def reservoir_variant(reservoir_method: str, reservoir_size: int) -> int | None:
    """The store.reservoir_path variant for an already-resolved reservoir_size
    -- ELM's num_features (reservoir_size - 1, excluding the constant term),
    None for every other method (whose reservoir file isn't split by
    requested width; see store.reservoir_path)."""
    return reservoir_size - 1 if reservoir_method == "ELM" else None


def script_path(stage_dir: str, method: str, name: str = "main.py") -> Path:
    return store.REPO_ROOT / "src" / stage_dir / method / name


class Runner:
    def __init__(self, campaign: dict, run_dir: Path, jobs: dict, force: bool):
        self.campaign = campaign
        self.run_dir = run_dir
        self.jobs = jobs
        self.force = force
        self.manifest_path = run_dir / "manifest.jsonl"

    def log_manifest(
        self, stage: str, task_id: str, params: dict, status: str, error: str | None = None, timing: dict | None = None
    ) -> None:
        entry = {"stage": stage, "task_id": task_id, "params": params, "status": status}
        if error:
            entry["error"] = error
        if timing:
            entry.update(timing)
        with open(self.manifest_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def task_log_path(self, stage: str, task_id: str) -> Path:
        return self.run_dir / "logs" / stage / f"{task_id}.log"

    def _run_and_record(
        self,
        stage: str,
        task_id: str,
        out: Path,
        meta_params: dict,
        build_cmd,
        stdin_path: Path | None = None,
        capture_stdout: bool = True,
    ) -> dict | None:
        """Shared skeleton for every _run_*_task below: run build_cmd(tmp)
        writing atomically to `out`, then record the stage script's
        io/compute timing breakdown (see src/_stage_timing.py) in
        out's meta.json. capture_stdout=True means cmd's stdout IS out's
        content (input/target/reservoir); False means cmd itself writes to
        the --out/--result-out path it's given (regression/experiment)."""
        log_path = self.task_log_path(stage, task_id)
        start = time.time()
        with store.atomic_output(out) as tmp:
            run_cmd(build_cmd(tmp), log_path, stdin_path=stdin_path, stdout_path=tmp if capture_stdout else None)
        timing = parse_stage_timing(log_path)
        store.write_meta(out, meta_params, time.time() - start, **(timing or {}))
        return timing

    def run_stage(self, stage: str, tasks: list[Task]) -> None:
        if not tasks:
            return
        handler = getattr(self, f"_run_{stage}_task")
        max_workers = self.jobs.get(stage, 1)
        print(f"[{stage}] {len(tasks)} task(s), jobs={max_workers}")
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(self._guarded, handler, t): t for t in tasks}
            for fut in as_completed(futures):
                t = futures[fut]
                status, error, timing = fut.result()
                self.log_manifest(stage, t.task_id, t.params, status, error, timing)
                timing_suffix = f" (io={timing['io_sec']:.3f}s compute={timing['compute_sec']:.3f}s)" if timing else ""
                print(f"  [{stage}] {t.task_id} {status}" + (f" ({error})" if error else "") + timing_suffix)

    def _guarded(self, handler, task: Task) -> tuple[str, str | None, dict | None]:
        try:
            return handler(task)
        except Exception as e:  # noqa: BLE001 - isolate task failures
            return "failed", str(e), None

    # ---- input ----

    def _run_input_task(self, task: Task) -> tuple[str, str | None, dict | None]:
        p = task.params
        out = store.input_path(p["input_method"], p["U"], p["D"], p["size"], p["split"])
        if not self.force and store.is_cached(out, p):
            return "skipped_cached", None, None
        script = script_path("00-input", p["input_method"], "gen.py")
        seed = 0 if p["split"] == "train" else 1
        timing = self._run_and_record(
            "input", task.task_id, out, p, lambda tmp: [PYTHON, script, p["U"], p["D"], p["size"], seed]
        )
        return "done", None, timing

    # ---- target ----

    def _run_target_task(self, task: Task) -> tuple[str, str | None, dict | None]:
        p = task.params
        out = store.target_path(p["target_method"], p["U"], p["D"], p["size"], p["split"])
        if not self.force and store.is_cached(out, p):
            return "skipped_cached", None, None
        in_path = store.input_path(p["input_method"], p["U"], p["D"], p["size"], p["split"])
        script = script_path("10-target", p["target_method"])
        timing = self._run_and_record("target", task.task_id, out, p, lambda tmp: [PYTHON, script], stdin_path=in_path)
        return "done", None, timing

    # ---- reservoir ----

    def _run_reservoir_task(self, task: Task) -> tuple[str, str | None, dict | None]:
        p = task.params
        variant = p.get("num_features")  # ELM only -- see store.reservoir_path
        out = store.reservoir_path(p["reservoir_method"], p["U"], p["D"], p["size"], p["split"], variant=variant)
        if not self.force and store.is_cached(out, p):
            return "skipped_cached", None, None
        in_path = store.input_path(p["input_method"], p["U"], p["D"], p["size"], p["split"])
        script = script_path("20-reservoir", p["reservoir_method"])

        extra_args = []
        if "num_features" in p:
            extra_args = ["--num-features", p["num_features"]]
        elif "max_degree" in p:
            extra_args = ["--max-degree", p["max_degree"]]

        timing = self._run_and_record(
            "reservoir", task.task_id, out, p, lambda tmp: [PYTHON, script, *extra_args], stdin_path=in_path
        )
        actual_width = first_row_width(out)

        if p["reservoir_method"] == "NGRC":
            # NGRC's width depends on max_degree, which varies per campaign
            # at the same (U, D) -- record it in a degree-keyed sidecar
            # instead of the scalar size.json other methods share, which
            # would otherwise flag a false conflict.
            sizes = self._write_ngrc_degree_sizes(p, script, in_path)
            expected_width = sizes[p["max_degree"]]
            if actual_width != expected_width:
                raise RuntimeError(
                    f"NGRC U={p['U']} D={p['D']} max_degree={p['max_degree']}: generated row width "
                    f"{actual_width} does not match --degree-sizes prediction {expected_width}"
                )
        else:
            store.record_feature_count(p["reservoir_method"], p["U"], p["D"], actual_width, variant=variant)
        return "done", None, timing

    def _write_ngrc_degree_sizes(self, p: dict, script: Path, in_path: Path) -> dict[int, int]:
        sizes_log = self.task_log_path("reservoir", f"ngrc_degree_sizes_{p['U']}_{p['D']}")
        fd, tmp_name = tempfile.mkstemp(suffix=".txt")
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            run_cmd([PYTHON, script, "--degree-sizes", p["max_degree"]], sizes_log, stdin_path=in_path, stdout_path=tmp_path)
            sizes = {}
            for line in tmp_path.read_text().splitlines():
                degree, count = line.split()
                sizes[int(degree)] = int(count)
        finally:
            tmp_path.unlink(missing_ok=True)
        store.write_ngrc_degree_sizes(p["U"], p["D"], sizes)
        return sizes

    # ---- regression ----

    def _available_width(self, reservoir_method: str, U: int, D: int, requested: int) -> int:
        """The width the reservoir stage actually generated, to validate
        `requested` isn't asking for more than what's on disk. ELM and
        NGRC need extra care since neither's width is a pure function of
        (U, D): ELM generates exactly each leaf's own width (an
        exact-match lookup, not a ceiling); NGRC's width depends on
        max_degree, which can differ between campaigns at the same (U, D),
        so the ceiling is the highest degree generated so far."""
        if reservoir_method == "ELM":
            return store.read_feature_count(reservoir_method, U, D, variant=requested - 1)
        if reservoir_method == "NGRC":
            return max(store.read_ngrc_degree_sizes(U, D).values())
        return store.read_feature_count(reservoir_method, U, D)

    def _resolve_reservoir_size(self, p: dict) -> int:
        """reservoir_size is a plain int for every method except NGRC,
        which keeps its {ngrc_degree: N} form (resolved here via the
        degree_sizes sidecar). Validates the resolved width actually fits
        what the reservoir stage generated -- a misconfigured campaign
        should fail loudly, not silently train on a truncated slice (numpy
        slicing past an array's end doesn't raise)."""
        requested = p["reservoir_size"]
        if isinstance(requested, dict):
            requested = store.read_ngrc_degree_size(p["U"], p["D"], requested["ngrc_degree"])

        available = self._available_width(p["reservoir_method"], p["U"], p["D"], requested)
        if requested > available:
            raise ValueError(
                f"requested reservoir_size={requested} exceeds the {available} features "
                f"generated for {p['reservoir_method']} U={p['U']} D={p['D']}"
            )
        return requested

    def _resolve_use_size(self, p: dict, reservoir_size: int) -> int:
        """use_size may be a plain int or {times_reservoir_size: N} (N x
        reservoir_size, rounded). pipeline/grid.py's
        _resolve_use_size_directive already resolves this to a plain int
        at campaign-expansion time (dropping any leaf that would exceed
        train_pool_size) -- this is just a safety net for that invariant."""
        requested = p["use_size"]
        if isinstance(requested, dict):
            if "times_reservoir_size" in requested:
                requested = round(requested["times_reservoir_size"] * reservoir_size)
            else:
                raise ValueError(f"unknown use_size directive: {requested}")
        if requested > p["train_pool_size"]:
            raise ValueError(f"use_size={requested} exceeds train_pool_size={p['train_pool_size']}")
        return requested

    def _run_regression_task(self, task: Task) -> tuple[str, str | None, dict | None]:
        p = task.params
        reservoir_size = self._resolve_reservoir_size(p)
        use_size = self._resolve_use_size(p, reservoir_size)
        resolved_params = {**p, "reservoir_size": reservoir_size, "use_size": use_size}
        out = store.regression_path(p["target_method"], p["reservoir_method"], p["U"], p["D"], reservoir_size, use_size)
        if not self.force and store.is_cached(out, resolved_params):
            return "skipped_cached", None, None

        variant = reservoir_variant(p["reservoir_method"], reservoir_size)
        target_file = store.target_path(p["target_method"], p["U"], p["D"], p["train_pool_size"], "train")
        reservoir_file = store.reservoir_path(
            p["reservoir_method"], p["U"], p["D"], p["train_pool_size"], "train", variant=variant
        )
        script = store.REPO_ROOT / "src" / "30-regression" / p["target_method"] / "LR.py"

        timing = self._run_and_record(
            "regression",
            task.task_id,
            out,
            resolved_params,
            lambda tmp: [
                PYTHON, script,
                "--target-file", target_file,
                "--reservoir-file", reservoir_file,
                "--use-size", use_size,
                "--reservoir-size", reservoir_size,
                "--out", tmp,
            ],
            capture_stdout=False,
        )
        return "done", None, timing

    # ---- experiment ----

    def _run_experiment_task(self, task: Task) -> tuple[str, str | None, dict | None]:
        p = task.params
        reservoir_size = self._resolve_reservoir_size(p)
        use_size = self._resolve_use_size(p, reservoir_size)
        resolved_params = {**p, "reservoir_size": reservoir_size, "use_size": use_size}
        out = store.experiment_path(p["target_method"], p["reservoir_method"], p["U"], p["D"], reservoir_size, use_size)
        if not self.force and store.is_cached(out, resolved_params):
            return "skipped_cached", None, None

        variant = reservoir_variant(p["reservoir_method"], reservoir_size)
        target_test = store.target_path(p["target_method"], p["U"], p["D"], p["test_size"], "test")
        reservoir_test = store.reservoir_path(
            p["reservoir_method"], p["U"], p["D"], p["test_size"], "test", variant=variant
        )
        wout = store.regression_path(p["target_method"], p["reservoir_method"], p["U"], p["D"], reservoir_size, use_size)
        script = store.REPO_ROOT / "src" / "40-experiment" / p["target_method"] / "main.py"

        cmd = [
            PYTHON, script,
            "--U", p["U"],
            "--D", p["D"],
            "--target-test-file", target_test,
            "--reservoir-test-file", reservoir_test,
            "--wout-file", wout,
            "--reservoir-size", reservoir_size,
            "--use-size", use_size,
            "--train-pool-size", p["train_pool_size"],
            "--test-size", p["test_size"],
            "--input-method", p["input_method"],
            "--target-method", p["target_method"],
            "--reservoir-method", p["reservoir_method"],
        ]
        existence_tm = p.get("existence_target_method")
        if existence_tm:
            existence_wout = store.regression_path(existence_tm, p["reservoir_method"], p["U"], p["D"], reservoir_size, use_size)
            cmd += ["--existence-wout-file", existence_wout]
        if p["target_method"] in ("MAXSSP", "MINSSP"):
            input_test = store.input_path(p["input_method"], p["U"], p["D"], p["test_size"], "test")
            cmd += ["--input-test-file", input_test]

        timing = self._run_and_record(
            "experiment", task.task_id, out, resolved_params, lambda tmp: cmd + ["--result-out", tmp], capture_stdout=False
        )
        return "done", None, timing


def run_campaign(campaign_path: Path, jobs_override: int | None, stages_filter: list[str] | None, force: bool) -> Path:
    campaign = load_campaign(campaign_path)
    if stages_filter:
        campaign["stages"] = [s for s in campaign["stages"] if s in stages_filter]

    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = store.RUNS_ROOT / f"{run_id}_{campaign['name']}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.snapshot.yaml").write_text(Path(campaign_path).read_text())

    jobs = dict(DEFAULT_JOBS)
    if jobs_override:
        jobs = {k: jobs_override if k != "regression" else jobs.get("regression", 1) for k in jobs}

    runner = Runner(campaign, run_dir, jobs, force)
    tasks = build_tasks(campaign)
    for stage in STAGE_ORDER:
        if stage in campaign["stages"]:
            runner.run_stage(stage, tasks[stage])

    if campaign.get("visualize"):
        from pipeline import aggregate, plot

        aggregate.rebuild_index()
        for spec_name in campaign["visualize"]:
            plot.render(store.REPO_ROOT / "configs" / "plots" / f"{spec_name}.yaml")

    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run")
    run_p.add_argument("campaign", type=Path)
    run_p.add_argument("--jobs", type=int, default=None)
    run_p.add_argument("--stages", type=str, default=None, help="comma-separated subset of input,target,reservoir,regression,experiment")
    run_p.add_argument("--force", action="store_true")
    args = parser.parse_args()

    stages_filter = args.stages.split(",") if args.stages else None
    run_dir = run_campaign(args.campaign, args.jobs, stages_filter, args.force)
    print(f"run directory: {run_dir}")


if __name__ == "__main__":
    main()
