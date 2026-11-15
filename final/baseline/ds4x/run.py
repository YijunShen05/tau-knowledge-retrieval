"""Run the official four-trial banking experiment in resumable, per-task shards."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import contextvars
import fcntl
from contextlib import contextmanager
import time
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import threading
import native_deepseek
from tempfile import TemporaryDirectory

BASE = Path(__file__).resolve().parents[1]
REPO = BASE / "tau2-v1.0.0"
OUT = BASE / "runs/ds1017-4x"
PILOT = ["task_" + n for n in ("001", "003", "004", "007", "012", "027", "043", "062", "098", "102")]
ARMS = ("bm25", "golden_retrieval")
MODEL = "openrouter/deepseek/deepseek-v4.1-flash-1017"
PIN = "17e07b1da2bbc0cadfddeea36412686e0604127b"
RNG = random.Random(123)
SEEDS = [RNG.randint(0, 1000000) for _ in range(4)]


NATIVE_OUT = BASE / "runs/ds1017-native-4x"


def completed(path, tid, arm, model):
    if not path.exists():
        return set()
    data = json.loads(path.read_bytes())
    info = data["info"]
    assert (info["retrieval_config"], info["agent_info"]["llm"], info["seed"], info["git_commit"]) == (arm, model, 123, PIN)
    slots = set()
    for sim in data["simulations"]:
        if sim["termination_reason"] == "infrastructure_error":
            continue
        n = sim["trial"]
        assert sim["task_id"] == tid and 0 <= n < 4 and sim["seed"] == SEEDS[n] and n not in slots
        slots.add(n)
    return slots


def pending_native(tid, arm):
    old = completed(OUT / tid / arm / "results.json", tid, arm, MODEL)
    new = completed(NATIVE_OUT / tid / arm / "results.json", tid, arm, native_deepseek.MODEL)
    assert not old.intersection(new)
    return sorted(set(range(4)) - old - new)


@contextmanager
def rollout_slot():
    # Old in-flight shards finish first. Subsequently, one long rollout occupies
    # just one slot while the other slots stay available to any task or arm.
    marker = NATIVE_OUT / "one-active-shard"
    if os.environ.get("DM_DS_ROUTE") != "native" or not marker.exists():
        yield
        return
    for prior in marker.read_text().splitlines():
        path = NATIVE_OUT / prior / "shard.lock"
        if path.parent == Path(os.environ["DM_DS_LEDGER"]).parent:
            continue
        with path.open("a") as previous:
            fcntl.flock(previous, fcntl.LOCK_SH)
    with (NATIVE_OUT / "active-shard.lock").open("a") as drain:
        fcntl.flock(drain, fcntl.LOCK_SH)
        while True:
            limit_file = NATIVE_OUT / "rollout-limit"
            limit = int(limit_file.read_text()) if limit_file.exists() else 4
            if not 1 <= limit <= 776:
                raise ValueError("Invalid rollout concurrency")
            # Extra capacity is used first so older four-slot workers can drain.
            for index in reversed(range(limit)):
                with (NATIVE_OUT / f"rollout-{index}.lock").open("a") as slot:
                    try:
                        fcntl.flock(slot, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        continue
                    yield
                    return
            time.sleep(0.5)


def child(entry=None, current_ledger=None):
    pause = Path(os.environ["DM_DS_BATCH_PAUSE"])
    if pause.exists():
        sys.exit("OpenRouter credit check blocked this batch.")
    from tau2.cli import main
    from tau2.runner import batch
    from tau2.evaluator import evaluator_nl_assertions as nl
    from tau2.utils import llm_utils
    context = contextvars.ContextVar("simulation_id")
    ledger = Path(os.environ["DM_DS_LEDGER"]) if current_ledger is None else None
    lock = threading.Lock()
    run_simulation = batch.run_simulation
    if os.environ.get("DM_DS_ROUTE") == "native":
        native_deepseek.install(llm_utils)
        resume = batch.try_resume
        allowed = set(json.loads(os.environ["DM_DS_TRIALS"]))

        def resume_selected(**kwargs):
            results, done, tasks = resume(**kwargs)
            done.update((i, task.id, seed) for task in tasks for i, seed in enumerate(SEEDS) if i not in allowed)
            return results, done, tasks

        batch.try_resume = resume_selected
    generate = llm_utils.completion
    assert nl.DEFAULT_LLM_NL_ASSERTIONS == "gpt-4.1-2025-04-14"
    nl.DEFAULT_LLM_NL_ASSERTIONS = "openrouter/openai/gpt-4.1"

    def identify(orchestrator, **kwargs):
        token = context.set(orchestrator.simulation_id)
        try:
            with rollout_slot():
                return run_simulation(orchestrator, **kwargs)
        finally:
            context.reset(token)

    def record(*args, **kwargs):
        if pause.exists():
            raise RuntimeError("Provider credit check stopped this batch")
        try:
            response = generate(*args, **kwargs)
        except Exception as error:
            message = str(error).lower()
            if (getattr(error, "status_code", None) in (401, 402, 403)
                    or "requires more credits" in message or "insufficient credits" in message):
                pause.touch()
            raise
        raw = response.to_dict()
        usage = raw.get("usage") or {}
        record = dict(simulation=context.get(), generation=raw["id"], model=raw["model"],
                      usd=usage.get("cost"), input=usage.get("prompt_tokens"), output=usage.get("completion_tokens"))
        if kwargs.get("model") == native_deepseek.MODEL:
            record.update(provider="deepseek", requested_model=native_deepseek.MODEL,
                          cache_hit=usage.get("prompt_cache_hit_tokens"),
                          cache_miss=usage.get("prompt_cache_miss_tokens"), created=raw.get("created"))
        target_ledger = current_ledger.get() if current_ledger is not None else ledger
        with lock, target_ledger.open("a") as file:
            file.write(json.dumps(record) + "\n")
        return response

    batch.run_simulation = identify
    llm_utils.completion = record
    (main if entry is None else entry)()


def launch(pair):
    if Path(os.environ["DM_DS_BATCH_PAUSE"]).exists():
        return
    tid, arm = pair
    native = os.environ.get("DM_DS_ROUTE") == "native"
    folder = (NATIVE_OUT if native else OUT) / tid / arm
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "shard.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        trials = None
        if native:
            trials = pending_native(tid, arm)
            if os.environ.get("DM_DS_ONE") == "1":
                trials = trials[:1]
            if not trials:
                return
        run_shard(tid, arm, folder, trials)
        if native and set(trials).intersection(pending_native(tid, arm)):
            # Transient failures in one shard must leave unrelated live trials running.
            raise RuntimeError(f"{tid}/{arm} has an incomplete native trial; see {folder / 'runner.log'}")


def run_shard(tid, arm, folder, trials=None):
    target = folder / "results.json"
    if trials is None and not target.exists() and tid in PILOT[:5]:
        src = json.loads((BASE / "runs" / ("easy5-ds1017-" + arm) / "results.json").read_bytes())
        assert src["info"]["git_commit"] == PIN and src["info"]["retrieval_config"] == arm
        assert src["info"]["agent_info"]["llm"] == MODEL and src["info"]["seed"] == 123
        src["tasks"] = [t for t in src["tasks"] if t["id"] == tid]
        src["simulations"] = [s for s in src["simulations"] if s["task_id"] == tid]
        [s] = src["simulations"]
        assert s["trial"] == 0 and s["seed"] == SEEDS[0] and s["termination_reason"] == "user_stop"
        src["info"]["num_trials"] = 4
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(src))
        temporary.replace(target)
    if target.exists():
        previous = json.loads(target.read_bytes())["simulations"]
        done = {(s["trial"], s["seed"]) for s in previous if s["termination_reason"] != "infrastructure_error"}
        if done == set(enumerate(SEEDS)):
            return
    env = dict(os.environ, OPENROUTER_API_KEY=(BASE.parent / "key.txt").read_text().strip(),
               TAU2_DATA_DIR=str(REPO / "data"), LITELLM_LOCAL_MODEL_COST_MAP="True",
               DM_DS_LEDGER=str(folder / "receipts.jsonl"), PYTHONUNBUFFERED="1")
    model = MODEL
    model_args = {"extra_body": {"reasoning": {"effort": "high"}}}
    if trials is not None:
        env["DM_DS_TRIALS"] = json.dumps(trials)
        env["DEEPSEEK_API_KEY"] = (BASE.parent / "ds_key.txt").read_text().strip()
        model = native_deepseek.MODEL
        model_args = {"reasoning_effort": "high", "extra_body": {"thinking": {"type": "enabled"}}}
    cmd = [str(REPO / ".venv/bin/python"), str(Path(__file__).resolve()), "run", "--domain", "banking_knowledge",
           "--agent", "llm_agent", "--user", "user_simulator", "--retrieval-config", arm,
           "--agent-llm", model, "--agent-llm-args", json.dumps(model_args),
           "--user-llm","openrouter/openai/gpt-5.2","--user-llm-args",'{"reasoning_effort":"low"}',
           "--seed","123","--num-trials","4","--max-steps","200","--max-errors","10",
           "--max-concurrency","4","--max-retries","0","--auto-resume","--save-to",str(folder),"--task-ids",tid]
    with (folder / "runner.log").open("a") as log:
        subprocess.run(cmd, cwd=REPO, env=env, stdout=log, stderr=log, check=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["run"]:
        child()
    else:
        p = argparse.ArgumentParser()
        p.add_argument("stage", choices=("pilot","remaining","native-one","native-remaining"))
        p.add_argument("--workers",type=int,default=4)
        a=p.parse_args()
        assert subprocess.check_output(["git","-C",str(REPO),"rev-parse","HEAD"],text=True).strip()==PIN
        subprocess.run(["git","-C",str(REPO),"diff","--quiet","HEAD"],check=True)
        catalog=sorted(t["id"] for t in json.loads((REPO/"data/tau2/domains/banking_knowledge/tasks.json").read_bytes()))
        assert len(catalog)==len(set(catalog))==97
        if a.stage == "remaining":
            checked = json.loads((OUT / "pilot-verified.json").read_bytes())
            controls = json.loads((OUT / "offline-checks.json").read_bytes())
            assert set(checked) == set(PILOT) and controls["matched_tasks"] == 97
            assert all(len(checked[t][arm]["rewards"]) == 4 for t in PILOT for arm in ARMS)
        native = a.stage.startswith("native-")
        os.environ["DM_DS_ROUTE"] = "native" if native else "openrouter"
        os.environ["DM_DS_ONE"] = "1" if a.stage == "native-one" else "0"
        tasks=PILOT if a.stage=="pilot" else [t for t in catalog if t not in PILOT]
        pairs = [(tid, arm) for tid in (catalog if native else tasks) for arm in ARMS]
        if native:
            pairs = [pair for pair in pairs if pending_native(*pair)]
            if a.stage == "native-one":
                pairs = pairs[:1]
        with TemporaryDirectory(prefix="tau2-ds-batch-") as batch_dir:
            pause = Path(batch_dir) / "credit-stop"
            os.environ["DM_DS_BATCH_PAUSE"] = str(pause)
            worker_count = a.workers
            limit_path = NATIVE_OUT / "rollout-limit"
            if native and limit_path.exists():
                # A single available trial slot only needs one resident runner.
                worker_count = min(worker_count, max(1, int(limit_path.read_text())))
            with ThreadPoolExecutor(max_workers=worker_count) as ex:
                list(ex.map(launch, pairs))
            if pause.exists():
                sys.exit("Provider error stopped this batch; accepted results are checkpointed.")
