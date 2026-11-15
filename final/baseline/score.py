"""Compute pass^k from a complete set of tau2 scored trajectories."""
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def summarize(payload, expected_tasks, trials):
    by_task = defaultdict(dict)
    for sim in payload["simulations"]:
        task, trial = sim["task_id"], sim["trial"]
        if trial in by_task[task]:
            raise ValueError(f"Duplicate {task}, trial {trial}")
        info = sim["reward_info"]
        if info is None or info["reward"] not in (0, 1):
            raise ValueError(f"Missing or nonbinary reward for {task}, trial {trial}")
        by_task[task][trial] = int(info["reward"])
    if set(by_task) != set(expected_tasks) or not by_task:
        raise ValueError("Expected and observed task IDs differ")
    if any(set(values) != set(range(trials)) for values in by_task.values()):
        raise ValueError("Missing or unexpected trial indices")
    counts = [sum(values.values()) for values in by_task.values()]
    scores = {
        f"pass^{k}_pct": round(100 * sum(math.comb(c, k) for c in counts)
                               / (len(counts) * math.comb(trials, k)), 2)
        for k in range(1, trials + 1)
    }
    info = payload["info"]
    return dict(retrieval=info["retrieval_config"], agent=info["agent_info"]["llm"],
                user=info["user_info"]["llm"], seed=info["seed"], tasks=len(counts),
                episodes=len(payload["simulations"]), successful_episodes=sum(counts),
                trials_per_task=trials, successes_per_task_histogram=dict(sorted(Counter(counts).items())),
                termination_reasons=dict(Counter(s["termination_reason"] for s in payload["simulations"])), **scores)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--expected-tasks", required=True, type=int)
    parser.add_argument("--trials", required=True, type=int)
    args = parser.parse_args()
    data = json.loads(args.results.read_bytes())
    ids = {task["id"] for task in data["tasks"]}
    if len(ids) != args.expected_tasks:
        parser.error("unexpected task count in results metadata")
    print(json.dumps(summarize(data, ids, args.trials), indent=2))
