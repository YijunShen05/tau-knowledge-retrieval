"""Summarize accepted rollouts, provider usage, and assistant tool calls offline."""
from collections import Counter, defaultdict
from decimal import Decimal
import json
from pathlib import Path

from shards import read_task
from run import ARMS, OUT, NATIVE_OUT, SEEDS


def main():
    verified = json.loads((OUT / "full-verified.json").read_bytes())
    seen = set()
    matched_calls = 0
    totals = defaultdict(Counter)
    tools = defaultdict(Counter)
    trial_zero = Counter()
    for task, groups in verified.items():
        for arm in ARMS:
            data, _ = read_task(task, arm)
            receipts = {}
            for root in (OUT, NATIVE_OUT):
                path = root / task / arm / "receipts.jsonl"
                if path.exists():
                    for line in path.read_text().splitlines():
                        r = json.loads(line, parse_float=Decimal)
                        receipts.setdefault(r["simulation"], {})[r["generation"]] = r
            for s in data["simulations"]:
                if s["termination_reason"] == "infrastructure_error":
                    continue
                n = s["trial"]
                result = s["reward_info"]["reward"]
                assert s["seed"] == SEEDS[n] and groups[arm]["rewards"][n] == result
                if n == 0:
                    trial_zero[arm] += int(result)
                call_ids = Counter(call["id"] for message in s["messages"]
                                   for call in message.get("tool_calls") or [])
                response_ids = Counter(message["id"] for message in s["messages"]
                                       if message["role"] == "tool")
                assert call_ids == response_ids, (task, arm, n)
                matched_calls += sum(response_ids.values())
                calls = dict(receipts.get(s["id"], {}))
                for m in s["messages"]:
                    if m["role"] == "assistant":
                        for c in m.get("tool_calls") or []:
                            name = c["name"]
                            tools[arm][name] += 1
                            if name == "call_discoverable_agent_tool":
                                tools["discovered"][str(c["arguments"].get("agent_tool_name"))] += 1
                    raw = m.get("raw_data") or {}
                    usage = raw.get("usage")
                    if not usage:
                        continue
                    g = raw["id"]
                    r = calls.setdefault(g, dict(model=raw["model"], input=usage["prompt_tokens"],
                                                 output=usage["completion_tokens"], usd=usage.get("cost")))
                    assert (r["input"], r["output"]) == (usage["prompt_tokens"], usage["completion_tokens"])
                    if "deepseek" in r["model"]:
                        details = usage.get("prompt_tokens_details") or {}
                        r["cache_hit"] = usage.get("prompt_cache_hit_tokens", details.get("cached_tokens"))
                        r["cache_miss"] = usage.get("prompt_cache_miss_tokens", r["input"] - r["cache_hit"])
                        assert r["cache_hit"] + r["cache_miss"] == r["input"]
                        for choice in raw["choices"]:
                            assert not (choice.get("message") or {}).get("annotations")
                        for key, count in (usage.get("server_tool_use") or {}).items():
                            if "search" in key or "fetch" in key:
                                assert count == 0, (s["id"], key)
                for g, r in calls.items():
                    assert g not in seen
                    seen.add(g)
                    model = r["model"]
                    role = "agent" if "deepseek" in model else {"openai/gpt-5.2":"user", "openai/gpt-4.1":"judge"}.get(model)
                    assert role, model
                    counts = totals[(arm, role)]
                    counts["calls"] += 1
                    for key in ("input", "output"):
                        counts[key] += r[key]
                    if role == "agent":
                        for key in ("cache_hit", "cache_miss"):
                            counts[key] += r[key]
                    if r.get("usd") is not None:
                        counts["usd"] += Decimal(str(r["usd"]))
    for (arm, role), counts in sorted(totals.items()):
        print(arm, role, dict(counts))
    print("matched tool calls and responses", matched_calls)
    print("trial 0", dict(trial_zero))
    for arm, counts in tools.items():
        print("tools", arm, dict(sorted(counts.items())))


if __name__ == "__main__":
    main()
