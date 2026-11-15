"""Read both service routes without rewriting either official checkpoint."""
import json

from run import OUT, NATIVE_OUT, MODEL, PIN
import native_deepseek


def read_task(tid, arm):
    first = None
    accepted = {}
    errors = {}
    routes = {}
    for root in (OUT, NATIVE_OUT):
        path = root / tid / arm / "results.json"
        if not path.exists():
            continue
        data = json.loads(path.read_bytes())
        info = data["info"]
        user = info["user_info"]
        agent = info["agent_info"]
        native = root == NATIVE_OUT
        route = "DeepSeek" if native else "OpenRouter"
        expected_model = native_deepseek.MODEL if native else MODEL
        expected_args = ({"reasoning_effort": "high", "extra_body": {"thinking": {"type": "enabled"}}}
                         if native else {"extra_body": {"reasoning": {"effort": "high"}}})
        assert info["git_commit"] == PIN and info["retrieval_config"] == arm
        assert (agent["implementation"],agent["llm"],agent["llm_args"]) == ("llm_agent",expected_model,expected_args)
        assert (user["implementation"],user["llm"],user["llm_args"]) == ("user_simulator","openrouter/openai/gpt-5.2",{"reasoning_effort":"low"})
        assert [info[k] for k in ("seed","num_trials","max_steps","max_errors")] == [123,4,200,10]
        if first is None:
            first = dict(data, simulations=[])
        else:
            assert first["tasks"] == data["tasks"]
        for sim in data["simulations"]:
            slot = sim["trial"]
            assert sim["task_id"] == tid
            if sim["termination_reason"] == "infrastructure_error":
                errors[slot] = sim
            else:
                assert slot not in accepted, (tid, arm, slot)
                accepted[slot] = sim
                routes[slot] = route
    if first is not None:
        first["simulations"] = list(accepted.values()) + [sim for slot, sim in errors.items() if slot not in accepted]
    return first, routes
