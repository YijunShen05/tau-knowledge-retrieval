"""Re-score the saved Task 098 trace and isolate its submitted account type offline."""
import argparse
import json
from pathlib import Path
import re

from tau2.data_model.simulation import SimulationRun
from tau2.data_model.tasks import Task
from tau2.domains.banking_knowledge.environment import get_environment
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation


def main(path):
    data = json.loads(path.read_bytes())
    task = Task.model_validate(data["tasks"][0])
    sim = SimulationRun.model_validate(data["simulations"][0])
    assert task.id == sim.task_id == "task_098" and data["info"]["retrieval_config"] == "bm25"
    gold = set(task.required_documents)
    seen = set()
    for message in sim.messages:
        ids = re.findall(r"^   ID: (doc_\S+)\s*$", message.content or "", re.M)
        if ids and message.role == "tool":
            seen.update(ids)
            print(f"retrieved={len(ids)} hit={len(set(ids) & gold)} cumulative={len(seen & gold)}/{len(gold)}")

    alternative = sim.model_copy(deep=True)
    submits = [(i, call) for i, message in enumerate(alternative.messages) for call in (getattr(message, "tool_calls", None) or [])
               if call.name == "submit_referral"]
    [reference] = [action for action in task.evaluation_criteria.actions if action.name == "submit_referral"]
    [(i, submit)] = submits
    print("Submitted:", submit.arguments["account_type"], "Reference:", reference.arguments["account_type"])
    submit.arguments["account_type"] = reference.arguments["account_type"]
    env = get_environment(retrieval_variant="bm25", task=task)
    env.set_state(task.initial_state.initialization_data, task.initial_state.initialization_actions,
                  alternative.messages[:i])
    receipt = alternative.messages[i + 1]
    assert receipt.role == "tool" and receipt.id == submit.id
    receipt.content = env.get_response(submit).content
    for label, trajectory in [("observed", sim), ("account_type_only", alternative)]:
        reward = evaluate_simulation(trajectory, task, EvaluationType.ALL, solo_mode=False,
                                     domain="banking_knowledge", env_kwargs={"retrieval_variant": "bm25", "task": task})
        print(label, "db_match=", reward.db_check.db_match, "reward=", reward.reward,
              {check.action.name: check.action_match for check in reward.action_checks})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    main(parser.parse_args().results)
