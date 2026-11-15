"""Re-score saved checkpoint trajectories locally, reusing the recorded text verdict."""
import json
from math import prod
from loguru import logger
from tau2.data_model.simulation import SimulationRun
from tau2.data_model.tasks import Task, RewardType
from tau2.evaluator.evaluator import evaluate_simulation, EvaluationType
from run import ARMS, PILOT, OUT

def main():
    logger.remove(); checked=0; verdicts=0
    for tid in PILOT:
        for arm in ARMS:
            j=json.loads((OUT/tid/arm/'results.json').read_bytes())
            task=Task.model_validate(j['tasks'][0])
            uses_nl=RewardType.NL_ASSERTION in task.evaluation_criteria.reward_basis
            if uses_nl:
                task.evaluation_criteria.reward_basis.remove(RewardType.NL_ASSERTION)
            for raw in j['simulations']:
                if raw['termination_reason']=='infrastructure_error':continue
                s=SimulationRun.model_validate(raw)
                r=evaluate_simulation(s,task,EvaluationType.ALL,False,'banking_knowledge',env_kwargs={'task':task,'retrieval_variant':arm})
                score=r.reward
                assert [c.action_match for c in r.action_checks or []]==[c.action_match for c in s.reward_info.action_checks or []]
                assert r.db_check==s.reward_info.db_check
                if uses_nl:
                    verdict=s.reward_info.nl_assertions
                    assert verdict and len(verdict)==len(task.evaluation_criteria.nl_assertions)
                    score*=prod(v.met for v in verdict); verdicts+=1
                assert score==s.reward_info.reward,(tid,arm,s.trial)
                checked+=1
    assert checked==80 and verdicts==8
    (OUT/'pilot-rescored.json').write_text(json.dumps(dict(checked=checked,reused_judgments=verdicts)))
    print(checked,verdicts)
if __name__=='__main__':main()
