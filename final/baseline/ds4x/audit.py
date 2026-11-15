"""Offline controls for the banking baseline. Run with the pinned tau2 interpreter."""
import json
from collections import Counter
from pathlib import Path

from loguru import logger
from tau2.data_model.tasks import Task
from tau2.data_model.simulation import SimulationRun
from tau2.domains.banking_knowledge.environment import get_environment, get_knowledge_base, get_tasks
from tau2.domains.banking_knowledge.retrieval import build_policy, resolve_variant
from tau2.evaluator.evaluator import evaluate_simulation, EvaluationType

from run import REPO, OUT, SEEDS

ARCHIVE = next((REPO / 'web/leaderboard/public/submissions/gpt-5-2_sierra_2026-02-26/trajectories').glob('*banking_knowledge*'))
logger.remove()


def check():
    archive = json.loads(ARCHIVE.read_bytes())
    tasks = {t.id: t for t in get_tasks()}
    ref = {t['id']: Task.model_validate(t) for t in archive['tasks']}
    fields = ('evaluation_criteria','user_scenario','initial_state','required_documents','user_tools')
    assert set(ref) == set(tasks) and len(tasks) == 97
    assert all(getattr(t, k) == getattr(ref[t.id], k) for t in tasks.values() for k in fields)
    bank = get_knowledge_base()
    docs = bank.get_all_documents()
    lookup = {k: d for d in docs for k in (d.id, d.title)}
    gold, bm = map(resolve_variant, ('golden_retrieval', 'bm25'))
    unresolved = []
    for t in tasks.values():
        prompt = build_policy(gold, bank, t)
        for k in t.required_documents:
            if k not in lookup or lookup[k].content not in prompt:
                unresolved.append((t.id,k))
    assert not unresolved, unresolved
    t = tasks['task_004']
    decoy = t.model_copy(deep=True)
    decoy.required_documents = []
    decoy.user_scenario.instructions = 'AUDIT_PRIVATE_CANARY'
    decoy.evaluation_criteria.actions[0].arguments['reason'] = 'AUDIT_PRIVATE_CANARY'
    for variant in (bm, gold):
        assert 'AUDIT_PRIVATE_CANARY' not in build_policy(variant, bank, decoy)
    assert build_policy(bm,bank,decoy) == build_policy(bm,bank,t)
    assert build_policy(gold,bank,decoy) != build_policy(gold,bank,t)
    env_b, env_g = [get_environment(retrieval_variant=a,task=t) for a in ('bm25','golden_retrieval')]
    btools, gtools = [{x.name for x in env.get_tools()} for env in (env_b,env_g)]
    assert btools-gtools=={'KB_search'} and not gtools-btools
    reply = env_b.tools.KB_search('account ownership identity verification')
    import re
    assert len(re.findall(r'^   ID: doc_',reply,re.M))==10
    controls = {}
    for tid, tool in [('task_004','transfer_to_human_agents'),('task_001','apply_for_credit_card'),('task_098','unlock_discoverable_agent_tool')]:
        shard = json.loads((OUT/tid/'bm25/results.json').read_bytes())
        s = SimulationRun.model_validate(next(s for s in shard['simulations'] if (s['trial']==0 if tid=='task_098' else s['reward_info']['reward']==1)))
        edited = s.model_copy(deep=True)
        t = tasks[tid]
        if tid=='task_004':
            for m in edited.messages:
                for c in getattr(m,'tool_calls',[]) or []:
                    if c.name == tool:
                        c.arguments['reason']='other'
        else:
            ids = {c.id for m in edited.messages for c in getattr(m,'tool_calls',[]) or [] if c.name in ({tool, 'call_discoverable_agent_tool'} if tid=='task_098' else {tool})}
            assert ids
            for m in edited.messages:
                if getattr(m,'tool_calls',None):
                    m.tool_calls=[c for c in m.tool_calls if c.id not in ids]
            edited.messages=[m for m in edited.messages if getattr(m,'id',None) not in ids]
        def score(tr):
            return evaluate_simulation(tr,t,EvaluationType.ALL,False,'banking_knowledge',env_kwargs={'retrieval_variant':'bm25','task':t}).reward
        controls[tid] = [score(s),score(edited)]
        assert controls[tid]==([0,1] if tid=='task_098' else [1,0]), controls
    by_task = {}
    for s in archive['simulations']:
        assert s['seed']==SEEDS[s['trial']]
        by_task.setdefault(s['task_id'],[]).append(s['reward_info']['reward'])
    assert len(by_task)==97 and all(len(v)==4 for v in by_task.values())
    return dict(documents=len(docs),matched_tasks=len(tasks),archive_commit=archive['info']['git_commit'],
                terminal_pass1=sum(map(sum,by_task.values()))/388,terminal_pass4=sum(all(v) for v in by_task.values())/97,
                checks=controls,seeds=SEEDS,basis=dict(Counter('+'.join(x.value for x in t.evaluation_criteria.reward_basis) for t in tasks.values())))
if __name__=='__main__':
    report=check()
    print(json.dumps(report,indent=2))
    (OUT/'offline-checks.json').write_text(json.dumps(report,indent=2))
