"""Verify the official results, pair seeds, and account for accepted API generations."""
import argparse
from collections import Counter
from decimal import Decimal
import json
from math import comb

from loguru import logger
from tau2.data_model.simulation import Results
from tau2.metrics.agent_metrics import compute_metrics
from datetime import datetime, timezone
from shards import read_task
from run import NATIVE_OUT, OUT, PILOT, SEEDS, ARMS, MODEL, PIN, REPO

def check_shard(tid, arm, owners, partial=False):
    data, routes = read_task(tid,arm)
    if data is None:
        assert partial, (tid,arm,"missing")
        return dict(rewards=[None]*4, pass_k={},cost={},success_cost=0,missing=0,peak_input={},infra_trials=[],excluded_attempts_with_receipts=0,routes={},estimated_native=0)
    sims=sorted((s for s in data['simulations'] if s['termination_reason']!='infrastructure_error'),key=lambda s:s['trial'])
    expected={(tid,i,seed) for i,seed in enumerate(SEEDS)}
    actual={(s['task_id'],s['trial'],s['seed']) for s in sims}
    assert len(actual)==len(sims) and actual<=expected
    if not partial: assert actual==expected,(tid,arm,'incomplete')
    passed={}
    rewards=[None]*4
    for s in sims: rewards[s['trial']]=s['reward_info']['reward']
    assert all(r in (0,1,None) for r in rewards)
    if len(sims)==4:
        passed=compute_metrics(Results.model_validate(data)).pass_hat_ks
        for k in range(1,5):
            assert abs(passed[k]-comb(int(sum(rewards)),k)/comb(4,k))<1e-9
    ledger={}
    for root in (OUT, NATIVE_OUT):
        path = root/tid/arm/'receipts.jsonl'
        if path.exists():
            for line in path.read_text().splitlines():
                r=json.loads(line,parse_float=Decimal)
                ledger.setdefault(r['simulation'],{})[r['generation']]=r
    cost=Counter()
    peaks=Counter()
    valid_ids={s['id'] for s in sims}
    assert len(valid_ids)==len(sims)
    goodcost=Decimal(0)
    missing=0
    estimated_native=Decimal(0)
    native_tokens=Counter()
    for s in sims:
        calls=dict(ledger.get(s['id'],{}))
        for m in s['messages']:
            raw=m.get('raw_data') or {}
            if raw.get('usage'):
                key=raw['id']
                val=raw['usage'].get('cost')
                if key in calls: assert calls[key]['usd']==(None if val is None else Decimal(str(val)))
                else:
                    calls[key]=dict(model=raw['model'],usd=Decimal(str(val)) if val is not None else None,input=raw['usage'].get('prompt_tokens'))
        assert calls
        assert not owners.intersection(calls), (tid,arm,'duplicate generation ID')
        owners.update(calls)
        for r in calls.values():
            peaks[r['model']]=max(peaks[r['model']], r.get('input') or 0)
            amount=r['usd']
            if r.get('provider')=='deepseek':
                for field in ('cache_hit','cache_miss','output'):
                    if r.get(field) is not None:
                        native_tokens[field]+=r[field]
            if amount is None and r.get('provider')=='deepseek':
                hit,miss,out=r.get('cache_hit'),r.get('cache_miss'),r.get('output')
                if None not in (hit,miss,out,r.get('created')) and hit+miss==r['input']:
                    utc=datetime.fromtimestamp(r['created'],timezone.utc)
                    # Only quote periods whose off-peak status is unambiguous even on holidays.
                    off_peak=utc.weekday()>=5 or utc.hour not in (1,2,3,6,7,8,9)
                    if off_peak:
                        amount=(hit*Decimal('0.003')+miss*Decimal('0.15')+out*Decimal('0.6'))/1000000
                        estimated_native+=amount
            if amount is None:
                missing+=1
            else:
                cost[r['model']]+=amount
                if s['reward_info']['reward']==1: goodcost+=amount
    return dict(routes=routes,native_tokens=dict(native_tokens),estimated_native=float(estimated_native),rewards=rewards,pass_k=passed,infra_trials=sorted({s['trial'] for s in data['simulations'] if s['termination_reason']=='infrastructure_error'}),cost={k:float(v) for k,v in cost.items()},success_cost=float(goodcost),missing=missing,peak_input=dict(peaks),excluded_attempts_with_receipts=len(set(ledger)-valid_ids))

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['pilot','full','snapshot']);args=p.parse_args()
    logger.remove()
    ids=PILOT if args.stage=='pilot' else sorted(t['id'] for t in json.loads((REPO/'data/tau2/domains/banking_knowledge/tasks.json').read_bytes()))
    generations=set()
    out={tid:{arm:check_shard(tid,arm,generations,args.stage=='snapshot') for arm in ARMS} for tid in ids}
    (OUT/(args.stage+'-verified.json')).write_text(json.dumps(out))
    for arm in ARMS:
        rows=[a[arm] for a in out.values()]
        print(arm,sum(v is not None for r in rows for v in r['rewards']),sum(v or 0 for r in rows for v in r['rewards']),sum(all(r['rewards']) for r in rows),round(sum(sum(r['cost'].values()) for r in rows),6),sum(r['missing'] for r in rows))
if __name__=='__main__': main()
