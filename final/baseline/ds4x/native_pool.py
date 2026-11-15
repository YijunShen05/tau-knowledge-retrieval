"""Resume official per-trial runs using shared imports and separate shard checkpoints."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextvars import ContextVar
import argparse
import fcntl
import json
import os
import tempfile
import threading
import time

import run

ledger = ContextVar("receipt_file")


def execute(workers):
    from loguru import logger
    from tau2.runner import batch
    from tau2.runner.checkpoint import create_checkpoint_saver
    from tau2.data_model.simulation import TextRunConfig
    from tau2.evaluator.evaluator import EvaluationType
    from tau2.domains.banking_knowledge.environment import get_tasks

    logger.remove()
    tasks = {t.id:t for t in get_tasks()}
    configs = {arm:TextRunConfig(domain="banking_knowledge", retrieval_config=arm,
                  agent="llm_agent", user="user_simulator", llm_agent="deepseek-flash",
                  llm_args_agent={"reasoning_effort":"high","extra_body":{"thinking":{"type":"enabled"}}},
                  llm_user="openrouter/openai/gpt-5.2",llm_args_user={"reasoning_effort":"low"},
                  max_steps=200,max_errors=10,num_trials=4,seed=123,max_retries=0) for arm in run.ARMS}
    pending={(tid,a) for tid in tasks for a in run.ARMS if run.pending_native(tid,a)}
    jobs={};held={}; errors=[]; retries={}
    def one(tid,arm,n,folder,save):
        token=ledger.set(folder/"receipts.jsonl")
        try:
            s=batch.run_single_task(configs[arm].model_copy(deep=True),tasks[tid].model_copy(deep=True),
                   seed=run.SEEDS[n],evaluation_type=EvaluationType.ALL_WITH_NL_ASSERTIONS)
            s.trial=n
            assert s.seed==run.SEEDS[n] and s.task_id==tid and s.reward_info is not None
            save(s)
        finally:
            ledger.reset(token)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        while pending or jobs:
            for pair in list(pending):
                tid,arm=pair; folder=run.NATIVE_OUT/tid/arm
                folder.mkdir(parents=True,exist_ok=True)
                lock=(folder/"shard.lock").open("a")
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    lock.close();continue
                pending.remove(pair)
                slots=run.pending_native(tid,arm)
                if not slots:
                    lock.close();continue
                path=folder/"results.json"
                assert path.exists(), f"Expected official shard metadata: {path}"
                data=json.loads(path.read_bytes())
                keep=[s for s in data["simulations"] if s["termination_reason"]!="infrastructure_error"]
                if len(keep)!=len(data["simulations"]):
                    data["simulations"]=keep
                    tmp=path.with_suffix(".recovering");tmp.write_text(json.dumps(data));tmp.replace(path)
                saved=create_checkpoint_saver(path,threading.Lock());held[pair]=[lock,len(slots)]
                for n in slots:
                    args=(tid,arm,n,folder,saved)
                    jobs[pool.submit(one,*args)]=args
            if not jobs:
                if pending:time.sleep(1)
                continue
            finished,_=wait(jobs,timeout=1,return_when=FIRST_COMPLETED)
            for f in finished:
                args=jobs.pop(f);tid,arm,n=args[:3];pair=(tid,arm)
                try:f.result()
                except Exception as e:
                    kind=type(e).__name__
                    print("trial error",tid,arm,n,kind,str(e).split("https://")[0][:120],flush=True)
                    if not os.path.exists(os.environ["DM_DS_BATCH_PAUSE"]) and retries.get(args[:3],0)<2:
                        retries[args[:3]]=retries.get(args[:3],0)+1
                        jobs[pool.submit(one,*args)]=args;continue
                    errors.append((tid,arm,n,kind))
                held[pair][1]-=1
                if not held[pair][1]:
                    held.pop(pair)[0].close()
    if errors:raise RuntimeError(f"Unfinished trials: {errors}")


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--workers",type=int,default=1);args=p.parse_args()
    with tempfile.TemporaryDirectory(prefix="ds-pool-") as d:
        os.environ.update(DM_DS_BATCH_PAUSE=d+"/credit-stop",DM_DS_ROUTE="native",DM_DS_TRIALS="[]",
            DM_DS_LEDGER=d+"/unused", DEEPSEEK_API_KEY=(run.BASE.parent/"ds_key.txt").read_text().strip(),
            OPENROUTER_API_KEY=(run.BASE.parent/"key.txt").read_text().strip(),
            TAU2_DATA_DIR=str(run.REPO/"data"), LITELLM_LOCAL_MODEL_COST_MAP="True")
        run.child(lambda:execute(args.workers),ledger)
