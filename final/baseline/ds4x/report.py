"""Build a portable Chinese HTML report from verified official four-trial shards."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
from shards import read_task
from run import BASE, REPO, OUT, PILOT, ARMS

DATA=REPO/'data/tau2/domains/banking_knowledge'
HERE=Path(__file__).parent
LABELS="""
1 按个人消费结构选卡
2 环保从业者的信用卡选择
3 环保偏好与订阅权益下的选卡
4 修改邮箱时出现账户归属争议
5 查询余额时核验账户归属
6 为加密货币交易者选卡
7 为中学教师匹配信用卡
8 客户坚持申请不符合条件的优惠
10 比较信用卡推荐奖励
12 客户要求升级到人工支持
14 核实外部渠道发来的银行通知
15 找到用户可以操作的推荐链接
16 核对推荐活动中的交易条件
17-22 核对信用卡返现并提交差额争议
23-25 按用户需要申请信用卡
26-29 核对返现争议及其处理状态
31 核查并提交信用卡交易争议
32-34 处理客户要求转人工的时机
35 对错误逾期记录立即转人工
36-37 识别欺诈交易并处理补卡
38-41 在多笔相似交易中定位争议
43 结清余额并执行信用卡关卡挽留流程
44-49 核对信用卡关卡条件与挽留方案
50-54 审查提额条件与冲突的账户请求
55-57 组合办理账户、转账或支票业务
58-59 联合选择储蓄账户与信用卡
60-61 按约束顺序开立和关闭账户
62 安排四项相互依赖的个人与企业账户操作
63-69 组合选择账户和信用卡以优化储蓄利率
70-71 比较企业账户和相互冲突的促销条件
72-74 核算并申诉 ATM 手续费
75-76 为跨境旅行比较 ATM 费用及账户
77-81 丢失钱包后的多卡安全与补发流程
82-86 核实借记卡交易并分流处理争议
87-89 按拒付代码诊断交易或取款失败
90-92 核查多张卡的 PIN 锁定与风险等级
93-97 核对储蓄利率及账户联动加成
98 在滚动窗口内选择室友开户推荐方案
99-101 在资格、奖励和时间窗口之间选择企业推荐方案
102 核验用户误述的信息与两家企业的推荐资格
"""

def label_map():
    result={}
    for line in LABELS.strip().splitlines():
        span,text=line.split(' ',1); ends=list(map(int,span.split('-')))
        for n in range(ends[0],ends[-1]+1):result[n]=text
    return result

def pack_sim(s):
    r=s['reward_info']; events=[]; hits=set();queries=0; calls={}
    for m in s['messages']:
        role=m['role']
        if role=='tool':
            tool=calls.get(m.get('id'),'归档工具回执'); msg=m.get('content') or ''
            if tool=='KB_search':
                ids=re.findall(r'^   ID: (doc_\S+)\s*$',msg,re.M)
                hits.update(ids);events.append([role,'检索返回',ids])
            else: events.append([role,tool,msg[:700],len(msg)>700])
        else:
            if m.get('content'):events.append([role,'',m['content']])
            for c in m.get('tool_calls') or []:
                calls[c['id']]=c['name'];queries+=c['name']=='KB_search'
                events.append([role,c['name'],c['arguments']])
    return dict(events=events,hits=sorted(hits),queries=queries,basis=r.get('reward_breakdown') or {},score=r['reward'],end=s['termination_reason'],
                checks=[bool(c['action_match']) for c in r.get('action_checks') or []],
                judge=[(v['met'],v['justification']) for v in r.get('nl_assertions') or []])

def main(stage):
    stats=json.loads((OUT/(stage+'-verified.json')).read_bytes())
    labels=label_map(); bank={}
    for f in (DATA/'documents').glob('*.json'):
        d=json.loads(f.read_bytes());bank[d['id']]=d['title']
    archive=next((REPO/'web/leaderboard/public/submissions/gpt-5-2_sierra_2026-02-26/trajectories').glob('*banking_knowledge*'))
    official=json.loads(archive.read_bytes()); authors={}; author_examples={}
    for s in official['simulations']:
        authors.setdefault(s['task_id'],{})[s['trial']]=s['reward_info']['reward']
        if s['trial']==0:
            author_examples[s['task_id']]=pack_sim(s)
    rows=[]
    for tid, arms in stats.items():
        t=json.loads((DATA/'tasks'/(tid+'.json')).read_bytes());ec=t['evaluation_criteria']
        row=dict(id=tid,title=labels[int(tid[5:])],basis=ec['reward_basis'],docs=[[k,bank.get(k,k)] for k in t['required_documents']],actions=ec.get('actions') or [],
                 assertions=ec.get('nl_assertions') or [],scenario=t['user_scenario']['instructions'],notes=t['description'].get('notes'),
                 terminal=[authors[tid][i] for i in range(4)],official_example=author_examples[tid],arms={})
        for arm in ARMS:
            raw,_=read_task(tid,arm)
            raw=raw or {'simulations':[]}
            valid=sorted((s for s in raw['simulations'] if s['termination_reason']!='infrastructure_error' and arms[arm]['rewards'][s['trial']] is not None),key=lambda s:s['trial'])
            assert all(s['reward_info']['reward']==arms[arm]['rewards'][s['trial']] for s in valid)
            common=[i for i in range(4) if all(arms[a]['rewards'][i] is not None for a in ARMS)]
            s=next((s for s in valid if common and s['trial']==common[0]),valid[0] if valid else None)
            recall=[]
            if arm=='bm25':
                required=set(t['required_documents'])
                for episode in valid:
                    calls={call['id'] for message in episode['messages'] for call in (message.get('tool_calls') or []) if call['name']=='KB_search'}
                    retrieved=set()
                    for msg in episode['messages']:
                        if msg['role']=='tool' and msg.get('id') in calls:
                            retrieved.update(re.findall(r'^   ID: (doc_\S+)\s*$',msg.get('content') or '',re.M))
                    group='none' if not calls else 'all' if required<=retrieved else 'partial'
                    recall.append([group,episode['reward_info']['reward']])
            row['arms'][arm]=dict(**arms[arm],trace=pack_sim(s) if s else None,selected_trial=s['trial'] if s else None,recall=recall)
        rows.append(row)
    paired={r['id'] for r in rows if all(None not in a['rewards'] for a in r['arms'].values())}
    author_scores=[]
    submissions=archive.parents[2]
    for prefix, label in [('gpt-5-2_sierra','GPT-5.2 high'),('claude-opus-4-5_sierra','Claude Opus 4.5'),('claude-sonnet-4-5_sierra','Claude Sonnet 4.5'),('gemini-3-flash_sierra','Gemini 3 Flash'),('gemini-3-pro_sierra','Gemini 3 Pro'),('gpt-5-2-none_sierra','GPT-5.2 none')]:
        folder=next(submissions.glob(prefix+'*'))/'trajectories'
        path=next(f for f in folder.glob('*.json') if 'banking' in f.name or 'terminal' in f.name)
        document=official if path==archive else json.loads(path.read_bytes())
        per_task={t:{} for t in paired}
        for sim in document['simulations']:
            if sim['task_id'] in paired:
                trial=sim['trial']; bag=per_task[sim['task_id']]
                assert trial not in bag
                bag[trial]=sim['reward_info']['reward']
        assert all(sorted(v)==[0,1,2,3] for v in per_task.values())
        author_scores.append([label,document['info']['retrieval_config'],sum(sum(v.values()) for v in per_task.values()),sum(all(v.values()) for v in per_task.values())])
    payload=dict(rows=rows,stage=stage,author_scores=author_scores, pilot=PILOT,audit=json.loads((OUT/'offline-checks.json').read_bytes()))
    escaped=json.dumps(payload,ensure_ascii=False,separators=(',',':')).replace('<',r'\u003c')
    out=BASE.parent/('ds4x-pilot.html' if stage=='pilot' else 'ds4x-report.html')
    out.write_text((HERE/'template.html').read_text().replace('/*PAYLOAD*/',escaped))
    print(out,out.stat().st_size)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('pilot','full','snapshot'));main(p.parse_args().stage)
