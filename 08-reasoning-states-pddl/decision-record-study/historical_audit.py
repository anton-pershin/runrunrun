import sys, json, collections, itertools, statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'lib'))
from pddl_env import BlocksWorldEnv
base=ROOT/'results/models/qwen-qwen3.5-9b'
paths=[('root',base/'llm_trajectories_action_catalog.json')]+[(f'batch-{i:02}',base/f'batch-{i:02}'/'llm_trajectories_action_catalog.json') for i in (2,3,4)]
rows=[]; mismatches=[]; verified_events=0
for run,path in paths:
    eps=json.loads(path.read_text())
    assert len(eps)==24
    assert len({(e['task_id'],e['condition']) for e in eps})==24
    for e in eps:
        env=BlocksWorldEnv(ROOT/'pddl/domain.pddl',ROOT/f"pddl/tasks/{e['task_id']}.pddl")
        tag=(run,e['task_id'],e['condition'])
        if env.snapshot()!=e['initial']: mismatches.append([tag,'initial'])
        injected=False; seen_states={tuple(env.snapshot()['state']):0}; failed=set(); consecutive=0; longest=0; repeat_fail=[]; cycles=[]; post=[]; losses=[]; first_goal=0 if env.solved else None; phase_fail=set()
        pregoal_repeat_fail=[]; pregoal_cycles=[]
        for i,t in enumerate(e['trace'],1):
            inject=False
            if e['condition']=='one_transient_failure' and not injected and env.attempts>=2:
                op=env._operators.get(t['action'].strip())
                if op is not None and op.applicable(env.state): inject=True; injected=True
            before_solved=env.solved
            actual=env.step(t['action'],transient_failure=inject)
            verified_events+=1
            if actual!=t: mismatches.append([tag,i,{k:[t.get(k),actual.get(k)] for k in set(t)|set(actual) if t.get(k)!=actual.get(k)}])
            if before_solved: post.append(i)
            if before_solved and not env.solved: losses.append(i)
            if env.solved and first_goal is None: first_goal=i
            if t['status'] not in ('success','transient_failure'):
                key=(tuple(t['state_before']),t['action'],t['status'])
                if key in failed:
                    repeat_fail.append(i)
                    if first_goal is None: pregoal_repeat_fail.append(i)
                failed.add(key)
                consecutive+=1; longest=max(longest,consecutive)
            else: consecutive=0
            if t['world_changed']:
                state=tuple(t['state'])
                if state in seen_states:
                    cycles.append([seen_states[state],i])
                    if first_goal is None: pregoal_cycles.append([seen_states[state],i])
                seen_states[state]=i
        if env.snapshot()!=e['final']: mismatches.append([tag,'final'])
        if env.solved!=e['goal_reached']: mismatches.append([tag,'outcome'])
        if injected!=e['disturbance_injected']: mismatches.append([tag,'injection'])
        rows.append(dict(run=run,task=e['task_id'],condition=e['condition'],solved=env.solved,n=len(e['trace']),error=e['error'],text=e['final_response'],injected=injected,status=collections.Counter(t['status'] for t in e['trace']),cycles=cycles,repeat_fail=repeat_fail,pregoal_cycles=pregoal_cycles,pregoal_repeat_fail=pregoal_repeat_fail,longest_failure_streak=longest,first_goal=first_goal,post_goal=post,goal_losses=losses,trace=e['trace'],final_state=e['final']['state']))

def aggregate(rs):
    statuses=collections.Counter()
    for r in rs: statuses.update(r['status'])
    return dict(episodes=len(rs),successes=sum(r['solved'] for r in rs),ever_goal=sum(r['first_goal'] is not None for r in rs),injected=sum(r['injected'] for r in rs),attempts=sum(r['n'] for r in rs),statuses=dict(statuses),errors=dict(collections.Counter(r['error'] for r in rs if r['error'])),early_unsolved_noerror=sum(not r['solved'] and not r['error'] for r in rs),cycles_events=sum(len(r['cycles']) for r in rs),cycles_episodes=sum(bool(r['cycles']) for r in rs),pregoal_cycles_events=sum(len(r['pregoal_cycles']) for r in rs),pregoal_cycles_episodes=sum(bool(r['pregoal_cycles']) for r in rs),repeated_failure_events=sum(len(r['repeat_fail']) for r in rs),repeated_failure_episodes=sum(bool(r['repeat_fail']) for r in rs),pregoal_repeated_failure_events=sum(len(r['pregoal_repeat_fail']) for r in rs),pregoal_repeated_failure_episodes=sum(bool(r['pregoal_repeat_fail']) for r in rs),post_goal_events=sum(len(r['post_goal']) for r in rs),post_goal_episodes=sum(bool(r['post_goal']) for r in rs),goal_loss_events=sum(len(r['goal_losses']) for r in rs),goal_loss_episodes=sum(bool(r['goal_losses']) for r in rs),max_failure_streak=max(r['longest_failure_streak'] for r in rs))
summary={'verification':dict(episodes=len(rows),events=verified_events,mismatches=mismatches),'all':aggregate(rows),'by_condition':{c:aggregate([r for r in rows if r['condition']==c]) for c in sorted({r['condition'] for r in rows})},'by_run':{run:aggregate([r for r in rows if r['run']==run]) for run,_ in paths},'by_run_condition':{f'{run}/{c}':aggregate([r for r in rows if r['run']==run and r['condition']==c]) for run,_ in paths for c in ('baseline','one_transient_failure')}}
groups=collections.defaultdict(list)
for r in rows: groups[(r['task'],r['condition'])].append(r)
div=[]
for (task,c),rs in sorted(groups.items()):
    full=[tuple((t['action'],t['status']) for t in r['trace']) for r in rs]
    success=[tuple(t['action'] for t in r['trace'] if t['world_changed']) for r in rs]
    pregoal=[tuple(t['action'] for t in r['trace'][:r['first_goal']] if t['world_changed']) for r in rs]
    finals=[tuple(r['final_state']) for r in rs]
    pairs=list(itertools.combinations(rs,2))
    div.append(dict(task=task,condition=c,successes=sum(r['solved'] for r in rs),full_trace_unique=len(set(full)),changed_action_trace_unique=len(set(success)),to_first_goal_changed_trace_unique=len(set(pregoal)),final_states_unique=len(set(finals)),outcome_disagree_pairs=sum(a['solved']!=b['solved'] for a,b in pairs)))
summary['diversity']=div
summary['diversity_aggregate']=dict(groups=len(div),mixed_outcome_groups=sum(0<d['successes']<4 for d in div),all_success_groups=sum(d['successes']==4 for d in div),all_failure_groups=sum(d['successes']==0 for d in div),full_trace_variable_groups=sum(d['full_trace_unique']>1 for d in div),changed_trace_variable_groups=sum(d['changed_action_trace_unique']>1 for d in div),full_trace_distinct_total=sum(d['full_trace_unique'] for d in div),changed_trace_distinct_total=sum(d['changed_action_trace_unique'] for d in div),pair_count=sum(len(list(itertools.combinations(rs,2))) for rs in groups.values()),outcome_disagreement_pairs=sum(d['outcome_disagree_pairs'] for d in div))
out=ROOT/'results/decision-record/historical-audit.json'
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps({'summary':summary,'rows':rows},indent=2))
print(json.dumps(summary,indent=2))
print('SCRATCH_ARTIFACT',out)
print('EPISODES:')
for r in rows:
    print(r['run'],r['task'],r['condition'],'OK' if r['solved'] else 'FAIL','N',r['n'],'first_goal',r['first_goal'],'cycle',r['cycles'],'repeat_failure',r['repeat_fail'],'post',r['post_goal'],'loss',r['goal_losses'],'ERR',r['error'],'TEXT',r['text'][:180].replace('\n',' '))
