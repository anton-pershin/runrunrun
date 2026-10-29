import asyncio
import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from consensus_controller import run_consensus_controller


def run(*args, **kwargs):
    return asyncio.run(run_consensus_controller(*args, **kwargs))

class ConsensusTests(unittest.TestCase):
    def test_majority_first_operation_is_selected_deterministically(self):
        calls=[]
        plans=[['a','b'],['a','c'],['x','y']]
        async def propose(**kwargs):
            calls.append(kwargs); return plans[len(calls)-1]
        executed=[]
        async def execute(action):
            executed.append(action)
            return {'status':'success','goal_reached':action=='b'}
        result=run({'goal_reached':False},propose,execute,samples=3)
        self.assertEqual(executed,['a','b'])
        self.assertEqual(result['selected_plans'][0]['support'],2)
        self.assertEqual(len(calls),3)

    def test_consensus_plan_is_committed_after_vote(self):
        calls=[]
        async def propose(**kwargs):
            calls.append(kwargs)
            return ['a','b'] if len(calls)%2 else ['a','c']
        async def execute(action):
            return {'status':'success','goal_reached':action=='b'}
        result=run({'goal_reached':False},propose,execute,samples=3)
        self.assertEqual(len(calls),3)
        self.assertEqual(result['executed_actions'],['a','b'])
        self.assertEqual(result['termination'],'goal_reached')

    def test_no_quorum_replans_after_bounded_vote(self):
        calls=[]
        async def propose(**kwargs):
            calls.append(kwargs)
            if len(calls)<=3: return [str(len(calls))]
            return ['z']
        async def execute(action):
            return {'status':'success','goal_reached':action=='z'}
        result=run({'goal_reached':False},propose,execute,samples=3,min_support=2,max_votes=2)
        self.assertEqual(result['revisions'][0]['reason'],'no_quorum')
        self.assertEqual(result['executed_actions'],['z'])

    def test_observed_failure_discards_consensus_suffix_and_revotes(self):
        calls=[]; executed=[]
        async def propose(**kwargs):
            calls.append(kwargs)
            return ['bad','stale'] if len(calls)<=2 else ['repair']
        async def execute(action):
            executed.append(action)
            return {'status':'precondition_failed','goal_reached':False} if action=='bad' else {'status':'success','goal_reached':True}
        result=run({'goal_reached':False},propose,execute,samples=2,min_support=2)
        self.assertEqual(executed,['bad','repair'])
        self.assertEqual(result['revisions'][1]['reason'],'failure')
        self.assertEqual(result['revisions'][1]['failures'][0]['action'],'bad')

    def test_all_invalid_samples_stop_at_request_budget(self):
        calls=[]
        async def propose(**kwargs):
            calls.append(1)
            raise ValueError('unusable output')
        async def execute(action):
            self.fail('must not execute without quorum')
        result=run({},propose,execute,samples=3,max_votes=2)
        self.assertEqual(len(calls),6)
        self.assertEqual(result['termination'],'error')

    def test_repeated_failures_stop_at_action_budget(self):
        async def propose(**kwargs): return ['bad']
        async def execute(action): return {'status':'precondition_failed'}
        result=run({},propose,execute,samples=3,max_actions=2,max_votes=5)
        self.assertEqual(len(result['trace']),2)
        self.assertEqual(result['termination'],'action_limit')

    def test_review_happens_before_any_operation(self):
        from reviewed_controller import run_reviewed_controller
        events=[]
        async def propose(**kwargs):
            events.append(('draft',kwargs['remaining_plan']))
            return ['bad','suffix']
        async def review(**kwargs):
            events.append(('review',kwargs['remaining_plan']))
            return ['good']
        async def execute(action):
            events.append(('execute',action))
            return {'status':'success','goal_reached':True}
        result=asyncio.run(run_reviewed_controller({},propose,review,execute))
        self.assertEqual(events,[('draft',[]),('review',['bad','suffix']),('execute','good')])
        self.assertTrue(result['goal_reached'])

    def test_harness_dispatches_review_mode(self):
        from experiment import run_policy
        async def propose(**kwargs): return ['bad']
        async def review(**kwargs): return ['good']
        async def execute(action):
            self.assertEqual(action,'good')
            return {'status':'success','goal_reached':True}
        result=asyncio.run(run_policy({},propose,execute,mode='review',max_actions=4,review=review))
        self.assertTrue(result['goal_reached'])

    def test_step_consensus_does_not_execute_unchecked_suffix(self):
        calls=[]; executed=[]
        async def propose(**kwargs):
            calls.append(1)
            return ['a','bad_suffix'] if len(calls)<=3 else ['b']
        async def execute(action):
            executed.append(action)
            return {'status':'success','goal_reached':action=='b'}
        result=asyncio.run(run_consensus_controller({},propose,execute,samples=3,
                                                   execution_mode='recede'))
        self.assertEqual(executed,['a','b'])
        self.assertEqual(len(calls),6)

    def test_harness_dispatches_step_vote_mode(self):
        from experiment import run_policy
        async def propose(**kwargs): return ['good','bad']
        async def execute(action): return {'status':'success','goal_reached':True}
        result=asyncio.run(run_policy({},propose,execute,mode='step_vote',max_actions=4))
        self.assertTrue(result['goal_reached'])

    def test_harness_dispatches_consensus_mode(self):
        from experiment import run_policy
        calls=[]
        async def propose(**kwargs):
            calls.append(1)
            return ['done']
        async def execute(action): return {'status':'success','goal_reached':True}
        result=asyncio.run(run_policy({},propose,execute,mode='consensus',max_actions=4))
        self.assertEqual(len(calls),3)
        self.assertTrue(result['goal_reached'])

if __name__=='__main__': unittest.main()
