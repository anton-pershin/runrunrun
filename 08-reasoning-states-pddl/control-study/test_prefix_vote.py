import asyncio
import unittest
from prefix_vote import run_prefix_vote


class PrefixVoteTests(unittest.TestCase):
    def test_only_first_five_decisions_vote(self):
        calls=[]
        async def propose(**context):
            calls.append(context)
            return ['act','unchecked']
        actions=[]
        async def execute(action):
            actions.append(action)
            return {'status':'success','goal_reached':len(actions)==7}
        r=asyncio.run(run_prefix_vote({},propose,execute,vote_steps=5))
        self.assertEqual(len(calls),17)
        self.assertEqual(actions,['act']*7)
        self.assertEqual(len(r['votes']),5)

    def test_disagreement_does_not_trigger_extra_round(self):
        calls=[]
        async def propose(**context):
            calls.append(1)
            return [['z'],['a'],['b']][len(calls)-1]
        async def execute(action):
            self.assertEqual(action,'z')
            return {'status':'success','goal_reached':True}
        r=asyncio.run(run_prefix_vote({},propose,execute,vote_steps=1))
        self.assertEqual(len(calls),3)
        self.assertEqual(r['votes'][0]['support'],1)

    def test_majority_wins(self):
        plans=iter([['x'],['a'],['a']])
        async def propose(**context): return next(plans)
        async def execute(action):
            self.assertEqual(action,'a')
            return {'status':'success','goal_reached':True}
        r=asyncio.run(run_prefix_vote({},propose,execute,vote_steps=1))
        self.assertTrue(r['goal_reached'])

    def test_failed_attempt_consumes_early_decision(self):
        calls=[]
        async def propose(**context): calls.append(context); return ['a']
        actions=[]
        async def execute(action):
            actions.append(action)
            return {'status':'precondition_failed','goal_reached':False}
        r=asyncio.run(run_prefix_vote({},propose,execute,vote_steps=1,max_actions=2))
        self.assertEqual(len(calls),4)
        self.assertEqual(r['termination'],'action_limit')
        self.assertEqual(len(calls[-1]['failures']),1)

    def test_invalid_samples_are_logged_and_bounded(self):
        calls=[]
        async def propose(**context): calls.append(1); raise ValueError('bad')
        async def execute(action): self.fail('no valid action')
        r=asyncio.run(run_prefix_vote({},propose,execute,vote_steps=5))
        self.assertEqual(len(calls),3)
        self.assertEqual(r['termination'],'error')
        self.assertEqual(len(r['votes'][0]['errors']),3)

    def test_harness_passes_prefix_length(self):
        from experiment import run_policy
        calls=[]
        async def propose(**context): calls.append(1); return ['a']
        async def execute(action): return {'status':'success','goal_reached':True}
        r=asyncio.run(run_policy({},propose,execute,mode='prefix_vote',max_actions=5,vote_steps=0))
        self.assertTrue(r['goal_reached'])
        self.assertEqual(len(calls),1)

    def test_zero_prefix_is_single_sampling(self):
        calls=[]
        async def propose(**context): calls.append(1); return ['a']
        async def execute(action): return {'status':'success','goal_reached':True}
        r=asyncio.run(run_prefix_vote({},propose,execute,vote_steps=0))
        self.assertEqual(len(calls),1)
        self.assertTrue(r['goal_reached'])

if __name__=='__main__': unittest.main()
