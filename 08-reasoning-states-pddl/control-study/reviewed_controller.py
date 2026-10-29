"""Draft, review against observations/contracts, then commit. No world model."""
from controller import run_controller


async def run_reviewed_controller(initial, propose, review, execute, *, max_actions=50,
                                  max_replans=25):
    drafts=[]
    async def reviewed(**context):
        draft = await propose(**context)
        drafts.append(list(draft))
        return await review(observation=context['observation'],failures=context['failures'],
                            remaining_plan=draft)
    result=await run_controller(initial,reviewed,execute,mode='commit',
                                max_actions=max_actions,max_replans=max_replans,
                                transient_retries=1)
    result['drafts']=drafts
    return result
