"""Run a Pydantic AI agent against Blocks World and save tool trajectories.

Uses the configured Caila OpenAI-compatible endpoint when present. The agent
sees the full symbolic state and goal, and interacts only through execute_action.
No chain-of-thought is requested or stored; the dataset records tool actions,
outcomes, and world states.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic_ai import Agent, RunContext, UsageLimits
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "lib") not in sys.path:
    sys.path.insert(0, str(ROOT / "lib"))
from pddl_env import BlocksWorldEnv  # noqa: E402

DEFAULT_BASE_URL = "https://caila.io/api/openai/v1"
DEFAULT_MODEL = "gpt-6-luna"
MAX_TOOL_CALLS = 50


@dataclass
class RunDeps:
    env: BlocksWorldEnv
    trace: list[dict[str, Any]]
    condition: str
    inject_after: int
    injected: bool = False


def make_agent() -> Agent[RunDeps, str]:
    api_key = os.environ.get("HERMES_CUSTOM_CAILA_IO_API_KEY")
    if not api_key:
        raise RuntimeError("HERMES_CUSTOM_CAILA_IO_API_KEY is not set")
    model_name = os.environ.get("REASONING_STATES_MODEL", DEFAULT_MODEL)
    base_url = os.environ.get("REASONING_STATES_BASE_URL", DEFAULT_BASE_URL)
    model = OpenAIChatModel(
        model_name,
        provider=OpenAIProvider(base_url=base_url, api_key=api_key),
    )
    agent: Agent[RunDeps, str] = Agent(
        model,
        deps_type=RunDeps,
        output_type=str,
        instructions=(
            "You control a Blocks World through the execute_action tool. "
            "The user message gives the current complete symbolic state and goal. "
            "Call list_available_actions to get the full fixed action vocabulary. "
            "It lists every grounded action, including actions whose preconditions "
            "are currently false; it does not filter infeasible moves. Use the "
            "returned exact action spellings. Choose one action at a time. Use "
            "execute_action, read its result, "
            "then continue until the goal is reached. If an action fails, inspect "
            "the returned state and choose a sensible next action; do not repeat "
            "an unchanged failed action indefinitely. Never claim success unless "
            "the tool result says goal_reached=true. When the goal is reached, "
            "return a brief completion message. Do not explain hidden reasoning."
        ),
    )

    @agent.tool
    def execute_action(ctx: RunContext[RunDeps], action: str) -> dict[str, Any]:
        """Attempt one grounded PDDL action, such as '(pickup b1)'."""
        env = ctx.deps.env
        inject = False
        if ctx.deps.condition == "one_transient_failure" and not ctx.deps.injected:
            if env.attempts >= ctx.deps.inject_after:
                parsed = action.strip()
                op = env._operators.get(parsed)
                if op is not None and op.applicable(env.state):
                    inject = True
                    ctx.deps.injected = True
        transition = env.step(action, transient_failure=inject)
        ctx.deps.trace.append(transition)
        return transition

    @agent.tool
    def list_available_actions(ctx: RunContext[RunDeps]) -> list[str]:
        """List all grounded actions for this task, whether currently feasible or not."""
        return ctx.deps.env.list_available_actions()

    return agent


async def run_episode(
    agent: Agent[RunDeps, str],
    task: dict[str, Any],
    condition: str,
) -> dict[str, Any]:
    domain = ROOT / "pddl" / "domain.pddl"
    problem = ROOT / task["problem_file"]
    env = BlocksWorldEnv(domain, problem)
    initial = env.snapshot()
    trace: list[dict[str, Any]] = []
    deps = RunDeps(
        env=env,
        trace=trace,
        condition=condition,
        inject_after=2,
    )
    prompt = (
        "Solve this task by using execute_action repeatedly.\n"
        f"Current state (true facts): {json.dumps(initial['state'])}\n"
        f"Goal (facts that must become true): {json.dumps(initial['goal'])}\n"
        f"Task id: {task['task_id']}"
    )
    error: str | None = None
    final_text = ""
    try:
        result = await agent.run(
            prompt,
            deps=deps,
            usage_limits=UsageLimits(
                request_limit=MAX_TOOL_CALLS + 2,
                tool_calls_limit=MAX_TOOL_CALLS,
            ),
        )
        final_text = result.output
    except Exception as exc:  # Keep failed episodes as explicit data.
        error = f"{type(exc).__name__}: {exc}"

    return {
        "task_id": task["task_id"],
        "controller": "pydantic_ai_tool_agent",
        "condition": condition,
        "model": os.environ.get("REASONING_STATES_MODEL", DEFAULT_MODEL),
        "base_url": os.environ.get("REASONING_STATES_BASE_URL", DEFAULT_BASE_URL),
        "max_tool_calls": MAX_TOOL_CALLS,
        "disturbance_injected": deps.injected,
        "initial": initial,
        "trace": trace,
        "final": env.snapshot(),
        "goal_reached": env.solved,
        "final_response": final_text,
        "error": error,
    }


async def main_async(args: argparse.Namespace) -> None:
    tasks = json.loads((ROOT / "results" / "tasks.json").read_text())
    if args.limit:
        tasks = tasks[: args.limit]
    agent = make_agent()
    episodes: list[dict[str, Any]] = []
    conditions = ["baseline", "one_transient_failure"] if args.condition == "both" else [args.condition]
    for task in tasks:
        for condition in conditions:
            episode = await run_episode(agent, task, condition)
            episodes.append(episode)
            print(
                f"{task['task_id']} {condition}: calls={len(episode['trace'])} "
                f"goal_reached={episode['goal_reached']} "
                f"injected={episode['disturbance_injected']}"
            )
            if args.fail_fast and episode["error"]:
                raise RuntimeError(episode["error"])
    out = ROOT / "results" / args.output
    out.write_text(json.dumps(episodes, indent=2) + "\n")
    print(f"saved={out} episodes={len(episodes)}")
    print(f"successes={sum(e['goal_reached'] for e in episodes)}/{len(episodes)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="task count; 0 means all")
    parser.add_argument(
        "--condition",
        choices=["both", "baseline", "one_transient_failure"],
        default="both",
    )
    parser.add_argument("--output", default="llm_trajectories.json")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
