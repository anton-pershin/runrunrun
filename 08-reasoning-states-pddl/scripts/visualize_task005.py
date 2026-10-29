"""Render proxy-state trajectories as aligned, time-indexed SVG panels.

Y is the number of goal facts true after each tool call. X is tool-call time.
Action labels are vertical. Each trajectory uses its own aligned panel to avoid
overlap while retaining common time and goal-count scales.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS = [
    ROOT / "results/models/qwen-qwen3.5-9b/llm_trajectories_action_catalog.json"
]
DEFAULT_OUT = ROOT / "results/models/qwen-qwen3.5-9b/task-005-baseline-proxy-states.svg"
COLORS = {
    "gpt-6-luna": "#2463a6",
    "qwen-qwen3.5-9b": "#c05a24",
    "other": "#27836f",
}
STATUS_STYLE = {
    "success": ("#2e8b57", "#2e8b57", "circle"),
    "precondition_failed": ("#c0392b", "#c0392b", "x"),
    "unknown_action": ("#8e44ad", "#8e44ad", "diamond"),
    "malformed_action": ("#7f8c8d", "#7f8c8d", "square"),
    "transient_failure": ("#e67e22", "#e67e22", "triangle"),
}
ACTION_STYLE = {
    "stack": "#2463a6",
    "unstack": "#8e44ad",
    "pickup": "#148f77",
    "putdown": "#ca6f1e",
    "other": "#34495e",
}


def action_family(action: str) -> str:
    name = action.strip().lstrip("(").split()[0]
    return name if name in ACTION_STYLE else "other"


def model_key(path: Path) -> str:
    for parent in path.parents:
        if parent.name in COLORS:
            return parent.name
    return "other"


def load_episode(path: Path, task_id: str, condition: str) -> tuple[dict, list[dict]]:
    data = json.loads(path.read_text())
    episode = next(
        e for e in data if e["task_id"] == task_id and e["condition"] == condition
    )
    goals = set(episode["initial"]["goal"])
    initial_n = len(goals & set(episode["initial"]["state"]))
    points = [{"t": 0, "action": "START", "status": "start", "count": initial_n}]
    for event in episode["trace"]:
        count = len(goals & set(event["state"]))
        points.append(
            {
                "t": event["attempt"],
                "action": event["action"],
                "family": action_family(event["action"]),
                "status": event["status"],
                "count": count,
            }
        )
    return episode, points


def esc(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render(task_id: str, condition: str, paths: list[Path], output: Path) -> None:
    loaded = [(p, *load_episode(p, task_id, condition)) for p in paths]
    if not loaded:
        raise ValueError("provide at least one trajectory dataset")
    max_t = max(len(points) - 1 for _, _, points in loaded)
    max_goal = max(len(ep["initial"]["goal"]) for _, ep, _ in loaded)
    width = max(1600, 160 + max_t * 44)
    left, right, top = 95, 45, 125
    tick_h, panel_gap = 35, 35
    panel_plot_h = 250
    panel_h = panel_plot_h + tick_h
    height = top + len(loaded) * panel_h + max(0, len(loaded)-1) * panel_gap + 135
    plot_w = width - left - right
    x = lambda t: left + plot_w * t / max(1, max_t)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<style>text{font-family:DejaVu Sans,Arial,sans-serif;fill:#202a35}.title{font-size:25px;font-weight:bold}.sub{font-size:14px}.axis{font-size:13px}.tick{font-size:11px}.action{font-size:9px}.legend{font-size:12px}.panel{font-size:16px;font-weight:bold}</style>',
        f'<text x="40" y="38" class="title">Blocks World · {esc(task_id)} · {esc(condition)} · trajectory comparison</text>',
        '<text x="40" y="64" class="sub">Each row is one rollout. Shared scales: X = tool-call time; Y = number of goal facts true after each call. START is t=0.</text>',
        '<text x="40" y="86" class="sub">Markers show outcomes; vertical action labels use action-family colors. Panels avoid overlap without changing data coordinates.</text>',
    ]

    # Fixed action-family legend.
    svg.append(f'<text x="{left}" y="108" class="legend" font-weight="bold">Action labels:</text>')
    lx = left + 100
    for family, label in [("stack", "stack"), ("unstack", "unstack"), ("pickup", "pickup"), ("putdown", "putdown")]:
        svg.append(f'<text x="{lx}" y="108" class="legend" fill="{ACTION_STYLE[family]}" style="fill:{ACTION_STYLE[family]};font-weight:bold">{label}</text>')
        lx += 110

    for panel_i, (path, episode, points) in enumerate(loaded):
        key = model_key(path)
        color = COLORS.get(key, COLORS["other"])
        panel_top = top + panel_i * (panel_h + panel_gap)
        plot_top = panel_top + 22
        plot_bottom = plot_top + panel_plot_h
        y = lambda n, pt=plot_top: pt + panel_plot_h * (max_goal - n) / max(1, max_goal)
        svg.append(f'<rect x="30" y="{panel_top}" width="{width-60}" height="{panel_h}" rx="8" fill="{"#f5f8fb" if panel_i % 2 == 0 else "#fbf8f4"}"/>')
        svg.append(f'<text x="{left}" y="{panel_top+16}" class="panel" fill="{color}" style="fill:{color}">Run {panel_i+1}: {esc(episode["model"])} · {len(points)-1} calls · final {points[-1]["count"]}/{max_goal} · goal_reached={episode["goal_reached"]}</text>')
        # Same y and x scale in each panel.
        for n in range(max_goal + 1):
            yy = y(n)
            svg.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" stroke="#dce2e8" stroke-width="1"/>')
            svg.append(f'<text x="{left-15}" y="{yy+4:.1f}" text-anchor="end" class="tick">{n}</text>')
        tick_values = list(range(0, max_t+1, 2))
        if max_t not in tick_values:
            tick_values.append(max_t)
        for t in tick_values:
            xx=x(t)
            svg.append(f'<line x1="{xx:.1f}" y1="{plot_top}" x2="{xx:.1f}" y2="{plot_bottom}" stroke="#edf0f2" stroke-width="1"/>')
            svg.append(f'<text x="{xx:.1f}" y="{plot_bottom+16}" text-anchor="middle" class="tick">{t}</text>')
        svg.append(f'<text x="22" y="{plot_top+panel_plot_h/2}" transform="rotate(-90 22 {plot_top+panel_plot_h/2})" text-anchor="middle" class="axis">Goal facts currently true</text>')
        coords = " ".join(f"{x(p['t']):.1f},{y(p['count']):.1f}" for p in points)
        svg.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.6" opacity="0.95"/>')
        for p in points:
            xx, yy = x(p["t"]), y(p["count"])
            if p["status"] == "start":
                fill, stroke, marker = color, color, "circle"
            else:
                fill, stroke, marker = STATUS_STYLE.get(p["status"], (color, color, "circle"))
                # Use model color for successful points; status colors flag errors.
                if p["status"] == "success":
                    fill = stroke = color
            if marker == "circle":
                svg.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="4.2" fill="{fill}" stroke="#fff" stroke-width="1"/>')
            elif marker == "x":
                svg.append(f'<path d="M {xx-4:.1f} {yy-4:.1f} L {xx+4:.1f} {yy+4:.1f} M {xx+4:.1f} {yy-4:.1f} L {xx-4:.1f} {yy+4:.1f}" stroke="{stroke}" stroke-width="2.2"/>')
            elif marker == "diamond":
                svg.append(f'<polygon points="{xx:.1f},{yy-5:.1f} {xx+5:.1f},{yy:.1f} {xx:.1f},{yy+5:.1f} {xx-5:.1f},{yy:.1f}" fill="{fill}"/>')
            elif marker == "square":
                svg.append(f'<rect x="{xx-4:.1f}" y="{yy-4:.1f}" width="8" height="8" fill="{fill}"/>')
            elif marker == "triangle":
                svg.append(f'<polygon points="{xx:.1f},{yy-5:.1f} {xx+5:.1f},{yy+4:.1f} {xx-5:.1f},{yy+4:.1f}" fill="{fill}"/>')
            if p["t"] > 0:
                # Attach the vertical action label to its point, slightly above the marker.
                label_y = yy - 10
                family_color = ACTION_STYLE.get(p["family"], ACTION_STYLE["other"])
                svg.append(f'<text x="{xx+1:.1f}" y="{label_y:.1f}" class="action" text-anchor="start" transform="rotate(-90 {xx+1:.1f} {label_y:.1f})" fill="{family_color}" style="fill:{family_color};font-weight:bold">{esc(p["action"])}</text>')
        svg.append(f'<line x1="{left}" y1="{plot_bottom}" x2="{width-right}" y2="{plot_bottom}" stroke="#34495e"/>')
        svg.append(f'<text x="{left+plot_w/2}" y="{plot_bottom+34}" text-anchor="middle" class="axis">Tool-call time</text>')

    # Outcome legend and status summaries.
    ly2 = height - 78
    svg.append(f'<text x="{left}" y="{ly2}" class="legend" font-weight="bold">Outcome markers:</text>')
    legend_items = [("circle", "success", "#2e8b57"), ("x", "precondition_failed", "#c0392b"), ("diamond", "unknown_action", "#8e44ad"), ("square", "malformed_action", "#7f8c8d"), ("triangle", "transient_failure", "#e67e22")]
    lx = left + 130
    for shape, label, color in legend_items:
        if shape == "circle":
            svg.append(f'<circle cx="{lx}" cy="{ly2-4}" r="4" fill="{color}"/>')
        elif shape == "x":
            svg.append(f'<path d="M {lx-4} {ly2-8} L {lx+4} {ly2} M {lx+4} {ly2-8} L {lx-4} {ly2}" stroke="{color}" stroke-width="2"/>')
        elif shape == "diamond":
            svg.append(f'<polygon points="{lx},{ly2-9} {lx+5},{ly2-4} {lx},{ly2+1} {lx-5},{ly2-4}" fill="{color}"/>')
        elif shape == "square":
            svg.append(f'<rect x="{lx-4}" y="{ly2-8}" width="8" height="8" fill="{color}"/>')
        else:
            svg.append(f'<polygon points="{lx},{ly2-10} {lx+5},{ly2-1} {lx-5},{ly2-1}" fill="{color}"/>')
        svg.append(f'<text x="{lx+10}" y="{ly2}" class="legend">{label}</text>')
        lx += 190

    svg.append('</svg>')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(svg))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='task-005')
    parser.add_argument('--condition', default='baseline')
    parser.add_argument('--data', nargs='+', type=Path, default=DEFAULT_DATASETS)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    render(args.task, args.condition, args.data, args.out)
    print(args.out.resolve())


if __name__ == '__main__':
    main()
