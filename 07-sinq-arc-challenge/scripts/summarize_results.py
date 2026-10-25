#!/usr/bin/env python3
"""Summarize saved AIME exact-match JSONL records without hiding missing rows."""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def summarize(path: Path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    by_dataset = defaultdict(list)
    keys = set()
    duplicates = []
    for row in rows:
        key = (row["dataset"], str(row["id"]))
        if key in keys:
            duplicates.append(key)
        keys.add(key)
        by_dataset[row["dataset"]].append(row)
    result = {"path": str(path), "rows": len(rows), "duplicates": duplicates, "datasets": {}}
    for dataset, items in sorted(by_dataset.items()):
        result["datasets"][dataset] = {
            "n": len(items),
            "correct": sum(row["correct"] is True for row in items),
            "accuracy": sum(row["correct"] is True for row in items) / len(items) if items else None,
            "unparsed": sum(row["prediction"] is None for row in items),
            "truncated": sum(row["finish_reason"] == "length" for row in items),
            "mean_elapsed_s": sum(row["elapsed_s"] for row in items) / len(items) if items else None,
            "ids": sorted(str(row["id"]) for row in items),
        }
    result["overall_correct"] = sum(row["correct"] is True for row in rows)
    result["overall_accuracy"] = result["overall_correct"] / len(rows) if rows else None
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.files:
        print(json.dumps(summarize(path), indent=2))


if __name__ == "__main__":
    main()
