#!/usr/bin/env python3
"""Run example structured questions over the hand-annotated news JSONL."""
import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "annotations" / "news-annotations.jsonl"
rows = [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines() if line.strip()]

def events_of_type(row, *types):
    return [e for e in row["events"] if e["type"] in types]

def main():
    crash = [r for r in rows if r["scope"] == "CrashNews"]
    injured = [r for r in rows if events_of_type(r, "InjuryProcess")]
    fatalities = [r for r in rows if events_of_type(r, "DeathProcess")]
    no_injury = [r for r in rows if any(e.get("outcome") == "NoInjuryReported" for e in r["events"])]
    traffic_disruption = [r for r in rows if events_of_type(r, "TrafficDisruption")]
    train_crashes = [r for r in rows if any(e["type"] == "Train" for e in r["entities"]) and events_of_type(r, "TrafficCollision", "RoadTrafficCrash")]
    driver_fatality = []
    for row in rows:
        event_by_id = {e["id"]: e for e in row["events"]}
        entity_by_id = {e["id"]: e for e in row["entities"]}
        relations = row["relations"]
        crash_ids = {e["id"] for e in events_of_type(row, "RoadTrafficCrash", "TrafficCollision")}
        death_ids = {e["id"] for e in events_of_type(row, "DeathProcess")}
        has_death_outcome = {
            rel["object"] for rel in relations
            if rel["subject"] in crash_ids and rel["predicate"] == "hasReportedOutcome"
            and rel["object"] in death_ids
        }
        affects_driver = any(
            rel["subject"] in has_death_outcome and rel["predicate"] == "affects"
            and rel["object"] in entity_by_id and entity_by_id[rel["object"]].get("role") == "Driver"
            for rel in relations
        )
        if affects_driver:
            driver_fatality.append(row)
    print(f"Q1 Which messages are in crash scope? {len(crash)} of {len(rows)}: " + ", ".join(str(r["record_id"]) for r in crash))
    print(f"Q2 Which reports state injuries? {len(injured)}: " + ", ".join(str(r["record_id"]) for r in injured))
    print(f"Q3 Which reports state a death? {len(fatalities)}: " + ", ".join(str(r["record_id"]) for r in fatalities))
    print(f"Q4 Which explicitly report no injuries? {len(no_injury)}: " + ", ".join(str(r["record_id"]) for r in no_injury))
    print(f"Q5 Which reports describe traffic disruption? {len(traffic_disruption)}: " + ", ".join(str(r["record_id"]) for r in traffic_disruption))
    print(f"Q6 Which crash reports involve a train? {len(train_crashes)}: " + ", ".join(str(r["record_id"]) for r in train_crashes))
    print(f"Q7 Which crash reports link to a death outcome affecting a driver? {len(driver_fatality)}: " + ", ".join(str(r["record_id"]) for r in driver_fatality))
    print("  Two relation hops: crash --hasReportedOutcome--> death --affects--> driver")
    print("\nEvidence samples:")
    for rid, etype in [(1, "DeathProcess"), (4, "InjuryAssessment"), (8, "RoadObstruction"), (11, "RoadTrafficCrash")]:
        row = next(r for r in rows if r["record_id"] == rid)
        ev = next(e for e in row["events"] if e["type"] == etype)
        print(f"  record {rid} {etype} [{ev['certainty']}]: {ev['evidence']}")

if __name__ == "__main__":
    main()
