# Study 09: A BFO-grounded toy ontology for car-crash news

## In one paragraph

This small study asks whether Basic Formal Ontology (BFO) can give a useful, disciplined starting point for describing car-crash news. It builds a tiny OWL ontology with domain terms such as `TrafficCollision`, `Vehicle`, `Person`, and `TransportSite`, then places those terms under selected BFO terms. A hand-written example models a train-car crash report. This is a teaching proof of concept, not a finished traffic ontology, not a fully annotated corpus, and not a reasoner-validated ontology.

## What the words mean

An ontology is a carefully defined set of kinds of things and relationships between them. It is more specific than a list of labels: it says, for example, that a collision is a process, and that a vehicle can take part in that process.

BFO (Basic Formal Ontology) is an upper-level ontology. It supplies broad categories used across many fields, such as `process`, `object`, and `site`. This study adds a domain-level ontology: the more specific vocabulary needed for crash news. The aim is to reuse BFO's broad categories rather than invent a separate top-level scheme.

OWL is a standard format for machine-readable ontologies. A reasoner is software that checks logical consequences and detects some contradictions. The validator in this study checks XML structure and selected links to BFO; it is not a reasoner.

## Research question and limits

Question: Can a small, readable vocabulary for crash-news events be aligned to BFO without confusing events with the people, vehicles, and places involved?

The supplied JSONL file has 13 records and about 5,100 characters of news text. A first-pass read suggests that record 8 describes a road obstruction and prayer, not a vehicle collision. The sample also includes a train colliding with a car. This exposed two important scope choices: not every road incident is a crash, and a car-crash collection may contain crashes involving non-road vehicles. The ontology therefore uses the broader parent `TrafficCollision` and a narrower `RoadTrafficCrash` class. This is a modeling proposal, not an official or externally reviewed classification.

The study models one example as an illustration. It does not claim that the existing `entities` and `relations` fields provide complete or reliable crash annotations. In the supplied records, those fields often miss the vehicles, people, and outcomes that the prose contains. Do not interpret the example individuals as a corrected annotation of the full dataset.

## Files

- `ontology/car-crash.owl`: OWL in RDF/XML, with labels, plain-language definitions, BFO alignment, and example individuals.
- `annotations/news-annotations.jsonl`: one evidence-linked structured annotation for each of the 13 messages.
- `scripts/validate.py`: a Python-standard-library structural check. It fetches the pinned BFO OWL source unless a local copy is provided with `--bfo`.
- `scripts/answer_queries.py`: runnable examples of questions answered from the annotations.
- `results/query-results.txt`: output from those example questions.
- `results/corpus-review.txt`: first-pass manual scope count and the selected example record.
- `results/validation.txt`: output from the ontology structural validation.

## How to read the modeling decisions

1. `TrafficCollision` is a `process` (BFO:0000015). A process is something that happens over time. A crash is an event, not a physical object.
2. `RoadTrafficCrash` is a more specific kind of `TrafficCollision`. The hierarchy leaves room for rail collisions while allowing a separate road-only category.
3. `Vehicle` and `Person` are placed under BFO `object` (BFO:0000030). An object is a material entity with its own unified identity. This toy model deliberately does not split vehicle subtypes beyond `RoadVehicle`.
4. `TransportSite` is placed under BFO `site` (BFO:0000029). A site is a place where something can be located. This is a simplifying choice: a road itself is a physical infrastructure object, while a named location or crossing can be treated as a site. Better data could require both classes.
5. `hasParticipant` is an easy-to-read local name for BFO `has participant` (BFO:0000057). `occursAt` is a local name for BFO `occurs in` (BFO:0000066). They are declared equivalent in OWL, so software can treat either name as the same relation.
6. `InjuryProcess` and `DeathProcess` are also processes. They are not asserted to happen for every crash. Domain properties connect a crash to a reported outcome, but the comments warn that a reported association alone does not establish causation.
7. The sample instance graph describes a report of a freight train and passenger car collision at a railway crossing. It demonstrates typing a particular event and linking participants and place. A separate death individual is only a vocabulary example; it is not asserted as an outcome of that crash.

The JSONL format has one JSON object per message. Each object contains:

- a `scope` value (`CrashNews` or `RelatedNonCrashIncident`);
- `events`, each with a local ID, ontology class, evidence phrase, and a certainty/source label;
- `entities`, such as people, vehicles, and places, with evidence and optional role/count;
- `relations`, which connect those IDs using ontology relation names.

The evidence field keeps the annotation auditable: a reader can check why a label was assigned. `preliminary` and `witness_report` preserve the report's uncertainty or source where visible. They are not numerical confidence scores. A missing injury event means the article did not state an injury outcome in this annotation; it does not mean that nobody was hurt. The explicit `NoInjuryReported` outcome is used only for record 4, whose report says nobody was injured.

The schema has intentionally been extended for the corpus: vehicle subtypes (car, truck, bus, train, etc.), injury and death processes, hospitalization, traffic disruption, obstruction, and notable actions such as overtaking and emergency braking. These domain classes are modeled as BFO processes or objects as appropriate. A generic `hasReportedOutcome` relation records that the article associates an outcome with a crash; it does not claim scientifically or legally proven causation. Driver/passenger roles and quantities are kept as annotation fields in this pilot rather than fully modeled as OWL role individuals and quantified collections.

## Example questions and answers

Run `python scripts/answer_queries.py` to reproduce these queries over the annotation file:

| Question | Answer in this 13-message sample |
|---|---|
| Which messages describe a crash? | 12: records 1–7 and 9–13. Record 8 is a related road obstruction, not a reported crash. |
| Which reports state that someone was injured? | 7: records 2, 5, 6, 7, 10, 12, and 13. |
| Which reports state a death? | 2: records 1 and 3. |
| Which report explicitly says nobody was injured? | Record 4. |
| Which reports mention traffic disruption? | Records 11 and 12. |
| Which crash report involves a train? | Record 4. |
| Which crash reports link to a death outcome affecting a driver? | Records 1 and 3. Follow `crash → hasReportedOutcome → death event → affects → person`, then check that person's `Driver` role. |

The last question is a multi-hop graph query, not just a check for a `DeathProcess` label: it joins a crash to its reported death event, follows that death event to the affected person, and filters on the person's role. This makes it possible to state exactly why each result qualifies. The current runner implements this query by following those annotation links; it does not call an OWL reasoner.

These are practical questions because each requires linking the story's events, participants, and outcomes, rather than finding one word. For example, injury wording varies: a person “was injured,” “received injuries,” or was taken to hospital. A structured annotation makes the extraction decision and its supporting phrase visible. The current query script is a small exact query runner, not a sophisticated natural-language question-answering system.

## How to test the ontology's value

The demonstration is that one explicit vocabulary supports reusable questions across varied wording and helps separate important cases: crash versus non-crash; vehicle versus person versus place; confirmed statement versus preliminary or witness-reported statement; no injuries reported versus no injury information present. It also exposes limits and annotation choices for review.

This is not evidence that an ontology outperforms every alternative. A well-designed database schema could answer these same questions. The ontology-specific benefit to test next is whether its declared class hierarchy and BFO links improve consistency, reuse, and validation when new vehicle types, event types, or datasets are added. The present query script follows the annotation vocabulary directly; it does not invoke an OWL reasoner, so these answers demonstrate the value of explicit structured semantics, not automated logical inference.

## Reproduce

Requirements: Python 3.10+ (only the standard library) and network access to GitHub for downloading BFO during ontology validation. No package installation, GPU, or API key is needed.

From this directory, run:

```sh
python scripts/answer_queries.py
python scripts/validate.py
```

To validate offline after obtaining the exact pinned BFO file, run:

```sh
python scripts/validate.py --bfo /path/to/bfo-core.owl
```

The script checks that the XML parses; the pinned source contains the BFO class/property IRIs used; the selected domain classes have the intended direct BFO parents; the import is pinned; and the example individuals are present. It does not check every OWL rule, infer class memberships, test whether the ontology is consistent under a reasoner, or prove that the definitions are domain-correct.

## Result and interpretation

The study produced a compact, readable BFO alignment and a structural check that can be rerun. The useful result is primarily methodological: separating what happens (a process) from who or what participates (objects) and where it happens (a site) prevents several common modeling errors. The news sample also shows why scope definitions matter: an obstruction is not necessarily a crash, and a crash can involve a train.

The evidence is weak by design. There is one small corpus, a single example, no independent ontology review, no full annotation exercise, and no OWL reasoner run. The study supports feasibility of a first draft, not the claim that this is the best ontology or that it would improve extraction performance.

## Next useful checks

- Ask a domain expert whether the scope should include rail crashes and non-collision incidents.
- Annotate several reports independently and compare disagreements before expanding the ontology.
- Decide how to represent time, uncertainty, causes, injuries, fatalities, and multiple collisions in one story.
- Run an OWL 2 reasoner and review the BFO mapping with an ontology specialist before treating the file as more than a prototype.
- Consider whether roads should be physical infrastructure objects, sites, or both in the intended application.

## Source and attribution

Basic Formal Ontology 2020 source: <https://github.com/BFO-ontology/BFO-2020/tree/bd1fb61c5fa4e358fc9690f0d214b1ac63fae594/21838-2/owl>. This study imports the upstream `bfo-core.owl`; it does not redistribute that file. News text comes from the user-provided local JSONL corpus and is not copied into the repository.
