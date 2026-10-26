#!/usr/bin/env python3
"""Dependency-free structural checks for this toy OWL file."""
from __future__ import annotations
import argparse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

OWL = "http://www.w3.org/2002/07/owl#"
RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"
BFO = "http://purl.obolibrary.org/obo/"
BASE = "https://raw.githubusercontent.com/BFO-ontology/BFO-2020/bd1fb61c5fa4e358fc9690f0d214b1ac63fae594/21838-2/owl/bfo-core.owl"
HERE = Path(__file__).resolve().parents[1]
ONTOLOGY = HERE / "ontology" / "car-crash.owl"


def parse(path: Path) -> ET.Element:
    return ET.parse(path).getroot()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bfo", type=Path, help="local copy of the pinned BFO core OWL file")
    args = ap.parse_args()
    root = parse(ONTOLOGY)
    bfo_path = args.bfo
    if bfo_path:
        bfo_root = parse(bfo_path)
    else:
        with urllib.request.urlopen(BASE, timeout=30) as response:
            bfo_root = ET.fromstring(response.read())

    def full_about(element: ET.Element) -> str:
        value = element.get(f"{{{RDF}}}about", "")
        return value if "://" in value else "https://example.org/car-crash-news" + value

    bfo_classes = {full_about(x) for x in bfo_root.findall(f"{{{OWL}}}Class")}
    bfo_properties = {full_about(x) for x in bfo_root.findall(f"{{{OWL}}}ObjectProperty")}
    local_classes = {full_about(x) for x in root.findall(f"{{{OWL}}}Class")}
    labels = {x.text for x in bfo_root.findall(f".//{{{RDFS}}}label") if x.text}
    expected_bfo = {
        "process": "http://purl.obolibrary.org/obo/BFO_0000015",
        "object": "http://purl.obolibrary.org/obo/BFO_0000030",
        "site": "http://purl.obolibrary.org/obo/BFO_0000029",
    }
    errors = []
    for label, iri in expected_bfo.items():
        if iri not in bfo_classes:
            errors.append(f"BFO class missing from pinned source: {iri}")
    class_map = {
        "#TrafficCollision": expected_bfo["process"],
        "#Vehicle": expected_bfo["object"],
        "#Person": expected_bfo["object"],
        "#TransportSite": expected_bfo["site"],
        "#InjuryProcess": expected_bfo["process"],
        "#DeathProcess": expected_bfo["process"],
    }
    for fragment, parent in class_map.items():
        child = "https://example.org/car-crash-news" + fragment
        node = next((x for x in root.findall(f"{{{OWL}}}Class") if full_about(x) == child), None)
        parents = {x.get(f"{{{RDF}}}resource") for x in node.findall(f"{{{RDFS}}}subClassOf")} if node is not None else set()
        if node is None or parent not in parents:
            errors.append(f"Expected {fragment} to be a subclass of {parent}")
    for prop in root.findall(f"{{{OWL}}}ObjectProperty"):
        for eq in prop.findall(f"{{{OWL}}}equivalentProperty"):
            iri = eq.get(f"{{{RDF}}}resource")
            if iri and iri.startswith(BFO) and iri not in bfo_properties:
                errors.append(f"BFO property missing from pinned source: {iri}")
    imports = [x.get(f"{{{RDF}}}resource") for x in root.findall(f".//{{{OWL}}}imports")]
    if imports != [BASE]:
        errors.append("Ontology import does not match the pinned BFO source")
    if "process" not in labels or "object" not in labels or "site" not in labels:
        errors.append("Pinned BFO labels are missing")
    individuals = root.findall(f"{{{OWL}}}NamedIndividual")
    if len(individuals) < 6:
        errors.append("Expected the example named individuals")
    if errors:
        print("FAIL")
        print("\n".join("- " + e for e in errors))
        return 1
    print(f"PASS: XML parsed; {len(local_classes)} domain classes; {len(individuals)} sample individuals; BFO mappings and imported properties found in pinned BFO core.")
    print("This is a structural check, not a description-logic reasoner or full OWL conformance check.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
