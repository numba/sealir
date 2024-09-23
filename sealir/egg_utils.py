from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass

from egglog import EGraph


@dataclass(frozen=True)
class Term:
    key: str
    type: str
    op: str
    children: tuple[TermRef, ...]
    eclass: str
    cost: int

    @property
    def literal(self) -> int | str:
        return self.op

    def ref(self) -> TermRef:
        return TermRef(self.key, self.type, self.op, self.eclass)


@dataclass(frozen=True)
class TermRef:
    key: str
    type: str
    op: str
    eclass: str


class EClassData:
    _terms: dict[str, Term]
    _eclasses: dict[str, set[TermRef]]

    def __init__(self, terms: dict[str, Term]):
        self._terms = terms
        self._eclasses = defaultdict(set)
        for term in terms.values():
            self._eclasses[term.eclass].add(term.ref())

    @property
    def terms(self) -> dict[str, Term]:
        return self._terms

    @property
    def eclasses(self) -> dict[str, set[TermRef]]:
        return self._eclasses


def extract_eclasses(egraph: EGraph) -> EClassData:
    serialized = egraph._egraph.serialize([])
    serialized.map_ops(egraph._state.op_mapping())
    jsdata = json.loads(serialized.to_json())
    terms = reconstruct(jsdata["nodes"], jsdata["class_data"])
    return EClassData(terms)


def reconstruct(
    nodes: dict[str, dict], class_data: dict[str, str]
) -> dict[str, Term]:
    done: dict[str, Term] = {}

    # First pass map keys ot TermRefs
    key_to_termref = {}
    for key in nodes:
        node_data = nodes[key]
        eclass = node_data["eclass"]
        typ = class_data[eclass]["type"]
        op = node_data["op"].strip('"')

        key_to_termref[key] = TermRef(key=key, type=typ, op=op, eclass=eclass)

    # Second pass map keys to Terms
    for key, termref in key_to_termref.items():
        node_data = nodes[key]
        cost = node_data["cost"]
        children = tuple(key_to_termref[x] for x in node_data["children"])
        done[key] = Term(
            key, termref.type, termref.op, children, termref.eclass, cost
        )

    return done
