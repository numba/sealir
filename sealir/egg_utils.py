from __future__ import annotations

import json
import html
from io import StringIO
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


class ECTree:
    _ecdata: EClassData
    _parents: dict[Term, set[Term]]
    _parent_eclasses: dict[str, set[str]]

    def __init__(self, ecdata):
        self._ecdata = ecdata
        self._compute_parents()

    def _compute_parents(self):
        parentmap = defaultdict(set)
        parent_ecmap = defaultdict(set)

        self._parents = parentmap
        self._parent_eclasses = parent_ecmap

        for term in self._ecdata.terms.values():
            for child in term.children:
                parentmap[child].add(term)
                parent_ecmap[child.eclass].add(term.eclass)

    def root_eclasses(self) -> set[str]:
        """Find root eclasses
        """
        return {ec for ec in self._ecdata.eclasses
                if not self._parent_eclasses[ec]}

    def write_html_root(self, rootec: str) -> StringIO:
        buf = StringIO()
        ecd = self._ecdata

        drawn = set()

        def write_eclass(ec):
            escaped_ec = html.escape(ec)
            # logic to skip
            # - skip if eclass has more than one member
            # - and it is already drawn
            if len(ecd.eclasses[ec]) > 1 and ec in drawn:
                buf.write(f"<div class='eclass'>")
                buf.write(f"<a class='eclass_name' data-eclass='{escaped_ec}'>[{escaped_ec}]</a>")
                buf.write(f"</div>")
                return
            else:
                drawn.add(ec)
            # draw box
            buf.write(f"<div class='eclass' data-eclass='{escaped_ec}'>")
            buf.write(f"<span class='eclass_name'>{html.escape(ec)}</span>")
            for termref in ecd.eclasses[ec]:
                term = ecd.terms[termref.key]
                write_term(term)
            buf.write("</div>")

        def write_term(term: Term):
            escaped_op = html.escape(term.op)
            buf.write(f"<div class='term' data-term-op='{escaped_op}' >")
            buf.write(escaped_op)

            buf.write(f"<div class='content'>")

            for child in term.children:
                write_eclass(child.eclass)
            buf.write("</div>")
            buf.write("</div>")

        write_eclass(rootec)
        return buf
