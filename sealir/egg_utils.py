from __future__ import annotations

import sys
import json
import html
from io import StringIO
from collections import defaultdict
from dataclasses import dataclass

from egglog import EGraph


MAX = sys.maxsize

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
    _depthmap: dict[str, int]

    def __init__(self, ecdata):
        self._ecdata = ecdata
        self._compute_parents()
        self._compute_depth()

    def _compute_depth(self) -> None:
        """Computes the depth of each eclass in the ECTree.

        The depth is the minimum acyclic distance from the eclass to any of its
        leaf eclasses.
        """
        depthmap = self._depthmap = {}

        frontiers: set[str] = set()
        leaves = set()

        for term in self._ecdata.terms.values():
            if not term.children:
                ec = term.eclass
                frontiers.add(ec)
                leaves.add(ec)
                depthmap[ec] = 0


        visited = set()
        while frontiers:
            new_frontiers = set()

            for ec in frontiers:
                visited.add(ec)

                d = 1 + depthmap[ec]
                for p in self._parent_eclasses[ec]:
                    if ec in leaves:
                        depthmap.setdefault(p, d)
                    else:
                        depthmap[p] = min(depthmap.get(p, MAX), d)

                    new_frontiers.add(p)

            frontiers = new_frontiers - visited


    def _compute_parents(self):
        self._parents = defaultdict(set)
        self._parent_eclasses = defaultdict(set)

        parentmap = self._parents
        parent_ecmap = self._parent_eclasses

        for term in self._ecdata.terms.values():
            for child in term.children:
                parentmap[self._ecdata.terms[child.key]].add(term)
                parent_ecmap[child.eclass].add(term.eclass)

    def root_eclasses(self) -> set[str]:
        """Find root eclasses
        """
        return {ec for ec in self._ecdata.eclasses
                if not self._parent_eclasses[ec]}

    def leave_eclasses(self) -> set[str]:
        """Find leaves
        """
        leaves = set(self._ecdata.eclasses)
        for ec in self._ecdata.eclasses:
            leaves -= self._parent_eclasses[ec]
        return leaves

    def write_html_root(self) -> StringIO:
        buf = StringIO()
        ordered = sorted(self._depthmap.items(), key=lambda x: (x[1], x[0]),
                         reverse=True)
        for ec, _ in ordered:
            self.write_html_eclass(ec, buf)
        return buf

    def write_html_eclass(self, rootec: str, buf: StringIO):
        ecd = self._ecdata

        drawn = set()

        def write_eclass(ec, *, ref_only=False):
            depth = self._depthmap[ec]
            escaped_ec = html.escape(f"{ec}({depth})")
            # logic to skip
            # - skip if eclass has more than one member
            # - and it is already drawn
            if ref_only or (len(ecd.eclasses[ec]) > 1 and ec in drawn):
                buf.write(f"<div class='eclass'>")
                buf.write(f"<a class='eclass_name' data-eclass='{escaped_ec}'>[{escaped_ec}]</a>")
                buf.write(f"</div>")
                return
            else:
                drawn.add(ec)
            # draw box
            buf.write(f"<div class='eclass' data-eclass='{escaped_ec}'>")
            buf.write(f"<span class='eclass_name'>{escaped_ec}</span>")
            terms = [ecd.terms[termref.key] for termref in ecd.eclasses[ec]]
            depthmap = self._depthmap
            def child_depth(term: Term):
                return max([0, *(depthmap[ch.eclass] for ch in term.children)])
            for term in sorted(terms, key=child_depth):
                write_term(term)
            buf.write("</div>")

        def write_term(term: Term):
            escaped_op = html.escape(term.op)
            buf.write(f"<div class='term' data-term-op='{escaped_op}' >")
            buf.write(escaped_op)

            buf.write(f"<div class='content'>")

            for child in term.children:
                write_eclass(child.eclass, ref_only=True)
            buf.write("</div>")
            buf.write("</div>")

        write_eclass(rootec)
