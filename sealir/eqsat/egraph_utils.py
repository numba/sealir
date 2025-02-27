from typing import TypedDict


class NodeDict(TypedDict):
    children: list[str]
    cost: float
    eclass: str
    op: str
    subsumed: bool


class ClassDataDict(TypedDict):
    type: str


class EGraphJsonDict(TypedDict):
    nodes: dict[str, NodeDict]
    root_eclasses: list[str]
    class_data: dict[str, ClassDataDict]
