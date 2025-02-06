import logging
from pprint import pformat
from typing import TypeAlias

from sealir import ase

_logger = logging.getLogger(__name__)


_DEBUG = False
_DEBUG_HTML = False

SExpr: TypeAlias = ase.SExpr


def pp(expr: SExpr):
    if _DEBUG:
        print(
            pformat(ase.as_tuple(expr, -1, dedup=True))
            .replace(",", "")
            .replace("'", "")
        )
        # print(pformat(expr.as_dict()))


def internal_prefix(name: str) -> str:
    # "!" will always sort to the front of all visible characters.
    return "!" + name


from .restructuring import (
    SourceInfoDebugger,
    convert_to_rvsdg,
    restructure_source,
)
from .scfg_to_sexpr import convert_to_sexpr
