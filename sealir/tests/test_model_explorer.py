try:
    from sealir_model_explorer_adapters.egglog_adapter.main import (
        EgraphJsonAdapter,
    )
    from sealir_model_explorer_adapters.rvsdg_adapter.main import RvsdgAdapter
except ImportError:
    has_adapters = False
else:
    has_adapters = True

import tempfile
from pathlib import Path

import egglog
import pytest

from sealir.eqsat.rvsdg_convert import egraph_conversion
from sealir.eqsat.rvsdg_eqsat import GraphRoot
from sealir.model_explorer.core import prepare_egraph, prepare_rvsdg
from sealir.tests.test_rvsdg_egraph_roundtrip import frontend

_skip_reason = "model_explorer adapters not installed"


skip_if_missing_adapters = pytest.mark.skipif(
    not has_adapters, reason=_skip_reason
)


def _example(n):
    c = 0
    for i in range(n):
        c += i
    return c + i


@skip_if_missing_adapters
def test_adapt_rvsdg():
    rvsdg, _ = frontend(_example)

    with tempfile.TemporaryDirectory() as tempdir:
        print("tempdir", tempdir)
        filepath = prepare_rvsdg(rvsdg, Path(tempdir) / "temp_rvsdg.rvsdg")
        # Exercise the adapter
        RvsdgAdapter().convert(filepath, {})


@skip_if_missing_adapters
def test_adapt_egraph():

    rvsdg, _ = frontend(_example)

    memo = egraph_conversion(rvsdg)
    func = memo[rvsdg]

    egraph = egglog.EGraph()
    egraph.let("root", GraphRoot(func))

    with tempfile.TemporaryDirectory() as tempdir:
        print("tempdir", tempdir)
        filepath = prepare_egraph(
            egraph, Path(tempdir) / "temp_egraph.egraph_json"
        )
        # Exercise the adapter
        EgraphJsonAdapter().convert(filepath, {})
