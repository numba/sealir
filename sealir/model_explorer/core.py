from __future__ import annotations

import pickle

from egglog import EGraph

from sealir.grammar import NamedSExpr

_extensions = [
    "sealir_model_explorer_adapters.egglog_adapter",
    "sealir_model_explorer_adapters.rvsdg_adapter",
]


def prepare_egraph(egraph: EGraph, filepath: str) -> str:
    """
    Prepare an EGraph for visualization by serializing it to a JSON file.

    Args:
        egraph (EGraph): The EGraph to be serialized
        filepath (str): The base filepath where the JSON representation will
                        be saved. File extension is always appended.

    Returns:
        str: The filename of the JSON EGraph file
    """
    out = egraph._serialize(
        n_inline_leaves=1, split_primitive_outputs=False
    ).to_json()
    filename = f"{filepath}.egraph_json"
    with open(filename, "w") as fout:
        fout.write(out)
    return filename


def prepare_rvsdg(rvsdg_root: NamedSExpr, filepath: str) -> str:
    """
    Prepare an RVSDG for visualization by serializing it to a pickled file.

    Args:
        rvsdg_root (NamedSExpr): The RVSDG root to be serialized
        filepath (str): The base filepath where the pickled representation will
                        be saved. File extension is always appended.

    Returns:
        str: The filename of the pickled RVSDG file
    """
    filename = f"{filepath}.rvsdg"
    payload = rvsdg_root._serialize()
    with open(filename, "wb") as fout:
        pickle.dump(payload, file=fout)
    return filename


def visualize_egraph(egraph: EGraph, *, filepath: str) -> None:
    """
    Visualize an EGraph by serializing it to a JSON file and using model
    explorer to render it.

    Args:
        egraph (EGraph): The EGraph to visualize
        filepath (str): The base filepath where the JSON representation will be
                        saved. File extensions is always appended.
    """
    import model_explorer  # type: ignore[import-not-found]

    model_explorer.visualize(
        prepare_egraph(egraph, filepath), extensions=_extensions
    )


def visualize_rvsdg(rvsdg_root: NamedSExpr, *, filepath: str) -> None:
    """
    Visualize an RVSDG by serializing it to a pickled file and using model
    explorer to render it.

    Args:
        rvsdg_root (NamedSExpr): The RVSDG root to visualize
        filepath (str): The base filepath where the pickled representation will
                        be saved. File extension is always appended.
    """
    import model_explorer

    model_explorer.visualize(
        prepare_rvsdg(rvsdg_root, filepath), extensions=_extensions
    )
