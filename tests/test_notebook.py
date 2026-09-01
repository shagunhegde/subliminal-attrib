"""The notebook must run top to bottom.

Cells are inserted programmatically as phases land, and a mis-targeted insertion
puts a cell before the one that defines its inputs -- which fails only at run
time, on Colab, after a long setup. This catches it here instead.
"""

import ast
import json
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "01_pipeline.ipynb"


def _code_cells():
    nb = json.loads(NOTEBOOK.read_text())
    return [
        "".join(c["source"])
        for c in nb["cells"]
        if c["cell_type"] == "code" and "".join(c["source"]).strip()
    ]


def test_notebook_is_valid_json_with_cells():
    nb = json.loads(NOTEBOOK.read_text())
    assert nb["nbformat"] == 4
    assert len(nb["cells"]) > 0
    assert all(c["cell_type"] in ("code", "markdown") for c in nb["cells"])


def test_every_code_cell_parses():
    """IPython magics (%pip, !cmd) are stripped before parsing."""
    for i, src in enumerate(_code_cells()):
        cleaned = "\n".join(
            "pass" if line.lstrip().startswith(("%", "!")) else line
            for line in src.splitlines()
        )
        try:
            ast.parse(cleaned)
        except SyntaxError as e:
            pytest.fail(f"code cell {i} does not parse: {e}")


def test_names_are_defined_before_use():
    """Walk the cells in order and check that each name a cell reads was bound
    by an earlier cell (or by that cell itself)."""
    bound: set[str] = set(dir(__builtins__)) | {
        "get_ipython", "__name__", "__file__", "In", "Out", "_",
    }
    import builtins

    bound |= set(dir(builtins))

    for i, src in enumerate(_code_cells()):
        cleaned = "\n".join(
            "pass" if line.lstrip().startswith(("%", "!")) else line
            for line in src.splitlines()
        )
        tree = ast.parse(cleaned)

        # Everything this cell binds, including comprehension and loop targets.
        binds: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                binds.add(node.id)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                binds.add(node.name)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    binds.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(node, ast.ExceptHandler) and node.name:
                binds.add(node.name)
            elif isinstance(node, ast.arg):
                binds.add(node.arg)

        used = {
            n.id for n in ast.walk(tree)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
        }
        missing = used - binds - bound
        assert not missing, (
            f"code cell {i} uses {sorted(missing)} before any earlier cell defines them.\n"
            f"first line: {src.strip().splitlines()[0]}"
        )
        bound |= binds
