"""Every notebook must run top to bottom.

Cells are inserted programmatically as phases land, and a mis-targeted insertion
puts a cell before the one that defines its inputs -- which fails only at run
time, on the rented GPU, after a long setup. This catches it here instead.

The PLAN v2 notebooks are one file per stage, so each is walked independently:
they run in separate kernels and share nothing but the artifacts on disk.
"""

import ast
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "notebooks"
NOTEBOOKS = sorted(ROOT.glob("*.ipynb")) + sorted(ROOT.glob("pivot/*.ipynb"))


def _code_cells(path):
    nb = json.loads(Path(path).read_text())
    return [
        "".join(c["source"])
        for c in nb["cells"]
        if c["cell_type"] == "code" and "".join(c["source"]).strip()
    ]


def _clean(src):
    """IPython magics (%pip, !cmd) are stripped before parsing."""
    return "\n".join(
        "pass" if line.lstrip().startswith(("%", "!")) else line
        for line in src.splitlines()
    )


def test_the_pivot_notebooks_are_all_present():
    """Stages 00-09; a missing one means the plan and the repo have drifted."""
    names = sorted(p.name[:2] for p in ROOT.glob("pivot/*.ipynb"))
    assert names == [f"{i:02d}" for i in range(10)]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_is_valid_json_with_cells(path):
    nb = json.loads(path.read_text())
    assert nb["nbformat"] == 4
    assert len(nb["cells"]) > 0
    assert all(c["cell_type"] in ("code", "markdown") for c in nb["cells"])


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_every_code_cell_parses(path):
    for i, src in enumerate(_code_cells(path)):
        try:
            ast.parse(_clean(src))
        except SyntaxError as e:
            pytest.fail(f"{path.name} code cell {i} does not parse: {e}")


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_names_are_defined_before_use(path):
    """Walk the cells in order and check that each name a cell reads was bound
    by an earlier cell (or by that cell itself)."""
    bound: set[str] = set(dir(__builtins__)) | {
        "get_ipython", "__name__", "__file__", "In", "Out", "_",
    }
    import builtins

    bound |= set(dir(builtins))

    for i, src in enumerate(_code_cells(path)):
        tree = ast.parse(_clean(src))

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
            f"{path.name} code cell {i} uses {sorted(missing)} before any earlier "
            f"cell defines them.\nfirst line: {src.strip().splitlines()[0]}"
        )
        bound |= binds
