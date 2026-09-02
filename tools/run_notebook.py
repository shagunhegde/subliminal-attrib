"""Execute a notebook, saving the output after every cell.

`jupyter nbconvert --execute` writes the result only when the whole notebook
finishes, so a stage that takes an hour shows nothing until it is over -- and
shows nothing at all if it dies partway. That is the wrong trade for this
project: every notebook here is a sequence of gates, and the interesting moment
is usually the cell that failed.

This runs the same kernel, one cell at a time, and rewrites the output notebook
after each. Open the output in JupyterLab and reload to watch it fill in; on a
crash, every cell up to the failure is on disk with its output intact.

    python tools/run_notebook.py notebooks/pivot/03_blackbox.ipynb \\
        --out-dir /workspace/executed

Exits non-zero if a cell raised, so it composes with `&&` in a shell chain.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def run(source: Path, out: Path, workdir: Path, timeout: int = -1) -> int:
    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError

    nb = nbformat.read(source, as_version=4)
    out.parent.mkdir(parents=True, exist_ok=True)

    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        allow_errors=False,
        resources={"metadata": {"path": str(workdir)}},
    )

    code_cells = [i for i, c in enumerate(nb.cells) if c.cell_type == "code"]
    started = time.time()
    print(f"{source.name}: {len(code_cells)} code cells -> {out}", flush=True)

    failed_at: int | None = None
    with client.setup_kernel():
        for i, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue
            n = code_cells.index(i) + 1
            head = (cell.source.strip().splitlines() or [""])[0][:64]
            print(f"  [{n}/{len(code_cells)}] {head}", flush=True)
            try:
                client.execute_cell(cell, i)
            except CellExecutionError as e:
                failed_at = n
                print(f"  !! cell {n} raised: {str(e).splitlines()[-1][:200]}", flush=True)
                break
            finally:
                # Save after every cell, success or failure, so the partial run
                # is always readable.
                nbformat.write(nb, out)

    nbformat.write(nb, out)
    elapsed = (time.time() - started) / 60
    if failed_at is not None:
        print(f"FAILED at cell {failed_at} after {elapsed:.1f} min; partial output in {out}")
        return 1
    print(f"OK: {len(code_cells)} cells in {elapsed:.1f} min -> {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("notebook", type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("/workspace/executed"))
    ap.add_argument("--workdir", type=Path, default=None,
                    help="kernel working directory (default: the repo root)")
    ap.add_argument("--timeout", type=int, default=-1, help="per-cell seconds; -1 for none")
    args = ap.parse_args()

    source = args.notebook.resolve()
    workdir = args.workdir or Path(__file__).resolve().parents[1]
    return run(source, args.out_dir / source.name, workdir, args.timeout)


if __name__ == "__main__":
    sys.exit(main())
