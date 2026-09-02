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


def _streaming_client_class():
    """A NotebookClient that echoes cell stdout as it arrives.

    Saving after each cell is not enough on its own: nbclient buffers a cell's
    output into the notebook and only writes it when the cell returns, so a cell
    that runs for an hour -- the gradient cache in 06, the batch poll in 03 --
    is completely silent while it is the only thing happening. `process_message`
    is the one seam where the kernel's stream messages are visible in flight.
    """
    from nbclient import NotebookClient

    class StreamingNotebookClient(NotebookClient):
        def process_message(self, msg, cell, cell_index):
            if msg.get("msg_type") == "stream":
                text = msg.get("content", {}).get("text", "")
                sys.stdout.write(text)
                sys.stdout.flush()
            return super().process_message(msg, cell, cell_index)

    return StreamingNotebookClient


def run(source: Path, out: Path, workdir: Path, timeout: int = -1) -> int:
    import nbformat
    from nbclient.exceptions import CellExecutionError

    nb = nbformat.read(source, as_version=4)
    out.parent.mkdir(parents=True, exist_ok=True)

    client = _streaming_client_class()(
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
