"""Execute the portable notebook in a new kernel of the active Python."""
from pathlib import Path
import os
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parent


def sanitize(node):
    if isinstance(node, str):
        return node.replace(str(ROOT), ".").replace(str(Path(sys.prefix)), "<python-environment>")
    if isinstance(node, list):
        return [sanitize(x) for x in node]
    if isinstance(node, dict):
        return {k: sanitize(v) for k, v in node.items()}
    return node


def main():
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
    path = ROOT / "xgrammar2_reproduction.ipynb"
    document = nbformat.read(path, as_version=4)
    for cell in document.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client = NotebookClient(document, km=manager, timeout=1800,
        resources={"metadata": {"path": str(ROOT)}})
    client.on_cell_execute = lambda cell, cell_index: print("Start cell", cell_index, flush=True)
    try:
        client.execute()
    finally:
        nbformat.write(nbformat.from_dict(sanitize(dict(document))), path)
    print("Notebook executed and saved")


if __name__ == "__main__":
    main()
