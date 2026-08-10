"""One-time offline builder for Trail Running Creator v14.

Run from the same folder as main.py (or v14.py):
    python build_master_graph_v14.py

If master_trails.graphml already exists, this keeps it and only builds the new
sparse production routing graph with useful walking connectors:
    master_routing.graphml   <- commit this to GitHub
    master_routing.pkl       <- optional faster local cache

If the trail master is missing/incompatible, v14 will rebuild it first.
"""

from pathlib import Path
import importlib.util
import sys


def load_app_module():
    here = Path(__file__).resolve().parent
    candidates = [here / "v14.py", here / "main.py"]

    for path in candidates:
        if not path.exists():
            continue
        spec = importlib.util.spec_from_file_location("trail_creator_v14_build_target", path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module, path

    raise RuntimeError("Could not find main.py or v14.py in this folder.")


def main():
    module, path = load_app_module()
    if not hasattr(module, "build_master_routing_graph"):
        raise RuntimeError(f"{path.name} does not contain the v14 offline routing builder.")

    print(f"Using app code from: {path.name}")
    module.build_master_routing_graph(rebuild_trails=False)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"BUILD FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
