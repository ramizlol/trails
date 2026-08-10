"""One-time helper for Trail Running Creator v11.

Run from the same folder as main.py (or v11.py):
    python build_master_graph.py

It downloads the TIFF-wide natural-trail network from OpenStreetMap, filters it,
precomputes edge elevation heuristics from output_USGS10m.tif, and writes:
    master_trails.graphml   <- commit this to GitHub
    master_trails.pkl       <- optional faster local cache
"""

from pathlib import Path
import importlib.util
import sys


def load_app_module():
    here = Path(__file__).resolve().parent
    candidates = [here / "main.py", here / "v11.py"]

    for path in candidates:
        if not path.exists():
            continue
        spec = importlib.util.spec_from_file_location("trail_creator_build_target", path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module, path

    raise RuntimeError("Could not find main.py or v11.py in this folder.")


def main():
    module, path = load_app_module()
    if not hasattr(module, "build_master_trail_graph"):
        raise RuntimeError(f"{path.name} does not contain the v11 offline builder.")

    print(f"Using app code from: {path.name}")
    module.build_master_trail_graph()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"BUILD FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
