"""Memory-bounded tiled build for Trail Running Creator.
 
Run from the same folder as main.py:
    python build_master_graph_tiled.py
 
Processes central-az.osm.pbf as four smaller tiles (one at a time) instead of
one big region, so peak memory during the osmium->XML->osmnx step is bounded
by the single largest tile rather than the whole region. Produces the same
master_trails.graphml / master_routing.graphml as python main.py --build-routing.
 
Split central-az.osm.pbf into the four tiles first:
    osmium extract -b -113.0520833,32.9354167,-111.90125,33.6101389 --strategy=simple central-az.osm.pbf -o tile-sw.osm.pbf --set-bounds --overwrite
    osmium extract -b -111.90125,32.9354167,-110.7504167,33.6101389 --strategy=simple central-az.osm.pbf -o tile-se.osm.pbf --set-bounds --overwrite
    osmium extract -b -113.0520833,33.6101389,-111.90125,34.2848611 --strategy=simple central-az.osm.pbf -o tile-nw.osm.pbf --set-bounds --overwrite
    osmium extract -b -111.90125,33.6101389,-110.7504167,34.2848611 --strategy=simple central-az.osm.pbf -o tile-ne.osm.pbf --set-bounds --overwrite
"""
 
from pathlib import Path
import importlib.util
import sys
 
TILE_FILES = ["tile-sw.osm.pbf", "tile-se.osm.pbf", "tile-nw.osm.pbf", "tile-ne.osm.pbf"]
 
 
def load_app_module():
    here = Path(__file__).resolve().parent
    candidates = [here / "v14.py", here / "main.py"]
 
    for path in candidates:
        if not path.exists():
            continue
        spec = importlib.util.spec_from_file_location("trail_creator_build_target", path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module, path
 
    raise RuntimeError("Could not find main.py or v14.py in this folder.")
 
 
def main():
    module, path = load_app_module()
    if not hasattr(module, "load_local_highway_graph_tiled"):
        raise RuntimeError(
            f"{path.name} does not have load_local_highway_graph_tiled(). "
            "Add the tiled loader function to main.py first."
        )
 
    here = Path(__file__).resolve().parent
    tile_paths = [str(here / name) for name in TILE_FILES]
    missing = [p for p in tile_paths if not Path(p).exists()]
    if missing:
        raise RuntimeError(
            "Missing tile file(s): " + ", ".join(missing) +
            ". Run the osmium extract commands in this script's docstring first."
        )
 
    print(f"Using app code from: {path.name}")
    print(f"Tiles: {', '.join(TILE_FILES)}")
 
    merged_source_G = module.load_local_highway_graph_tiled(tile_paths)
    module.build_master_routing_graph(rebuild_trails=True, source_G=merged_source_G)
 
 
if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"TILED BUILD FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
 