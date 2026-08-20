#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import sys
import time
from pathlib import Path

import networkx as nx

BASE_DIR = Path(__file__).resolve().parent
OSM_TILE_DIR = BASE_DIR / "osm_tiles"
OUTPUT_DIR = BASE_DIR / "routing_tiles"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"

DENSE_PARENT_REPLACEMENTS = {
    "-112.25_33.25": "phoenix_small",
    "-112.75_33.25": "dense_small",
    "-111.75_33.25": "dense_small",
}

MAX_CONNECTORS_SMALL_TILE = 10
MAX_CONNECTORS_LARGE_TILE = 16
DEFAULT_PARENT_SIZE_DEG = 0.5
DEFAULT_CHILD_SIZE_DEG = 0.125


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    p.add_argument("--tile", default=None)
    p.add_argument("--trails-only", action="store_true")
    return p.parse_args()


def import_app():
    import main as app

    required = [
        "load_local_highway_graph_from_pbf",
        "_classified_subgraph_from_local_osm",
        "add_local_dem_edge_elevations",
        "edge_routing_cost",
        "configure_osmnx_trail_tags",
        "trail_only_graph",
        "_component_unique_trail_length",
        "_offline_connector_candidate_pairs",
        "_build_walk_node_index",
        "_find_local_connector_path",
        "_merge_connector_path",
        "get_dem_signature",
    ]
    missing = [name for name in required if not hasattr(app, name)]
    if missing:
        raise RuntimeError("main.py missing: " + ", ".join(missing))
    return app


def tile_id_from_path(path: Path) -> str:
    name = path.name
    if not name.endswith(".osm.pbf"):
        raise ValueError(path)
    return name[:-8]


def parse_tile_origin(tile_id: str):
    lon_s, lat_s = tile_id.split("_", 1)
    return float(lon_s), float(lat_s)


def tile_size_for_path(path: Path) -> float:
    if path.parent.name in {"phoenix_small", "dense_small"}:
        return DEFAULT_CHILD_SIZE_DEG
    tile_id = tile_id_from_path(path)
    lon_s, lat_s = tile_id.split("_", 1)
    decimals = max(
        len(lon_s.split(".")[1]) if "." in lon_s else 0,
        len(lat_s.split(".")[1]) if "." in lat_s else 0,
    )
    return DEFAULT_CHILD_SIZE_DEG if decimals >= 3 else DEFAULT_PARENT_SIZE_DEG


def nominal_bounds(path: Path):
    west, south = parse_tile_origin(tile_id_from_path(path))
    size = tile_size_for_path(path)
    return {"west": west, "south": south, "east": west + size, "north": south + size}


def child_tiles_exist_for_parent(parent_id: str, child_dir_name: str) -> bool:
    child_dir = OSM_TILE_DIR / child_dir_name
    if not child_dir.exists():
        return False
    west, south = parse_tile_origin(parent_id)
    east = west + DEFAULT_PARENT_SIZE_DEG
    north = south + DEFAULT_PARENT_SIZE_DEG
    count = 0
    for path in child_dir.glob("*.osm.pbf"):
        try:
            cw, cs = parse_tile_origin(tile_id_from_path(path))
        except Exception:
            continue
        if west - 1e-9 <= cw < east - 1e-9 and south - 1e-9 <= cs < north - 1e-9:
            count += 1
    return count >= 16


def discover_input_tiles():
    if not OSM_TILE_DIR.exists():
        raise RuntimeError(f"Missing {OSM_TILE_DIR}")

    selected = []
    for path in sorted(OSM_TILE_DIR.glob("*.osm.pbf")):
        tile_id = tile_id_from_path(path)
        if tile_id == "test_tile":
            continue
        child_dir = DENSE_PARENT_REPLACEMENTS.get(tile_id)
        if child_dir and child_tiles_exist_for_parent(tile_id, child_dir):
            print(f"Using subdivided children instead of parent {tile_id}")
            continue
        selected.append(path)

    for child_dir_name in sorted(set(DENSE_PARENT_REPLACEMENTS.values())):
        child_dir = OSM_TILE_DIR / child_dir_name
        if child_dir.exists():
            selected.extend(sorted(child_dir.glob("*.osm.pbf")))

    unique = {str(p.resolve()): p for p in selected}
    result = list(unique.values())
    result.sort(key=lambda p: (nominal_bounds(p)["south"], nominal_bounds(p)["west"]))
    return result


def graph_actual_bounds(G):
    if G.number_of_nodes() == 0:
        return None
    xs, ys = [], []
    for _, d in G.nodes(data=True):
        try:
            xs.append(float(d["x"]))
            ys.append(float(d["y"]))
        except Exception:
            pass
    if not xs:
        return None
    return {"west": min(xs), "south": min(ys), "east": max(xs), "north": max(ys)}


def graph_class_counts(G):
    trails = connectors = 0
    for _, _, _, data in G.edges(keys=True, data=True):
        if str(data.get("route_class", "trail")) == "connector":
            connectors += 1
        else:
            trails += 1
    return trails, connectors


def add_sparse_connectors(app, trail_G, walk_G, max_connectors):
    if trail_G.number_of_edges() == 0 or walk_G.number_of_edges() == 0:
        return trail_G, {"connector_count": 0, "connector_checks": 0, "connector_path_meters": 0.0,
                         "components_before": 0, "components_after": 0, "errors": []}

    G = trail_G.copy()
    trail_base = app.trail_only_graph(G)
    components = [set(c) for c in nx.connected_components(trail_base.to_undirected(as_view=True))]
    component_lengths = {
        i: app._component_unique_trail_length(G, nodes)
        for i, nodes in enumerate(components)
    }
    candidate_rows = app._offline_connector_candidate_pairs(G, components, component_lengths)
    walk_index = app._build_walk_node_index(walk_G)
    union = nx.utils.UnionFind(range(len(components)))

    connector_count = connector_checks = 0
    connector_path_meters = 0.0
    errors = []

    print(f"  sparse connectors: {len(components)} components / {len(candidate_rows)} candidate pairs")

    for _, gap_m, a, b, source_hint, target_hint in candidate_rows:
        if connector_count >= max_connectors:
            break
        if union[a] == union[b]:
            continue

        connector_checks += 1
        result, error = app._find_local_connector_path(
            G, walk_G, walk_index, source_hint, target_hint, gap_m
        )
        if result is None:
            if error:
                errors.append(f"{a}<->{b}: {error}")
            continue

        copied = app._merge_connector_path(G, result, connector_count + 1)
        union.union(a, b)
        connector_count += 1
        connector_path_meters += float(copied)
        print(f"    connector {connector_count}: {copied:.0f} m")

    for _, _, _, data in G.edges(keys=True, data=True):
        if str(data.get("route_class", "trail")) == "connector":
            data["ascent_m"] = float(data.get("ascent_m", 0) or 0)
            data["descent_m"] = float(data.get("descent_m", 0) or 0)
            data["elevation_sample_count"] = int(float(data.get("elevation_sample_count", 0) or 0))
            data["routing_cost"] = float(app.edge_routing_cost(data))

    before = len(components)
    after = nx.number_connected_components(G.to_undirected(as_view=True)) if G.number_of_nodes() else 0
    return G, {
        "connector_count": connector_count,
        "connector_checks": connector_checks,
        "connector_path_meters": connector_path_meters,
        "components_before": before,
        "components_after": after,
        "errors": errors[:100],
    }


def atomic_pickle_dump(obj, output_path: Path):
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, output_path)


def normalize_tile_metadata(app, G, source_path, tile_id, dem_samples, connector_info):
    G.graph["tile_id"] = tile_id
    G.graph["tile_schema"] = "trail-routing-tile-v1"
    G.graph["tile_source_pbf"] = source_path.name
    G.graph["dem_signature"] = app.get_dem_signature()
    G.graph["master_elevation_precomputed"] = "1"
    G.graph["tile_elevation_unique_samples"] = int(dem_samples)
    G.graph["embedded_dem_edge_profiles"] = "1"
    G.graph["offline_connectors_prebuilt"] = "1"
    G.graph["offline_connector_count"] = int(connector_info["connector_count"])
    G.graph["offline_connector_checks"] = int(connector_info["connector_checks"])
    G.graph["offline_connector_path_meters"] = float(connector_info["connector_path_meters"])
    G.graph["offline_components_before"] = int(connector_info["components_before"])
    G.graph["offline_components_after"] = int(connector_info["components_after"])
    G.graph["offline_connector_errors_json"] = json.dumps(connector_info["errors"])
    G.graph["overpass_used"] = "0"


def process_tile(app, source_path, force=False, trails_only=False):
    tile_id = tile_id_from_path(source_path)
    output_path = OUTPUT_DIR / f"{tile_id}.pkl"

    if output_path.exists() and not force:
        print(f"SKIP {tile_id}: already built")
        with output_path.open("rb") as f:
            G = pickle.load(f)
        trail_edges, connector_edges = graph_class_counts(G)
        return {
            "id": tile_id, "file": output_path.name,
            "source_pbf": str(source_path.relative_to(BASE_DIR)),
            "nominal_bounds": nominal_bounds(source_path),
            "actual_graph_bounds": graph_actual_bounds(G),
            "nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
            "trail_edges": trail_edges, "connector_edges": connector_edges,
            "size_bytes": output_path.stat().st_size, "skipped_existing": True,
        }

    started = time.perf_counter()
    print("\n" + "=" * 72)
    print(f"BUILD {tile_id} from {source_path.relative_to(BASE_DIR)}")
    print("=" * 72)

    raw = app.load_local_highway_graph_from_pbf(pbf_path=str(source_path))
    print(f"  raw: {raw.number_of_nodes()} nodes / {raw.number_of_edges()} edges")

    trail_G = app._classified_subgraph_from_local_osm(raw, "trail")
    walk_G = nx.MultiDiGraph() if trails_only else app._classified_subgraph_from_local_osm(raw, "connector")
    print(f"  trails: {trail_G.number_of_nodes()} nodes / {trail_G.number_of_edges()} edges")
    if not trails_only:
        print(f"  walk temp: {walk_G.number_of_nodes()} nodes / {walk_G.number_of_edges()} edges")

    del raw
    gc.collect()

    if trail_G.number_of_edges() == 0:
        G = trail_G
        dem_samples = 0
        connector_info = {"connector_count": 0, "connector_checks": 0, "connector_path_meters": 0.0,
                          "components_before": 0, "components_after": 0, "errors": []}
    else:
        print("  sampling DEM...")
        trail_G, dem_samples = app.add_local_dem_edge_elevations(trail_G)
        for _, _, _, data in trail_G.edges(keys=True, data=True):
            data["route_class"] = "trail"
            data["routing_cost"] = float(app.edge_routing_cost(data))

        if trails_only or walk_G.number_of_edges() == 0:
            G = trail_G
            components = nx.number_connected_components(G.to_undirected(as_view=True))
            connector_info = {"connector_count": 0, "connector_checks": 0, "connector_path_meters": 0.0,
                              "components_before": components, "components_after": components, "errors": []}
        else:
            max_connectors = MAX_CONNECTORS_SMALL_TILE if tile_size_for_path(source_path) <= 0.1250001 else MAX_CONNECTORS_LARGE_TILE
            G, connector_info = add_sparse_connectors(app, trail_G, walk_G, max_connectors)

    del walk_G
    gc.collect()

    normalize_tile_metadata(app, G, source_path, tile_id, dem_samples, connector_info)
    atomic_pickle_dump(G, output_path)

    elapsed = time.perf_counter() - started
    trail_edges, connector_edges = graph_class_counts(G)
    row = {
        "id": tile_id, "file": output_path.name,
        "source_pbf": str(source_path.relative_to(BASE_DIR)),
        "nominal_bounds": nominal_bounds(source_path),
        "actual_graph_bounds": graph_actual_bounds(G),
        "nodes": int(G.number_of_nodes()), "edges": int(G.number_of_edges()),
        "trail_edges": int(trail_edges), "connector_edges": int(connector_edges),
        "dem_samples": int(dem_samples),
        "connector_count": int(connector_info["connector_count"]),
        "connector_path_meters": round(float(connector_info["connector_path_meters"]), 1),
        "size_bytes": output_path.stat().st_size,
        "build_seconds": round(elapsed, 2), "skipped_existing": False,
    }
    print(f"  SAVED {output_path.name}: {output_path.stat().st_size/1024/1024:.2f} MB | {G.number_of_nodes()} nodes / {G.number_of_edges()} edges | {elapsed:.1f}s")
    return row


def write_manifest(app, rows, started):
    rows = sorted(rows, key=lambda r: (r["nominal_bounds"]["south"], r["nominal_bounds"]["west"], r["id"]))
    nonempty = [r for r in rows if r.get("edges", 0) > 0]
    coverage = None
    if nonempty:
        coverage = {
            "west": min(r["nominal_bounds"]["west"] for r in nonempty),
            "south": min(r["nominal_bounds"]["south"] for r in nonempty),
            "east": max(r["nominal_bounds"]["east"] for r in nonempty),
            "north": max(r["nominal_bounds"]["north"] for r in nonempty),
        }
    manifest = {
        "schema": "trail-routing-tile-manifest-v1",
        "elevation_storage": "edge-samples-v47",
        "generated_unix": time.time(),
        "dem_file": os.path.basename(app.DEM_PATH),
        "dem_signature": app.get_dem_signature(),
        "tile_count": len(rows),
        "nonempty_tile_count": len(nonempty),
        "coverage_nominal": coverage,
        "total_size_bytes": sum(int(r.get("size_bytes", 0)) for r in rows),
        "build_elapsed_seconds": round(time.perf_counter() - started, 2),
        "tiles": rows,
    }
    tmp = MANIFEST_PATH.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, MANIFEST_PATH)
    return manifest


def main():
    args = parse_args()
    started = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    app = import_app()
    app.configure_osmnx_trail_tags()

    if not Path(app.DEM_PATH).exists():
        raise RuntimeError(f"DEM missing: {app.DEM_PATH}")

    tiles = discover_input_tiles()
    if args.tile:
        tiles = [p for p in tiles if tile_id_from_path(p) == args.tile]
        if not tiles:
            raise RuntimeError(f"Tile {args.tile!r} not found")

    print(f"Effective input tiles: {len(tiles)}")
    for p in tiles:
        print(f"  {tile_id_from_path(p):20s} {p.stat().st_size/1024/1024:7.2f} MB  {p.relative_to(BASE_DIR)}")

    rows = []
    for i, path in enumerate(tiles, 1):
        print(f"\n### TILE {i}/{len(tiles)} ###")
        rows.append(process_tile(app, path, force=args.force, trails_only=args.trails_only))
        write_manifest(app, rows, started)
        gc.collect()

    manifest = write_manifest(app, rows, started)
    print("\nALL DONE")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Tiles: {manifest['tile_count']}")
    print(f"Processed storage: {manifest['total_size_bytes']/1024/1024:.2f} MB")
    print(f"Elapsed: {manifest['build_elapsed_seconds']/60:.1f} min")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Cancelled.", file=sys.stderr)
        raise SystemExit(130)