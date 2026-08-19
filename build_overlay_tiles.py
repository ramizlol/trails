#!/usr/bin/env python3
"""Build V41 gray-trail overlay tiles with exact routing-edge identities.

Run beside main.py:
    python build_overlay_tiles.py

Input:
    routing_tiles/manifest.json
    routing_tiles/*.pkl

Output:
    overlay_tiles/manifest.json
    overlay_tiles/<tile-id>.json.gz

Each routing pickle is opened ONE AT A TIME, natural-trail geometry is extracted,
then the NetworkX graph is released before the next tile. Render can serve these
small gzip JSON files without unpickling NetworkX graphs just to draw gray trails.
"""

import gc
import gzip
import json
import os
import pickle
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ROUTING_DIR = BASE_DIR / "routing_tiles"
ROUTING_MANIFEST = ROUTING_DIR / "manifest.json"
OVERLAY_DIR = BASE_DIR / "overlay_tiles"
OVERLAY_MANIFEST = OVERLAY_DIR / "manifest.json"


def geometry_to_coords(geometry):
    if geometry is None:
        return []
    if hasattr(geometry, "coords"):
        return [(float(x), float(y)) for x, y in geometry.coords]
    if hasattr(geometry, "geoms"):
        coords = []
        for geom in geometry.geoms:
            part = geometry_to_coords(geom)
            if coords and part and coords[-1] == part[0]:
                coords.extend(part[1:])
            else:
                coords.extend(part)
        return coords
    return []


def oriented_edge_coords(G, u, v, data):
    coords = geometry_to_coords(data.get("geometry"))
    if not coords:
        coords = [
            (float(G.nodes[u]["x"]), float(G.nodes[u]["y"])),
            (float(G.nodes[v]["x"]), float(G.nodes[v]["y"])),
        ]

    ux = float(G.nodes[u]["x"])
    uy = float(G.nodes[u]["y"])
    first = abs(coords[0][0] - ux) + abs(coords[0][1] - uy)
    last = abs(coords[-1][0] - ux) + abs(coords[-1][1] - uy)
    if last < first:
        coords.reverse()
    return coords


def row_bounds(row):
    return row.get("nominal_bounds") or row.get("bounds") or row.get("actual_graph_bounds")


def main():
    if not ROUTING_MANIFEST.exists():
        raise SystemExit(f"Missing {ROUTING_MANIFEST}")

    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    routing_manifest = json.loads(ROUTING_MANIFEST.read_text(encoding="utf-8"))
    output_rows = []

    rows = list(routing_manifest.get("tiles", []))
    print(f"Routing tiles: {len(rows)}")

    for index, row in enumerate(rows, 1):
        tile_id = str(row.get("id", ""))
        routing_file = str(row.get("file", ""))
        if not tile_id or not routing_file:
            continue

        source = ROUTING_DIR / routing_file
        out_name = f"{tile_id}.json.gz"
        output = OVERLAY_DIR / out_name

        print(f"[{index}/{len(rows)}] {tile_id}")

        if int(row.get("trail_edges", 0) or 0) <= 0 or not source.exists():
            payload = {"tile_id": tile_id, "allowed_trails": [], "trail_records": [], "allowed_trail_segments": 0}
        else:
            with source.open("rb") as f:
                G = pickle.load(f)

            segments = []
            trail_records = []
            seen = set()
            for u, v, key, data in G.edges(keys=True, data=True):
                if str(data.get("route_class", "trail")) != "trail":
                    continue
                physical = (
                    min(int(u), int(v)),
                    max(int(u), int(v)),
                    round(float(data.get("length", 0) or 0), 1),
                )
                if physical in seen:
                    continue
                seen.add(physical)
                coords = oriented_edge_coords(G, u, v, data)
                if len(coords) >= 2:
                    geometry = [[float(lat), float(lon)] for lon, lat in coords]
                    segments.append(geometry)
                    trail_records.append({
                        "tile_id": tile_id,
                        "u": int(u),
                        "v": int(v),
                        "key": str(key),
                        "geometry": geometry,
                    })

            payload = {
                "tile_id": tile_id,
                "allowed_trails": segments,
                "trail_records": trail_records,
                "allowed_trail_segments": len(segments),
            }

            del G
            gc.collect()

        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        with gzip.open(output, "wb", compresslevel=6) as f:
            f.write(raw)

        output_rows.append({
            "id": tile_id,
            "file": out_name,
            "bounds": row_bounds(row),
            "segments": int(payload["allowed_trail_segments"]),
            "size_bytes": output.stat().st_size,
        })
        del payload, raw
        gc.collect()

    manifest = {
        "schema": "trail-overlay-tile-manifest-v1",
        "routing_manifest_schema": routing_manifest.get("schema"),
        "tile_count": len(output_rows),
        "total_segments": sum(r["segments"] for r in output_rows),
        "total_size_bytes": sum(r["size_bytes"] for r in output_rows),
        "coverage_nominal": routing_manifest.get("coverage_nominal"),
        "tiles": output_rows,
    }
    OVERLAY_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nDONE")
    print(f"Overlay tiles: {len(output_rows)}")
    print(f"Segments: {manifest['total_segments']}")
    print(f"Compressed size: {manifest['total_size_bytes']/1024/1024:.2f} MB")
    print(f"Manifest: {OVERLAY_MANIFEST}")


if __name__ == "__main__":
    main()