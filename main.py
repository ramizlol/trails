from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import math
import os
import random
import time
import threading
import xml.etree.ElementTree as ET

import networkx as nx
import numpy as np
import osmnx as ox
import rasterio
from pyproj import Transformer
from shapely.geometry import LineString
from rasterio.warp import transform as rio_transform, transform_bounds as rio_transform_bounds


app = FastAPI()


# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEM_PATH = os.path.join(BASE_DIR, "output_USGS10m.tif")

METERS_PER_MILE = 1609.344
FEET_PER_METER = 3.28084

DEFAULT_LAT = 33.586055
DEFAULT_LON = -112.083341

# Sample along trail/GPX geometry every 5 m.
# The source DEM is ~10 m resolution.
ELEVATION_SAMPLE_SPACING_M = 5.0

# GPX points within this distance of an allowed trail count as covered.
GPX_TRAIL_MATCH_TOLERANCE_M = 25.0

MAX_CACHED_GRAPHS = 5
GRAPH_CACHE = {}

# One filtered OSM trail graph covering the entire DEM/TIFF footprint.
# It is loaded/built lazily on the first trail request, then every route request
# extracts only a local subgraph around its start coordinate.
MASTER_GRAPH = None
MASTER_GRAPH_INFO = {}
MASTER_GRAPH_LOCK = threading.Lock()
MASTER_GRAPH_PATH = os.path.join(BASE_DIR, "master_trails_output_USGS10m.graphml")
DEM_BOUNDS_WGS84_CACHE = None

# Cache DEM values by rounded lat/lon. Graph construction already samples
# most trail points, so later route scoring can reuse those values instead
# of reopening/resampling the GeoTIFF for every finalist.
DEM_POINT_CACHE = {}
MAX_DEM_POINT_CACHE = 250000

APP_VERSION = "2026-08-09-v7-full-tiff-master-network"
ELEVATION_SMOOTHING_RADIUS = 5  # 11 points total ~= 55 m at 5 m spacing
PARTIAL_TUNING_MAX_DEFICIT_M = 0.75 * METERS_PER_MILE


# ============================================================
# REQUEST MODELS
# ============================================================

class RouteRequest(BaseModel):
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float
    target_distance_miles: float
    target_gain_ft: float


class TrailNetworkRequest(BaseModel):
    start_lat: float
    start_lon: float
    target_distance_miles: float


# ============================================================
# BASIC ENDPOINT
# ============================================================

@app.get("/")
def home():
    return {
        "status": "Trail Running Creator API is running",
        "version": APP_VERSION,
        "map": "/map",
        "docs": "/docs",
        "dem_info": "/dem-info",
        "default_start": {
            "lat": DEFAULT_LAT,
            "lon": DEFAULT_LON,
        },
    }


# ============================================================
# ROUTE PROFILES
# ============================================================

def get_route_profile(target_distance_miles: float):
    if target_distance_miles < 4.0:
        return {
            "name": "short-closed-beam",
            "search_radius_m": 1800,
            "beam_width": 500,
            "beam_max_steps": 80,
            "max_search_seconds": 20.0,
            "max_expanded_states": 120000,
            "max_closed_candidates": 150,
            "partial_tuning_base_candidates": 32,
            "continue_through_start_below_target": True,
        }

    if target_distance_miles < 8.0:
        return {
            "name": "medium-waypoint",
            "search_radius_m": 2500,
            "attempts": 1200,
            "anchor_counts": [2, 3, 3, 3],
            "min_anchor_distance_m": 150,
            "min_anchor_separation_m": 140,
            # Generate all attempts cheaply, then run the authoritative
            # continuous-route DEM calculation only on the best finalists.
            "accurate_finalists": 40,
            "candidate_pool_multiplier": 3,
        }

    if target_distance_miles < 15.0:
        return {
            "name": "long-waypoint",
            "search_radius_m": 3200,
            "attempts": 900,
            "anchor_counts": [3, 4, 4, 4],
            "min_anchor_distance_m": 300,
            "min_anchor_separation_m": 250,
            "accurate_finalists": 50,
            "candidate_pool_multiplier": 3,
        }

    return {
        "name": "ultra-waypoint",
        "search_radius_m": min(5000, int(target_distance_miles * 300)),
        "attempts": 700,
        "anchor_counts": [4, 4, 5],
        "min_anchor_distance_m": 400,
        "min_anchor_separation_m": 300,
        "accurate_finalists": 60,
        "candidate_pool_multiplier": 3,
    }


# ============================================================
# QUALITY LIMITS
# ============================================================

def get_route_quality_limits(target_distance_miles: float, target_gain_ft: float):
    if target_distance_miles < 4.0:
        distance_error_limit_miles = max(0.15, target_distance_miles * 0.08)
    else:
        distance_error_limit_miles = min(
            0.50,
            max(0.20, target_distance_miles * 0.06),
        )

    if target_gain_ft <= 300:
        gain_error_limit_ft = 100
    elif target_gain_ft <= 1000:
        gain_error_limit_ft = max(125, target_gain_ft * 0.25)
    else:
        gain_error_limit_ft = max(150, target_gain_ft * 0.18)

    return {
        "distance_error_limit_miles": distance_error_limit_miles,
        "gain_error_limit_ft": gain_error_limit_ft,
        "excellent_distance_error_miles": max(
            0.05,
            target_distance_miles * 0.025,
        ),
        "excellent_gain_error_ft": max(
            35,
            target_gain_ft * 0.08,
        ),
    }


# ============================================================
# GENERAL HELPERS
# ============================================================

def normalize_tag_values(value):
    if value is None:
        return set()

    if not isinstance(value, (list, tuple, set)):
        value = [value]

    result = set()

    for item in value:
        if item is None:
            continue

        for part in str(item).split(";"):
            part = part.strip().lower()
            if part:
                result.add(part)

    return result


def undirected_edge_key(u, v):
    return tuple(sorted((int(u), int(v))))


def haversine_meters(lat1, lon1, lat2, lon2):
    radius = 6371000.0

    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2.0) ** 2
    )

    return 2.0 * radius * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def node_angle_from_start(G, start_node, node):
    start_lat = float(G.nodes[start_node]["y"])
    start_lon = float(G.nodes[start_node]["x"])

    lat = float(G.nodes[node]["y"])
    lon = float(G.nodes[node]["x"])

    x = (lon - start_lon) * math.cos(math.radians(start_lat))
    y = lat - start_lat

    return math.atan2(y, x)


# ============================================================
# EDGE HELPERS
# ============================================================

def get_shortest_edge(G, u, v):
    edge_data = G.get_edge_data(u, v)

    if not edge_data:
        return None

    return min(
        edge_data.values(),
        key=lambda edge: float(edge.get("length", float("inf"))),
    )


def path_distance_meters(G, route_nodes):
    total = 0.0

    for i in range(len(route_nodes) - 1):
        edge = get_shortest_edge(G, route_nodes[i], route_nodes[i + 1])
        if edge is not None:
            total += float(edge.get("length", 0) or 0)

    return total


def path_gain_meters(G, route_nodes):
    total = 0.0

    for i in range(len(route_nodes) - 1):
        edge = get_shortest_edge(G, route_nodes[i], route_nodes[i + 1])
        if edge is not None:
            total += float(edge.get("ascent_m", 0) or 0)

    return total


# ============================================================
# OSM EDGE GEOMETRY
# ============================================================

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


def oriented_edge_coords(G, u, v, edge):
    geometry = edge.get("geometry")
    coords = geometry_to_coords(geometry)

    if not coords:
        coords = [
            (
                float(G.nodes[u]["x"]),
                float(G.nodes[u]["y"]),
            ),
            (
                float(G.nodes[v]["x"]),
                float(G.nodes[v]["y"]),
            ),
        ]

    u_lon = float(G.nodes[u]["x"])
    u_lat = float(G.nodes[u]["y"])

    first_lon, first_lat = coords[0]
    last_lon, last_lat = coords[-1]

    first_distance = abs(first_lon - u_lon) + abs(first_lat - u_lat)
    last_distance = abs(last_lon - u_lon) + abs(last_lat - u_lat)

    if last_distance < first_distance:
        coords.reverse()

    return coords


def route_coordinates(G, route_nodes):
    coordinates = []

    for i in range(len(route_nodes) - 1):
        u = route_nodes[i]
        v = route_nodes[i + 1]

        edge = get_shortest_edge(G, u, v)
        if edge is None:
            continue

        for lon, lat in oriented_edge_coords(G, u, v, edge):
            point = {"lat": float(lat), "lon": float(lon)}

            if coordinates:
                previous = coordinates[-1]
                if (
                    abs(previous["lat"] - point["lat"]) < 0.0000001
                    and abs(previous["lon"] - point["lon"]) < 0.0000001
                ):
                    continue

            coordinates.append(point)

    return coordinates


# ============================================================
# EXACT START-POINT INSERTION
# ============================================================

EXACT_POINT_MAX_TRAIL_OFFSET_M = 3.0


def _local_xy_m(lon, lat, ref_lon, ref_lat):
    """Approximate lon/lat as local meters around a reference point."""
    radius = 6371000.0
    x = math.radians(float(lon) - float(ref_lon)) * radius * math.cos(math.radians(float(ref_lat)))
    y = math.radians(float(lat) - float(ref_lat)) * radius
    return x, y


def nearest_position_on_polyline(coords, target_lon, target_lat):
    """Return nearest segment/t/fraction and distance from a point to a lon/lat polyline."""
    if len(coords) < 2:
        return None

    best = None
    cumulative_m = 0.0

    for index in range(len(coords) - 1):
        lon1, lat1 = coords[index]
        lon2, lat2 = coords[index + 1]

        ax, ay = _local_xy_m(lon1, lat1, target_lon, target_lat)
        bx, by = _local_xy_m(lon2, lat2, target_lon, target_lat)

        dx = bx - ax
        dy = by - ay
        denom = dx * dx + dy * dy

        if denom <= 1e-12:
            t = 0.0
        else:
            # Target point is local origin (0, 0).
            t = -(ax * dx + ay * dy) / denom
            t = max(0.0, min(1.0, t))

        px = ax + t * dx
        py = ay + t * dy
        distance_m = math.hypot(px, py)

        segment_m = haversine_meters(lat1, lon1, lat2, lon2)
        along_m = cumulative_m + segment_m * t

        if best is None or distance_m < best["distance_m"]:
            projected_lon = float(lon1) + (float(lon2) - float(lon1)) * t
            projected_lat = float(lat1) + (float(lat2) - float(lat1)) * t
            best = {
                "segment_index": index,
                "t": float(t),
                "distance_m": float(distance_m),
                "along_m": float(along_m),
                "projected_lon": float(projected_lon),
                "projected_lat": float(projected_lat),
            }

        cumulative_m += segment_m

    if best is not None:
        best["total_m"] = float(cumulative_m)

    return best


def split_polyline_at_position(coords, segment_index, t, split_lon, split_lat):
    """Split a directed lon/lat polyline at a location inside one segment."""
    split_point = (float(split_lon), float(split_lat))
    left = list(coords[: segment_index + 1])
    right = list(coords[segment_index + 1 :])

    if not left:
        left = [split_point]
    if not right:
        right = [split_point]

    if not left or haversine_meters(left[-1][1], left[-1][0], split_point[1], split_point[0]) > 0.05:
        left.append(split_point)
    else:
        left[-1] = split_point

    if not right or haversine_meters(split_point[1], split_point[0], right[0][1], right[0][0]) > 0.05:
        right.insert(0, split_point)
    else:
        right[0] = split_point

    return left, right


def edge_attributes_for_split_part(original_data, coords):
    """Copy edge tags and recompute geometry/length/elevation for one split piece."""
    attrs = dict(original_data)
    attrs["geometry"] = LineString(coords)
    attrs["length"] = float(polyline_distance_meters(coords))

    samples = densify_polyline(coords, ELEVATION_SAMPLE_SPACING_M)
    if len(samples) < 2:
        samples = list(coords)

    elevations = elevations_for_coords(samples)
    smoothed = smooth_elevations(
        elevations,
        radius=ELEVATION_SMOOTHING_RADIUS,
    )
    ascent_m, descent_m = calculate_ascent_descent(smoothed)

    attrs["ascent_m"] = float(ascent_m)
    attrs["descent_m"] = float(descent_m)
    attrs["elevation_sample_count"] = len(samples)
    attrs["virtual_split_edge"] = True
    return attrs


def insert_exact_routing_point(G, lat, lon):
    """
    Insert a temporary graph node at the requested coordinate when it lies on
    an allowed trail edge (within EXACT_POINT_MAX_TRAIL_OFFSET_M).

    The selected physical edge is split in both travel directions. This means
    loop search starts and finishes at the requested coordinate instead of
    snapping to an OSM junction tens of meters away.

    The cached graph is never mutated: this function returns a copy.
    """
    lat = float(lat)
    lon = float(lon)

    projected = ox.projection.project_graph(G)
    projected_crs = projected.graph.get("crs")
    if projected_crs is None:
        raise HTTPException(
            status_code=500,
            detail="Could not determine projected graph CRS for exact start insertion.",
        )

    transformer = Transformer.from_crs(
        "EPSG:4326",
        projected_crs,
        always_xy=True,
    )
    x, y = transformer.transform(lon, lat)

    try:
        edge_id, edge_distance_m = ox.distance.nearest_edges(
            projected,
            X=float(x),
            Y=float(y),
            return_dist=True,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not find nearest trail edge for exact start: {exc}",
        )

    values = list(edge_id) if not isinstance(edge_id, tuple) else list(edge_id)
    if len(values) < 3:
        raise HTTPException(
            status_code=500,
            detail="Nearest trail edge returned an invalid edge identifier.",
        )

    u, v, key = values[:3]
    edge_distance_m = float(np.atleast_1d(edge_distance_m)[0])

    # If the requested point is not actually on/very near a trail, keep the
    # former nearest-node behavior rather than inventing an off-trail connector.
    if edge_distance_m > EXACT_POINT_MAX_TRAIL_OFFSET_M:
        fallback = int(ox.distance.nearest_nodes(G, X=lon, Y=lat))
        fallback_lat = float(G.nodes[fallback]["y"])
        fallback_lon = float(G.nodes[fallback]["x"])
        return G, fallback, {
            "exact_inserted": False,
            "trail_offset_m": round(edge_distance_m, 2),
            "routing_lat": fallback_lat,
            "routing_lon": fallback_lon,
            "routing_offset_m": round(
                haversine_meters(lat, lon, fallback_lat, fallback_lon),
                2,
            ),
            "source_edge": [int(u), int(v), str(key)],
        }

    selected_data = G.get_edge_data(u, v, key)
    if selected_data is None:
        selected_data = get_shortest_edge(G, u, v)
    if selected_data is None:
        raise HTTPException(
            status_code=500,
            detail="Could not read the nearest trail edge for exact start insertion.",
        )

    selected_coords = oriented_edge_coords(G, u, v, selected_data)
    nearest = nearest_position_on_polyline(selected_coords, lon, lat)
    if nearest is None:
        raise HTTPException(
            status_code=500,
            detail="Could not locate the requested start along the nearest trail geometry.",
        )

    # Use the requested coordinate itself as the temporary routing node. The GPX
    # start is essentially on the edge, so this preserves the exact GPX start.
    split_lon = lon
    split_lat = lat

    H = G.copy()

    virtual_node = -1
    while virtual_node in H:
        virtual_node -= 1

    elevation = elevations_for_coords([(split_lon, split_lat)])[0]
    H.add_node(
        virtual_node,
        x=float(split_lon),
        y=float(split_lat),
        elevation=float(elevation),
        virtual_exact_point=True,
    )

    # Split every directed copy of this same physical u-v trail segment that
    # passes through the requested point. Normally this is u->v and v->u.
    split_count = 0
    original_pair = {int(u), int(v)}
    candidates = []

    for a, b in [(u, v), (v, u)]:
        edge_dict = G.get_edge_data(a, b) or {}
        for candidate_key, data in edge_dict.items():
            coords = oriented_edge_coords(G, a, b, data)
            position = nearest_position_on_polyline(coords, lon, lat)
            if position is None:
                continue
            if position["distance_m"] <= max(EXACT_POINT_MAX_TRAIL_OFFSET_M, edge_distance_m + 0.5):
                candidates.append((a, b, candidate_key, data, coords, position))

    # At minimum, make sure the exact edge returned by nearest_edges is split.
    if not candidates:
        candidates.append((u, v, key, selected_data, selected_coords, nearest))

    for a, b, candidate_key, data, coords, position in candidates:
        if H.has_edge(a, b, candidate_key):
            H.remove_edge(a, b, candidate_key)

        left, right = split_polyline_at_position(
            coords,
            position["segment_index"],
            position["t"],
            split_lon,
            split_lat,
        )

        left_length = polyline_distance_meters(left)
        right_length = polyline_distance_meters(right)

        # Ignore pathological near-zero pieces. This should not happen for the
        # GPX start used here, which is well inside the OSM edge.
        if left_length > 0.25:
            H.add_edge(
                a,
                virtual_node,
                key=candidate_key,
                **edge_attributes_for_split_part(data, left),
            )
            split_count += 1

        if right_length > 0.25:
            H.add_edge(
                virtual_node,
                b,
                key=candidate_key,
                **edge_attributes_for_split_part(data, right),
            )
            split_count += 1

    if H.degree(virtual_node) == 0:
        H.remove_node(virtual_node)
        fallback = int(ox.distance.nearest_nodes(G, X=lon, Y=lat))
        fallback_lat = float(G.nodes[fallback]["y"])
        fallback_lon = float(G.nodes[fallback]["x"])
        return G, fallback, {
            "exact_inserted": False,
            "trail_offset_m": round(edge_distance_m, 2),
            "routing_lat": fallback_lat,
            "routing_lon": fallback_lon,
            "routing_offset_m": round(haversine_meters(lat, lon, fallback_lat, fallback_lon), 2),
            "source_edge": [int(u), int(v), str(key)],
        }

    return H, virtual_node, {
        "exact_inserted": True,
        "trail_offset_m": round(edge_distance_m, 2),
        "routing_lat": float(split_lat),
        "routing_lon": float(split_lon),
        "routing_offset_m": 0.0,
        "source_edge": [int(u), int(v), str(key)],
        "split_directed_pieces": split_count,
    }


# ============================================================
# DEBUG / ALLOWED TRAIL GEOMETRY
# ============================================================

def graph_debug_segments(G):
    segments = []
    seen = set()

    for u, v, key, data in G.edges(keys=True, data=True):
        physical_key = (
            min(int(u), int(v)),
            max(int(u), int(v)),
            round(float(data.get("length", 0) or 0), 1),
        )

        if physical_key in seen:
            continue

        seen.add(physical_key)

        coords = oriented_edge_coords(G, u, v, data)
        if len(coords) < 2:
            continue

        segments.append(
            [[float(lat), float(lon)] for lon, lat in coords]
        )

    return segments


# ============================================================
# TRAIL FILTER
# ============================================================

def edge_is_allowed_trail(data):
    highways = normalize_tag_values(data.get("highway"))
    surfaces = normalize_tag_values(data.get("surface"))
    access = normalize_tag_values(data.get("access"))
    foot = normalize_tag_values(data.get("foot"))
    area = normalize_tag_values(data.get("area"))
    indoor = normalize_tag_values(data.get("indoor"))

    if not highways.intersection({"path", "track", "steps"}):
        return False

    hard_surfaces = {
        "asphalt",
        "concrete",
        "concrete:lanes",
        "concrete:plates",
        "paving_stones",
        "sett",
        "cobblestone",
    }

    if surfaces.intersection(hard_surfaces):
        return False

    if "yes" in area:
        return False

    if "yes" in indoor:
        return False

    if access.intersection({"no", "private"}):
        if not foot.intersection({"yes", "designated", "permissive"}):
            return False

    return True


# ============================================================
# DENSIFY POLYLINE
# ============================================================

def densify_polyline(coords, spacing_m=ELEVATION_SAMPLE_SPACING_M):
    if not coords:
        return []

    if len(coords) == 1:
        return coords[:]

    dense = [coords[0]]

    for i in range(len(coords) - 1):
        lon1, lat1 = coords[i]
        lon2, lat2 = coords[i + 1]

        segment_distance = haversine_meters(
            lat1,
            lon1,
            lat2,
            lon2,
        )

        if segment_distance <= 0:
            continue

        steps = max(1, int(math.ceil(segment_distance / spacing_m)))

        for step in range(1, steps + 1):
            fraction = step / steps

            lon = lon1 + (lon2 - lon1) * fraction
            lat = lat1 + (lat2 - lat1) * fraction

            dense.append((float(lon), float(lat)))

    return dense


# ============================================================
# DEM SMOOTHING / ASCENT
# ============================================================

def smooth_elevations(values, radius=2):
    if len(values) < 3:
        return [float(v) for v in values]

    result = []

    for i in range(len(values)):
        start = max(0, i - radius)
        end = min(len(values), i + radius + 1)
        window = values[start:end]
        result.append(sum(float(v) for v in window) / len(window))

    return result


def calculate_ascent_descent(elevations):
    ascent = 0.0
    descent = 0.0

    for i in range(len(elevations) - 1):
        delta = elevations[i + 1] - elevations[i]

        if delta > 0:
            ascent += delta
        elif delta < 0:
            descent += -delta

    return ascent, descent


def dem_value_to_meters(src, value):
    unit = ""

    try:
        if src.units:
            unit = src.units[0] or ""
    except Exception:
        unit = ""

    unit = str(unit).lower()

    if "foot" in unit or "feet" in unit or unit == "ft":
        return float(value) / FEET_PER_METER

    return float(value)


# ============================================================
# LOCAL DEM SAMPLING
# ============================================================

def sample_dem_points(points):
    """
    Return DEM elevation values for lon/lat points.

    Values are cached by rounded coordinate. This preserves the exact same
    DEM source and precision while avoiding repeated raster reads during
    finalist scoring.
    """
    if not os.path.exists(DEM_PATH):
        raise HTTPException(
            status_code=500,
            detail=f"DEM file not found: {DEM_PATH}",
        )

    unique = {}
    for lon, lat in points:
        key = (
            round(float(lat), 7),
            round(float(lon), 7),
        )
        unique[key] = (float(lon), float(lat))

    lookup = {}
    missing = {}

    for key, point in unique.items():
        if key in DEM_POINT_CACHE:
            lookup[key] = float(DEM_POINT_CACHE[key])
        else:
            missing[key] = point

    if missing:
        keys = list(missing.keys())
        lons = [missing[key][0] for key in keys]
        lats = [missing[key][1] for key in keys]

        with rasterio.open(DEM_PATH) as src:
            if src.crs is None:
                raise HTTPException(
                    status_code=500,
                    detail="DEM has no CRS information.",
                )

            xs, ys = rio_transform(
                "EPSG:4326",
                src.crs,
                lons,
                lats,
            )

            outside = []
            for key, x, y in zip(keys, xs, ys):
                if not (
                    src.bounds.left <= x <= src.bounds.right
                    and src.bounds.bottom <= y <= src.bounds.top
                ):
                    outside.append(key)

            if outside:
                first_lat, first_lon = outside[0]
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "DEM does not cover the entire requested area. "
                        f"First uncovered point: {first_lat}, {first_lon}"
                    ),
                )

            samples = list(
                src.sample(
                    zip(xs, ys),
                    indexes=1,
                    masked=True,
                )
            )

            for key, sample in zip(keys, samples):
                value = sample[0]

                if np.ma.is_masked(value):
                    raise HTTPException(
                        status_code=400,
                        detail="DEM contains NoData.",
                    )

                value = float(value)

                if not math.isfinite(value):
                    raise HTTPException(
                        status_code=400,
                        detail="DEM returned invalid elevation.",
                    )

                if (
                    src.nodata is not None
                    and math.isclose(
                        value,
                        float(src.nodata),
                        rel_tol=0.0,
                        abs_tol=1e-6,
                    )
                ):
                    raise HTTPException(
                        status_code=400,
                        detail="DEM contains NoData.",
                    )

                meters = dem_value_to_meters(src, value)
                lookup[key] = meters

                # Keep the cache bounded. Clearing is intentionally simple;
                # the graph cache still prevents expensive network rebuilds.
                if len(DEM_POINT_CACHE) >= MAX_DEM_POINT_CACHE:
                    DEM_POINT_CACHE.clear()

                DEM_POINT_CACHE[key] = meters

    return lookup

def elevations_for_coords(coords):
    lookup = sample_dem_points(coords)

    values = []
    for lon, lat in coords:
        key = (
            round(float(lat), 7),
            round(float(lon), 7),
        )
        values.append(float(lookup[key]))

    return values


# ============================================================
# ADD ELEVATION TO GRAPH EDGES
# ============================================================

def add_local_dem_edge_elevations(G):
    edge_samples = {}
    all_points = []

    for u, v, key, data in G.edges(keys=True, data=True):
        coords = oriented_edge_coords(G, u, v, data)
        samples = densify_polyline(coords, ELEVATION_SAMPLE_SPACING_M)

        if len(samples) < 2:
            samples = [
                (
                    float(G.nodes[u]["x"]),
                    float(G.nodes[u]["y"]),
                ),
                (
                    float(G.nodes[v]["x"]),
                    float(G.nodes[v]["y"]),
                ),
            ]

        edge_samples[(u, v, key)] = samples
        all_points.extend(samples)

    elevation_lookup = sample_dem_points(all_points)

    node_points = [
        (
            float(G.nodes[node]["x"]),
            float(G.nodes[node]["y"]),
        )
        for node in G.nodes
    ]

    node_lookup = sample_dem_points(node_points)

    for node in G.nodes:
        lat = float(G.nodes[node]["y"])
        lon = float(G.nodes[node]["x"])

        node_key = (
            round(lat, 7),
            round(lon, 7),
        )

        G.nodes[node]["elevation"] = float(node_lookup[node_key])

    for (u, v, key), samples in edge_samples.items():
        elevations = []

        for lon, lat in samples:
            sample_key = (
                round(float(lat), 7),
                round(float(lon), 7),
            )
            elevations.append(float(elevation_lookup[sample_key]))

        elevations = smooth_elevations(elevations, radius=ELEVATION_SMOOTHING_RADIUS)
        ascent, descent = calculate_ascent_descent(elevations)

        G[u][v][key]["ascent_m"] = float(ascent)
        G[u][v][key]["descent_m"] = float(descent)
        G[u][v][key]["elevation_sample_count"] = len(samples)

    return G, len(elevation_lookup)


# ============================================================
# MASTER TIFF TRAIL GRAPH + LOCAL GRAPH CACHE
# ============================================================

def get_dem_bounds_wgs84():
    """Return the TIFF footprint as (left, bottom, right, top) in EPSG:4326."""
    global DEM_BOUNDS_WGS84_CACHE

    if DEM_BOUNDS_WGS84_CACHE is not None:
        return DEM_BOUNDS_WGS84_CACHE

    if not os.path.exists(DEM_PATH):
        raise HTTPException(
            status_code=500,
            detail=f"DEM file not found: {DEM_PATH}",
        )

    with rasterio.open(DEM_PATH) as src:
        if src.crs is None:
            raise HTTPException(
                status_code=500,
                detail="DEM has no CRS information.",
            )

        left, bottom, right, top = rio_transform_bounds(
            src.crs,
            "EPSG:4326",
            src.bounds.left,
            src.bounds.bottom,
            src.bounds.right,
            src.bounds.top,
            densify_pts=21,
        )

    DEM_BOUNDS_WGS84_CACHE = (
        float(left),
        float(bottom),
        float(right),
        float(top),
    )
    return DEM_BOUNDS_WGS84_CACHE


def get_dem_signature():
    """Stable signature used to reject a stale saved master graph."""
    bounds = get_dem_bounds_wgs84()

    with rasterio.open(DEM_PATH) as src:
        width = int(src.width)
        height = int(src.height)

    return (
        f"{os.path.basename(DEM_PATH)}|"
        f"{bounds[0]:.8f},{bounds[1]:.8f},"
        f"{bounds[2]:.8f},{bounds[3]:.8f}|"
        f"{width}x{height}"
    )


def point_inside_dem(lat, lon):
    left, bottom, right, top = get_dem_bounds_wgs84()
    lat = float(lat)
    lon = float(lon)
    return left <= lon <= right and bottom <= lat <= top


def edge_fully_inside_dem(G, u, v, data):
    """
    Keep only trails whose complete stored geometry lies inside the TIFF.
    This deliberately drops boundary-crossing pieces so route elevation can
    never wander into NoData/outside-raster territory.
    """
    left, bottom, right, top = get_dem_bounds_wgs84()
    coords = oriented_edge_coords(G, u, v, data)

    if not coords:
        return False

    for lon, lat in coords:
        if not (left <= float(lon) <= right and bottom <= float(lat) <= top):
            return False

    return True


def configure_osmnx_trail_tags():
    extra_tags = [
        "surface",
        "footway",
        "foot",
        "access",
        "area",
        "indoor",
        "tracktype",
        "sac_scale",
        "trail_visibility",
    ]

    useful_tags = list(ox.settings.useful_tags_way)

    for tag in extra_tags:
        if tag not in useful_tags:
            useful_tags.append(tag)

    ox.settings.useful_tags_way = useful_tags


def master_graph_metadata(G, loaded_from_disk=False):
    physical = set()

    for u, v, key, data in G.edges(keys=True, data=True):
        physical.add(
            (
                min(int(u), int(v)),
                max(int(u), int(v)),
                round(float(data.get("length", 0) or 0), 1),
            )
        )

    return {
        "nodes": int(G.number_of_nodes()),
        "edges": int(G.number_of_edges()),
        "physical_segments": int(len(physical)),
        "filtered_edges_removed": int(
            float(G.graph.get("master_filtered_edges_removed", 0) or 0)
        ),
        "loaded_from_disk": bool(loaded_from_disk),
        "bbox": get_dem_bounds_wgs84(),
        "saved_graph": os.path.basename(MASTER_GRAPH_PATH),
    }


def try_load_saved_master_graph():
    if not os.path.exists(MASTER_GRAPH_PATH):
        return None

    try:
        G = ox.io.load_graphml(filepath=MASTER_GRAPH_PATH)
        saved_signature = str(G.graph.get("dem_signature", ""))

        if saved_signature != get_dem_signature():
            return None

        if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
            return None

        return G
    except Exception:
        # A bad/stale cache must never prevent the app from rebuilding.
        return None


def save_master_graph(G):
    try:
        ox.io.save_graphml(G, filepath=MASTER_GRAPH_PATH)
        return True
    except Exception:
        # Runtime cache still works even if the deployment filesystem cannot
        # persist the graph file.
        return False


def build_master_trail_graph():
    """
    Download and filter every allowed trail inside the complete TIFF footprint.

    This is done once per service lifetime (or loaded from the saved GraphML).
    Elevation is intentionally NOT precomputed for the whole master network:
    only the small local subgraph used by a route request receives DEM edge
    data, preserving v6 routing behavior without a huge startup cost.
    """
    configure_osmnx_trail_tags()

    bbox = get_dem_bounds_wgs84()
    trail_filter = '["highway"~"path|track|steps"]'

    try:
        G = ox.graph.graph_from_bbox(
            bbox,
            network_type="walk",
            custom_filter=trail_filter,
            simplify=True,
            retain_all=True,
            truncate_by_edge=False,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not download the master TIFF trail network: {exc}",
        )

    original_edges = G.number_of_edges()
    remove_edges = []

    for u, v, key, data in G.edges(keys=True, data=True):
        if not edge_is_allowed_trail(data):
            remove_edges.append((u, v, key))
            continue

        if not edge_fully_inside_dem(G, u, v, data):
            remove_edges.append((u, v, key))

    G.remove_edges_from(remove_edges)
    G.remove_nodes_from(list(nx.isolates(G)))

    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        raise HTTPException(
            status_code=400,
            detail="No usable trail network was found inside the TIFF footprint.",
        )

    G.graph["dem_signature"] = get_dem_signature()
    G.graph["master_filtered_edges_removed"] = int(
        original_edges - G.number_of_edges()
    )
    G.graph["master_tiff_name"] = os.path.basename(DEM_PATH)
    G.graph["master_network_version"] = APP_VERSION

    save_master_graph(G)
    return G


def get_master_trail_graph():
    """Return the one in-memory trail graph covering the entire TIFF."""
    global MASTER_GRAPH, MASTER_GRAPH_INFO

    if MASTER_GRAPH is not None:
        return MASTER_GRAPH, MASTER_GRAPH_INFO

    with MASTER_GRAPH_LOCK:
        if MASTER_GRAPH is not None:
            return MASTER_GRAPH, MASTER_GRAPH_INFO

        G = try_load_saved_master_graph()
        loaded_from_disk = G is not None

        if G is None:
            G = build_master_trail_graph()

        MASTER_GRAPH = G
        MASTER_GRAPH_INFO = master_graph_metadata(
            G,
            loaded_from_disk=loaded_from_disk,
        )

    return MASTER_GRAPH, MASTER_GRAPH_INFO


def extract_local_master_subgraph(master_G, lat, lon, radius_meters):
    """
    Extract only the nearby portion of the already-loaded TIFF-wide graph.
    The route algorithms therefore see a graph comparable in size to v6.
    """
    if not point_inside_dem(lat, lon):
        left, bottom, right, top = get_dem_bounds_wgs84()
        raise HTTPException(
            status_code=400,
            detail=(
                "Start coordinate is outside the elevation TIFF coverage. "
                f"TIFF bounds are west={left:.6f}, east={right:.6f}, "
                f"south={bottom:.6f}, north={top:.6f}."
            ),
        )

    local_bbox = ox.utils_geo.bbox_from_point(
        (float(lat), float(lon)),
        float(radius_meters),
    )

    dem_left, dem_bottom, dem_right, dem_top = get_dem_bounds_wgs84()
    left = max(float(local_bbox[0]), dem_left)
    bottom = max(float(local_bbox[1]), dem_bottom)
    right = min(float(local_bbox[2]), dem_right)
    top = min(float(local_bbox[3]), dem_top)

    if left >= right or bottom >= top:
        raise HTTPException(
            status_code=400,
            detail="No TIFF-covered search area exists around this start coordinate.",
        )

    try:
        local = ox.truncate.truncate_graph_bbox(
            master_G,
            (left, bottom, right, top),
            truncate_by_edge=True,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not extract trails around this start coordinate: {exc}",
        )

    if local.number_of_nodes() == 0 or local.number_of_edges() == 0:
        raise HTTPException(
            status_code=400,
            detail="No allowed trails were found near this start coordinate.",
        )

    try:
        nearest = ox.distance.nearest_nodes(
            local,
            X=float(lon),
            Y=float(lat),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not locate a nearby trail node: {exc}",
        )

    component = nx.node_connected_component(
        local.to_undirected(as_view=True),
        nearest,
    )

    G = local.subgraph(component).copy()

    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        raise HTTPException(
            status_code=400,
            detail="The nearby trail network is not connected enough to route from this start.",
        )

    return G


def download_trail_graph(lat, lon, radius_meters):
    """
    v7 compatibility wrapper.

    Unlike v6, this never downloads OSM around each individual start. It loads
    one TIFF-wide master network once, then extracts/caches a local graph and
    adds the same DEM edge elevation data used by the existing route engines.
    """
    cache_key = (
        round(float(lat), 5),
        round(float(lon), 5),
        int(radius_meters),
        ELEVATION_SAMPLE_SPACING_M,
        os.path.basename(DEM_PATH),
    )

    if cache_key in GRAPH_CACHE:
        cached = GRAPH_CACHE[cache_key]
        return (
            cached["graph"],
            cached["filtered_edges_removed"],
            True,
            cached["unique_elevation_samples"],
        )

    master_G, master_info = get_master_trail_graph()

    G = extract_local_master_subgraph(
        master_G,
        lat,
        lon,
        radius_meters,
    )

    G, unique_elevation_samples = add_local_dem_edge_elevations(G)

    if len(GRAPH_CACHE) >= MAX_CACHED_GRAPHS:
        oldest = next(iter(GRAPH_CACHE))
        GRAPH_CACHE.pop(oldest)

    GRAPH_CACHE[cache_key] = {
        "graph": G,
        "filtered_edges_removed": master_info["filtered_edges_removed"],
        "unique_elevation_samples": unique_elevation_samples,
    }

    return (
        G,
        master_info["filtered_edges_removed"],
        False,
        unique_elevation_samples,
    )


# ============================================================
# SIMPLE ROUTING GRAPH
# ============================================================

def make_simple_routing_graph(G):
    S = nx.DiGraph()
    S.add_nodes_from(G.nodes(data=True))

    for u, v, data in G.edges(data=True):
        length = float(data.get("length", 0) or 0)
        ascent = float(data.get("ascent_m", 0) or 0)

        if length <= 0:
            continue

        if (
            not S.has_edge(u, v)
            or length < float(S[u][v].get("length", float("inf")))
        ):
            S.add_edge(
                u,
                v,
                length=length,
                ascent_m=ascent,
            )

    return S


# ============================================================
# REPETITION METRICS
# ============================================================

def repeated_edge_stats(G, route_nodes):
    counts = {}
    lengths = {}

    for i in range(len(route_nodes) - 1):
        u = route_nodes[i]
        v = route_nodes[i + 1]

        edge_key = undirected_edge_key(u, v)
        edge = get_shortest_edge(G, u, v)

        if edge:
            length = float(edge.get("length", 0) or 0)
        else:
            length = 0.0

        counts[edge_key] = counts.get(edge_key, 0) + 1
        lengths[edge_key] = length

    repeated_edges = 0
    repeated_distance = 0.0

    for edge_key, count in counts.items():
        if count > 1:
            repeated_edges += count - 1
            repeated_distance += lengths.get(edge_key, 0) * (count - 1)

    return repeated_edges, repeated_distance


def repeated_node_occurrences(route_nodes):
    counts = {}

    for node in route_nodes[1:-1]:
        counts[node] = counts.get(node, 0) + 1

    return sum(
        count - 1
        for count in counts.values()
        if count > 1
    )


def count_immediate_reversals(route_nodes):
    count = 0

    for i in range(len(route_nodes) - 2):
        if route_nodes[i] == route_nodes[i + 2]:
            count += 1

    return count


# ============================================================
# ROUTE SCORE / CONTINUOUS ELEVATION
# ============================================================

def route_geometry_metrics(coords):
    """Authoritative distance/gain from one continuous route geometry."""
    if len(coords) < 2:
        return {
            "distance_meters": 0.0,
            "gain_meters": 0.0,
            "descent_meters": 0.0,
            "dem_sample_points": 0,
        }

    lonlat = [
        (float(point["lon"]), float(point["lat"]))
        for point in coords
    ]

    distance_m = 0.0
    for i in range(len(lonlat) - 1):
        lon1, lat1 = lonlat[i]
        lon2, lat2 = lonlat[i + 1]
        distance_m += haversine_meters(lat1, lon1, lat2, lon2)

    dense = densify_polyline(lonlat, ELEVATION_SAMPLE_SPACING_M)
    raw_elevations = elevations_for_coords(dense)
    smoothed = smooth_elevations(
        raw_elevations,
        radius=ELEVATION_SMOOTHING_RADIUS,
    )
    ascent_m, descent_m = calculate_ascent_descent(smoothed)

    return {
        "distance_meters": float(distance_m),
        "gain_meters": float(ascent_m),
        "descent_meters": float(descent_m),
        "dem_sample_points": len(dense),
    }


def score_route_coordinates(
    G,
    coords,
    route_nodes,
    target_distance_meters,
    target_gain_meters,
    partial_added_distance_m=0.0,
):
    geometry = route_geometry_metrics(coords)
    total_distance = geometry["distance_meters"]
    actual_gain = geometry["gain_meters"]

    if total_distance <= 0:
        return float("inf"), {}

    distance_error = abs(total_distance - target_distance_meters)
    distance_ratio = distance_error / max(target_distance_meters, 1.0)

    gain_error = abs(actual_gain - target_gain_meters)
    if target_gain_meters > 0:
        gain_ratio = gain_error / target_gain_meters
    else:
        gain_ratio = actual_gain / 30.48

    repeated_edges, repeated_distance = repeated_edge_stats(G, route_nodes)
    # A partial out-and-back repeats the same physical trail by definition.
    repeated_distance += max(0.0, float(partial_added_distance_m) / 2.0)
    repeat_ratio = repeated_distance / total_distance
    repeated_nodes = repeated_node_occurrences(route_nodes)
    immediate_reversals = count_immediate_reversals(route_nodes)

    if target_distance_meters < 4 * METERS_PER_MILE:
        repeat_weight = 50.0
        node_weight = 6.0
    else:
        repeat_weight = 300.0
        node_weight = 25.0

    score = (
        distance_ratio * 190.0
        + gain_ratio * 240.0
        + repeat_ratio * repeat_weight
        + repeated_nodes * node_weight
        + immediate_reversals * 12.0
    )

    return (
        score,
        {
            "total_distance_meters": total_distance,
            "actual_gain_meters": actual_gain,
            "actual_descent_meters": geometry["descent_meters"],
            "distance_error_meters": distance_error,
            "gain_error_meters": gain_error,
            "repeated_edges": repeated_edges,
            "repeated_distance_meters": repeated_distance,
            "repeat_ratio": repeat_ratio,
            "repeated_nodes": repeated_nodes,
            "immediate_reversals": immediate_reversals,
            "score": score,
            "route_coordinates": coords,
            "route_elevation_sample_count": geometry["dem_sample_points"],
            "partial_edge_used": partial_added_distance_m > 0,
            "partial_added_distance_meters": float(partial_added_distance_m),
        },
    )


def route_score(G, route_nodes, target_distance_meters, target_gain_meters):
    coords = route_coordinates(G, route_nodes)
    return score_route_coordinates(
        G,
        coords,
        route_nodes,
        target_distance_meters,
        target_gain_meters,
    )


# ============================================================
# PARTIAL-EDGE OUT-AND-BACK TUNING
# ============================================================

def polyline_prefix_by_distance(coords, requested_distance_m):
    """Return the first requested_distance_m of a lon/lat polyline."""
    if not coords:
        return []
    if requested_distance_m <= 0:
        return [coords[0]]

    result = [coords[0]]
    remaining = float(requested_distance_m)

    for i in range(len(coords) - 1):
        lon1, lat1 = coords[i]
        lon2, lat2 = coords[i + 1]
        segment_m = haversine_meters(lat1, lon1, lat2, lon2)

        if segment_m <= 0:
            continue

        if remaining >= segment_m:
            result.append((float(lon2), float(lat2)))
            remaining -= segment_m
            if remaining <= 0.01:
                break
            continue

        fraction = remaining / segment_m
        lon = lon1 + (lon2 - lon1) * fraction
        lat = lat1 + (lat2 - lat1) * fraction
        result.append((float(lon), float(lat)))
        remaining = 0.0
        break

    return result


def build_route_coordinates_with_excursion(
    G,
    route_nodes,
    insert_after_index,
    excursion_neighbor,
    outward_distance_m,
):
    """Build normal route plus a partial edge out-and-back at one route node."""
    output = []

    def append_lonlat(points):
        for lon, lat in points:
            point = {"lat": float(lat), "lon": float(lon)}
            if output:
                prev = output[-1]
                if (
                    abs(prev["lat"] - point["lat"]) < 1e-8
                    and abs(prev["lon"] - point["lon"]) < 1e-8
                ):
                    continue
            output.append(point)

    for i in range(len(route_nodes) - 1):
        u = route_nodes[i]
        v = route_nodes[i + 1]

        if i == insert_after_index:
            excursion_edge = get_shortest_edge(G, u, excursion_neighbor)
            if excursion_edge is not None:
                excursion_coords = oriented_edge_coords(
                    G,
                    u,
                    excursion_neighbor,
                    excursion_edge,
                )
                prefix = polyline_prefix_by_distance(
                    excursion_coords,
                    outward_distance_m,
                )
                if len(prefix) >= 2:
                    append_lonlat(prefix)
                    append_lonlat(list(reversed(prefix)))

        edge = get_shortest_edge(G, u, v)
        if edge is not None:
            append_lonlat(oriented_edge_coords(G, u, v, edge))

    return output


def partial_edge_tuning_candidates(
    G,
    route_nodes,
    base_metrics,
    target_distance_meters,
    target_gain_meters,
    max_edges_to_try=4,
):
    """
    Add one partial out-and-back to a promising closed loop.
    We choose the flattest candidate outgoing edges and solve the outward
    distance from the route's current distance deficit.
    """
    base_distance = float(base_metrics["total_distance_meters"])
    deficit = target_distance_meters - base_distance

    # Partial tuning only adds distance. Keep it for reasonably close bases.
    if deficit < 20.0 or deficit > PARTIAL_TUNING_MAX_DEFICIT_M:
        return []

    outward_needed = deficit / 2.0
    S = make_simple_routing_graph(G)
    options = []

    for index, u in enumerate(route_nodes[:-1]):
        next_node = route_nodes[index + 1]
        prev_node = route_nodes[index - 1] if index > 0 else None

        for neighbor in S.successors(u):
            data = S[u][neighbor]
            length = float(data.get("length", 0) or 0)
            ascent = float(data.get("ascent_m", 0) or 0)

            if length < max(25.0, outward_needed * 0.75):
                continue

            # Prefer side trails, but allow the main route edge if necessary.
            route_edge_penalty = 0.25 if neighbor in {next_node, prev_node} else 0.0
            gain_density = ascent / max(length, 1.0)
            options.append(
                (
                    gain_density + route_edge_penalty,
                    index,
                    neighbor,
                    length,
                )
            )

    options.sort(key=lambda item: item[0])
    options = options[:max_edges_to_try]

    results = []
    for _, index, neighbor, edge_length in options:
        # Try exact fill plus +/-25 m outward so final DEM distance has room.
        trial_outward = {
            max(10.0, outward_needed - 25.0),
            max(10.0, outward_needed),
            min(edge_length, outward_needed + 25.0),
        }

        for outward in sorted(trial_outward):
            if outward <= 0 or outward > edge_length:
                continue

            coords = build_route_coordinates_with_excursion(
                G,
                route_nodes,
                index,
                neighbor,
                outward,
            )
            if len(coords) < 2:
                continue

            partial_added = 2.0 * outward
            score, metrics = score_route_coordinates(
                G,
                coords,
                route_nodes,
                target_distance_meters,
                target_gain_meters,
                partial_added_distance_m=partial_added,
            )
            if metrics:
                metrics["partial_edge_from_node"] = int(route_nodes[index])
                metrics["partial_edge_toward_node"] = int(neighbor)
                metrics["partial_outward_distance_meters"] = float(outward)
                results.append((score, route_nodes, metrics))

    return results


# ============================================================
# TRUE CLOSED-LOOP SHORT ROUTE BEAM SEARCH
# ============================================================

def beam_search_short_loop(
    G,
    start_node,
    target_distance_meters,
    target_gain_meters,
    limits,
    profile,
):
    """Budgeted multi-objective closed-loop search with partial-edge tuning."""
    S = make_simple_routing_graph(G)
    reverse_S = S.reverse(copy=False)

    try:
        return_distance = nx.single_source_dijkstra_path_length(
            reverse_S,
            start_node,
            weight="length",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not calculate return-distance bounds: {exc}",
        )

    allowed_distance_error_m = limits["distance_error_limit_miles"] * METERS_PER_MILE
    max_acceptable_distance = target_distance_meters + allowed_distance_error_m

    beam_width = int(profile.get("beam_width", 500))
    max_steps = int(profile.get("beam_max_steps", 80))
    max_seconds = float(profile.get("max_search_seconds", 20.0))
    max_states = int(profile.get("max_expanded_states", 120000))
    max_closed = int(profile.get("max_closed_candidates", 150))
    partial_base_count = int(profile.get("partial_tuning_base_candidates", 24))

    target_gain_density = target_gain_meters / max(target_distance_meters, 1.0)
    started = time.perf_counter()
    states_expanded = 0
    budget_reached = False
    last_depth = 0

    beam = [{
        "route": (start_node,),
        "node": start_node,
        "distance": 0.0,
        "gain": 0.0,
        "used_edges": frozenset(),
        "repeat_distance": 0.0,
        "reversals": 0,
    }]

    closed_candidates = []
    closed_seen = set()

    def budget_hit():
        return (
            states_expanded >= max_states
            or (time.perf_counter() - started) >= max_seconds
        )

    def state_priorities(state, min_final_distance):
        distance_error = abs(min_final_distance - target_distance_meters) / max(
            target_distance_meters, 1.0
        )
        gain_density = state["gain"] / max(state["distance"], 1.0)
        density_error = abs(gain_density - target_gain_density) / max(
            target_gain_density, 0.003
        )
        repeat_ratio = state["repeat_distance"] / max(state["distance"], 1.0)
        reversal_penalty = state["reversals"] * 0.02

        return {
            "balanced": distance_error * 3.0 + density_error * 0.85 + repeat_ratio * 0.25 + reversal_penalty,
            "gain": density_error * 2.5 + distance_error * 1.0 + repeat_ratio * 0.20 + reversal_penalty,
            "distance": distance_error * 5.0 + density_error * 0.15 + repeat_ratio * 0.15 + reversal_penalty,
            "flat": gain_density * 8.0 + distance_error * 2.0 + repeat_ratio * 0.15 + reversal_penalty,
        }

    for depth in range(max_steps):
        last_depth = depth + 1
        if budget_hit():
            budget_reached = True
            break

        expanded = []

        for state in beam:
            if budget_hit():
                budget_reached = True
                break

            current = state["node"]
            for neighbor in S.successors(current):
                states_expanded += 1
                if budget_hit():
                    budget_reached = True
                    break

                edge = S[current][neighbor]
                edge_length = float(edge.get("length", 0) or 0)
                edge_gain = float(edge.get("ascent_m", 0) or 0)
                if edge_length <= 0:
                    continue

                new_distance = state["distance"] + edge_length
                if new_distance > max_acceptable_distance:
                    continue

                edge_key = undirected_edge_key(current, neighbor)
                already_used = edge_key in state["used_edges"]
                repeat_distance = state["repeat_distance"] + (
                    edge_length if already_used else 0.0
                )
                used_edges = set(state["used_edges"])
                used_edges.add(edge_key)

                route = state["route"]
                immediate_reversal = int(len(route) >= 2 and neighbor == route[-2])
                reversals = state["reversals"] + immediate_reversal
                new_route = route + (neighbor,)
                new_gain = state["gain"] + edge_gain

                if neighbor == start_node:
                    # A partial out-and-back can add at most 0.75 mi. Therefore a
                    # closed loop shorter than target - 0.75 mi is NOT a useful
                    # final/tuning base yet. Keep searching through the trailhead
                    # instead of terminating the state there.
                    tunable_min_distance = max(
                        0.0,
                        target_distance_meters - PARTIAL_TUNING_MAX_DEFICIT_M,
                    )

                    if new_distance >= tunable_min_distance:
                        route_key = tuple(new_route)
                        if route_key not in closed_seen:
                            closed_seen.add(route_key)

                            distance_ratio = abs(new_distance - target_distance_meters) / max(
                                target_distance_meters, 1.0
                            )
                            gain_ratio = abs(new_gain - target_gain_meters) / max(
                                target_gain_meters, 30.48
                            )
                            repeat_ratio = repeat_distance / max(new_distance, 1.0)
                            gain_density = new_gain / max(new_distance, 1.0)

                            closed_candidates.append({
                                "route": list(new_route),
                                "distance": new_distance,
                                "gain": new_gain,
                                "distance_ratio": distance_ratio,
                                "gain_ratio": gain_ratio,
                                "repeat_ratio": repeat_ratio,
                                "gain_density": gain_density,
                                "cheap_balanced": distance_ratio * 3.0 + gain_ratio * 1.2 + repeat_ratio * 0.25,
                            })

                            # Keep candidate storage bounded but diverse.
                            if len(closed_candidates) > max_closed * 6:
                                closed_candidates = diversify_closed_candidates(
                                    closed_candidates,
                                    target_distance_meters,
                                    target_gain_meters,
                                    max_closed * 3,
                                )

                    # Do not automatically terminate just because we touched the
                    # start. If we are still below target, the route may leave the
                    # trailhead again and combine another loop/branch before the
                    # final closure. This is especially important for short routes.
                    continue_through_start = bool(
                        profile.get("continue_through_start_below_target", True)
                    )

                    if (
                        not continue_through_start
                        or new_distance >= target_distance_meters
                    ):
                        continue

                    new_state = {
                        "route": new_route,
                        "node": start_node,
                        "distance": new_distance,
                        "gain": new_gain,
                        "used_edges": frozenset(used_edges),
                        "repeat_distance": repeat_distance,
                        "reversals": reversals,
                    }
                    priorities = state_priorities(new_state, new_distance)
                    expanded.append((priorities, new_state))
                    continue

                if neighbor not in return_distance:
                    continue

                min_final_distance = new_distance + float(return_distance[neighbor])
                if min_final_distance > max_acceptable_distance:
                    continue

                new_state = {
                    "route": new_route,
                    "node": neighbor,
                    "distance": new_distance,
                    "gain": new_gain,
                    "used_edges": frozenset(used_edges),
                    "repeat_distance": repeat_distance,
                    "reversals": reversals,
                }
                priorities = state_priorities(new_state, min_final_distance)
                expanded.append((priorities, new_state))

            if budget_reached:
                break

        if budget_reached:
            break
        if not expanded:
            break

        # Deduplicate similar states first.
        unique = []
        seen_buckets = set()
        expanded.sort(key=lambda item: item[0]["balanced"])
        for priorities, state in expanded:
            route = state["route"]
            previous = route[-2] if len(route) >= 2 else None
            bucket = (
                state["node"],
                previous,
                int(state["distance"] / 30.0),
                int(state["gain"] / 6.0),
                int(state["repeat_distance"] / 30.0),
            )
            if bucket in seen_buckets:
                continue
            seen_buckets.add(bucket)
            unique.append((priorities, state))

        # Reserve beam capacity for different objectives.
        quotas = {
            "balanced": int(beam_width * 0.40),
            "gain": int(beam_width * 0.30),
            "distance": int(beam_width * 0.20),
            "flat": beam_width - int(beam_width * 0.40) - int(beam_width * 0.30) - int(beam_width * 0.20),
        }

        selected = []
        selected_routes = set()
        for objective, quota in quotas.items():
            ranked = sorted(unique, key=lambda item: item[0][objective])
            count = 0
            for _, state in ranked:
                key = state["route"]
                if key in selected_routes:
                    continue
                selected_routes.add(key)
                selected.append(state)
                count += 1
                if count >= quota:
                    break

        if len(selected) < beam_width:
            for _, state in sorted(unique, key=lambda item: item[0]["balanced"]):
                if state["route"] in selected_routes:
                    continue
                selected_routes.add(state["route"])
                selected.append(state)
                if len(selected) >= beam_width:
                    break

        beam = selected

    if not closed_candidates:
        budget_text = " Search budget was reached." if budget_reached else ""
        raise HTTPException(
            status_code=400,
            detail=(
                "No closed loop was found within the search budget."
                + budget_text
            ),
        )

    closed_candidates = diversify_closed_candidates(
        closed_candidates,
        target_distance_meters,
        target_gain_meters,
        max_closed,
    )

    accurately_scored = []
    for candidate in closed_candidates:
        score, metrics = route_score(
            G,
            candidate["route"],
            target_distance_meters,
            target_gain_meters,
        )
        if metrics:
            accurately_scored.append((score, candidate["route"], metrics))

    # Try partial-edge tuning on promising under-distance bases.
    partial_bases = sorted(
        accurately_scored,
        key=lambda item: (
            abs(item[2]["actual_gain_meters"] - target_gain_meters),
            abs(item[2]["total_distance_meters"] - target_distance_meters),
        ),
    )[:partial_base_count]

    for _, route_nodes, metrics in partial_bases:
        accurately_scored.extend(
            partial_edge_tuning_candidates(
                G,
                route_nodes,
                metrics,
                target_distance_meters,
                target_gain_meters,
            )
        )

    best_any = None
    best_acceptable = None
    for score, route_nodes, metrics in accurately_scored:
        distance_error_miles = metrics["distance_error_meters"] / METERS_PER_MILE
        gain_error_ft = metrics["gain_error_meters"] * FEET_PER_METER
        acceptable = (
            distance_error_miles <= limits["distance_error_limit_miles"]
            and gain_error_ft <= limits["gain_error_limit_ft"]
        )

        if best_any is None or score < best_any[0]:
            best_any = (score, route_nodes, metrics)
        if acceptable and (best_acceptable is None or score < best_acceptable[0]):
            best_acceptable = (score, route_nodes, metrics)

    if best_acceptable is not None:
        _, route_nodes, metrics = best_acceptable
        return route_nodes, metrics, last_depth, states_expanded

    _, best_route, best_metrics = best_any
    best_distance = best_metrics["total_distance_meters"] / METERS_PER_MILE
    best_gain = best_metrics["actual_gain_meters"] * FEET_PER_METER
    budget_text = " Search budget was reached." if budget_reached else ""

    raise HTTPException(
        status_code=400,
        detail=(
            "Closed loops were found, but none met the requested quality limits. "
            f"Best accurately scored route was {best_distance:.2f} mi / "
            f"{round(best_gain)} ft gain."
            + budget_text
        ),
    )


def diversify_closed_candidates(
    candidates,
    target_distance_meters,
    target_gain_meters,
    limit,
):
    """Preserve candidates across distance/gain buckets, not only one score."""
    if len(candidates) <= limit:
        return list(candidates)

    buckets = {}
    for candidate in candidates:
        distance_bucket = int(candidate["distance"] / 80.0)
        gain_bucket = int(candidate["gain"] / 12.0)
        key = (distance_bucket, gain_bucket)
        current = buckets.get(key)
        if current is None or candidate["cheap_balanced"] < current["cheap_balanced"]:
            buckets[key] = candidate

    diverse = list(buckets.values())
    diverse.sort(
        key=lambda c: (
            c["cheap_balanced"],
            abs(c["gain"] - target_gain_meters),
            abs(c["distance"] - target_distance_meters),
        )
    )

    # If bucket representatives do not fill the limit, add best leftovers.
    if len(diverse) < limit:
        present = {tuple(c["route"]) for c in diverse}
        leftovers = sorted(candidates, key=lambda c: c["cheap_balanced"])
        for candidate in leftovers:
            key = tuple(candidate["route"])
            if key in present:
                continue
            present.add(key)
            diverse.append(candidate)
            if len(diverse) >= limit:
                break

    return diverse[:limit]


# ============================================================
# WAYPOINT SEARCH FOR 4+ MILE ROUTES
# ============================================================

def waypoint_path(S, source, target, used_edges):
    def weight(u, v, data):
        cost = float(data.get("length", 1.0))

        if undirected_edge_key(u, v) in used_edges:
            cost *= 40.0

        return cost

    return nx.shortest_path(
        S,
        source,
        target,
        weight=weight,
    )


def cheap_waypoint_score(
    G,
    route_nodes,
    target_distance_meters,
    target_gain_meters,
):
    """
    Fast exploratory score using values already stored on graph edges.

    This intentionally avoids route geometry densification and GeoTIFF reads.
    Finalists are rescored later with the authoritative continuous-route DEM
    calculation, so this is only a ranking heuristic.
    """
    total_distance = path_distance_meters(G, route_nodes)
    approximate_gain = path_gain_meters(G, route_nodes)

    if total_distance <= 0:
        return float("inf"), {}

    distance_ratio = abs(total_distance - target_distance_meters) / max(
        target_distance_meters,
        1.0,
    )

    if target_gain_meters > 0:
        gain_ratio = abs(approximate_gain - target_gain_meters) / target_gain_meters
    else:
        gain_ratio = approximate_gain / 30.48

    repeated_edges, repeated_distance = repeated_edge_stats(G, route_nodes)
    repeat_ratio = repeated_distance / max(total_distance, 1.0)
    repeated_nodes = repeated_node_occurrences(route_nodes)
    immediate_reversals = count_immediate_reversals(route_nodes)

    # Distance and approximate elevation are the primary exploratory goals.
    # Repetition remains a meaningful but secondary penalty.
    score = (
        distance_ratio * 190.0
        + gain_ratio * 150.0
        + repeat_ratio * 170.0
        + repeated_nodes * 12.0
        + immediate_reversals * 10.0
    )

    return score, {
        "total_distance_meters": total_distance,
        "approximate_gain_meters": approximate_gain,
        "repeated_edges": repeated_edges,
        "repeated_distance_meters": repeated_distance,
        "repeated_nodes": repeated_nodes,
        "immediate_reversals": immediate_reversals,
    }


def generate_waypoint_loop(
    G,
    start_node,
    target_distance_meters,
    target_gain_meters,
    profile,
    limits,
):
    """
    Two-stage waypoint search for 4+ mile loops.

    Stage 1:
      Generate every configured waypoint attempt and score it cheaply using
      edge length + cached edge ascent. No whole-route DEM sampling occurs.

    Stage 2:
      Rescore only the best diverse finalists using the authoritative
      continuous 5 m DEM profile + ~55 m smoothing.
    """
    S = make_simple_routing_graph(G)

    start_lat = float(G.nodes[start_node]["y"])
    start_lon = float(G.nodes[start_node]["x"])

    candidates = []

    max_radial_distance = min(
        profile["search_radius_m"] * 0.90,
        target_distance_meters * 0.32,
    )

    for node in S.nodes:
        if node == start_node:
            continue

        radial = haversine_meters(
            start_lat,
            start_lon,
            float(G.nodes[node]["y"]),
            float(G.nodes[node]["x"]),
        )

        if (
            radial >= profile["min_anchor_distance_m"]
            and radial <= max_radial_distance
        ):
            candidates.append(node)

    if len(candidates) < max(profile["anchor_counts"]):
        raise HTTPException(
            status_code=400,
            detail="Not enough trail junctions for this route.",
        )

    accurate_finalists = int(profile.get("accurate_finalists", 40))
    pool_multiplier = int(profile.get("candidate_pool_multiplier", 3))
    pool_limit = max(accurate_finalists, accurate_finalists * pool_multiplier)

    exploratory = []
    seen_routes = set()

    for _ in range(profile["attempts"]):
        anchor_count = random.choice(profile["anchor_counts"])
        anchors = random.sample(candidates, anchor_count)

        spacing_bad = False

        for i in range(len(anchors)):
            for j in range(i + 1, len(anchors)):
                a = anchors[i]
                b = anchors[j]

                separation = haversine_meters(
                    float(G.nodes[a]["y"]),
                    float(G.nodes[a]["x"]),
                    float(G.nodes[b]["y"]),
                    float(G.nodes[b]["x"]),
                )

                if separation < profile["min_anchor_separation_m"]:
                    spacing_bad = True
                    break

            if spacing_bad:
                break

        if spacing_bad:
            continue

        anchors.sort(
            key=lambda node: node_angle_from_start(
                G,
                start_node,
                node,
            )
        )

        if random.random() < 0.5:
            anchors.reverse()

        route = [start_node]
        used_edges = set()
        current = start_node
        failed = False

        for destination in anchors + [start_node]:
            try:
                leg = waypoint_path(
                    S,
                    current,
                    destination,
                    used_edges,
                )
            except nx.NetworkXNoPath:
                failed = True
                break

            for i in range(len(leg) - 1):
                used_edges.add(
                    undirected_edge_key(
                        leg[i],
                        leg[i + 1],
                    )
                )

            route.extend(leg[1:])
            current = destination

        if failed:
            continue

        route_key = tuple(route)
        if route_key in seen_routes:
            continue
        seen_routes.add(route_key)

        cheap_score, cheap_metrics = cheap_waypoint_score(
            G,
            route,
            target_distance_meters,
            target_gain_meters,
        )

        distance = cheap_metrics.get("total_distance_meters", 0.0)

        if (
            distance < target_distance_meters * 0.72
            or distance > target_distance_meters * 1.25
        ):
            continue

        # Preserve the best cheap candidates. We periodically trim instead of
        # allowing all 1200 routes to accumulate unnecessarily.
        exploratory.append((cheap_score, route, cheap_metrics))

        if len(exploratory) > pool_limit * 4:
            exploratory.sort(key=lambda item: item[0])
            exploratory = exploratory[:pool_limit]

    if not exploratory:
        raise HTTPException(
            status_code=400,
            detail="No suitable waypoint route found.",
        )

    exploratory.sort(key=lambda item: item[0])

    # Add simple distance/gain buckets so the exact finalists are not all
    # near-duplicates of one cheap optimum.
    finalists = []
    seen_buckets = set()

    for cheap_score, route, cheap_metrics in exploratory:
        distance_bucket = int(cheap_metrics["total_distance_meters"] / 100.0)
        gain_bucket = int(cheap_metrics["approximate_gain_meters"] / 15.0)
        repeat_bucket = int(cheap_metrics["repeated_distance_meters"] / 100.0)
        bucket = (distance_bucket, gain_bucket, repeat_bucket)

        if bucket in seen_buckets and len(finalists) >= accurate_finalists // 2:
            continue

        seen_buckets.add(bucket)
        finalists.append((cheap_score, route))

        if len(finalists) >= accurate_finalists:
            break

    # If diversity filtering left too few, fill from the best remaining routes.
    if len(finalists) < accurate_finalists:
        selected = {tuple(route) for _, route in finalists}
        for cheap_score, route, _ in exploratory:
            if tuple(route) in selected:
                continue
            finalists.append((cheap_score, route))
            selected.add(tuple(route))
            if len(finalists) >= accurate_finalists:
                break

    best_route = None
    best_metrics = None
    best_score = float("inf")

    best_any_route = None
    best_any_metrics = None
    best_any_score = float("inf")

    accurately_scored = 0

    for cheap_score, route in finalists:
        score, metrics = route_score(
            G,
            route,
            target_distance_meters,
            target_gain_meters,
        )
        accurately_scored += 1

        if not metrics:
            continue

        metrics["waypoint_attempts"] = profile["attempts"]
        metrics["waypoint_unique_candidates"] = len(exploratory)
        metrics["waypoint_accurate_finalists"] = accurately_scored
        metrics["waypoint_cheap_score"] = cheap_score

        if score < best_any_score:
            best_any_score = score
            best_any_route = route
            best_any_metrics = metrics

        distance_error_miles = (
            metrics["distance_error_meters"] / METERS_PER_MILE
        )

        gain_error_ft = (
            metrics["gain_error_meters"] * FEET_PER_METER
        )

        if (
            distance_error_miles <= limits["distance_error_limit_miles"]
            and gain_error_ft <= limits["gain_error_limit_ft"]
        ):
            if score < best_score:
                best_score = score
                best_route = route
                best_metrics = metrics

    if best_route is not None:
        best_metrics["waypoint_accurate_finalists"] = accurately_scored
        return (
            best_route,
            best_metrics,
            profile["attempts"],
        )

    if best_any_route is not None:
        best_distance = (
            best_any_metrics["total_distance_meters"] / METERS_PER_MILE
        )

        best_gain = (
            best_any_metrics["actual_gain_meters"] * FEET_PER_METER
        )

        raise HTTPException(
            status_code=400,
            detail=(
                "No route met the requested quality limits. "
                f"Best accurately scored candidate was {best_distance:.2f} mi / "
                f"{round(best_gain)} ft gain after "
                f"{accurately_scored} full DEM finalist evaluations."
            ),
        )

    raise HTTPException(
        status_code=400,
        detail="No suitable waypoint route found after finalist scoring.",
    )

# ============================================================
# GPX PARSING / ANALYSIS
# ============================================================

def parse_gpx_points(gpx_bytes: bytes):
    try:
        root = ET.fromstring(gpx_bytes)
    except ET.ParseError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid GPX/XML file: {exc}",
        )

    points = []

    # Track points first.
    for element in root.iter():
        tag = element.tag.split("}")[-1].lower()

        if tag not in {"trkpt", "rtept"}:
            continue

        lat = element.attrib.get("lat")
        lon = element.attrib.get("lon")

        if lat is None or lon is None:
            continue

        try:
            points.append(
                (
                    float(lon),
                    float(lat),
                )
            )
        except ValueError:
            continue

    if len(points) < 2:
        raise HTTPException(
            status_code=400,
            detail="GPX file contains fewer than two usable track/route points.",
        )

    return points


def polyline_distance_meters(coords):
    total = 0.0

    for i in range(len(coords) - 1):
        lon1, lat1 = coords[i]
        lon2, lat2 = coords[i + 1]

        total += haversine_meters(
            lat1,
            lon1,
            lat2,
            lon2,
        )

    return total


def analyze_gpx_trail_coverage(G, coords):
    """
    Match dense GPX points to the nearest edge in the exact allowed graph.
    Returns distance-based diagnostics in meters.
    """

    if not coords:
        return {
            "coverage_percent": 0.0,
            "mean_distance_to_trail_m": None,
            "max_distance_to_trail_m": None,
            "points_checked": 0,
            "points_within_tolerance": 0,
        }

    projected = ox.projection.project_graph(G)
    projected_crs = projected.graph.get("crs")

    if projected_crs is None:
        raise HTTPException(
            status_code=500,
            detail="Could not determine projected graph CRS for GPX matching.",
        )

    transformer = Transformer.from_crs(
        "EPSG:4326",
        projected_crs,
        always_xy=True,
    )

    lons = [point[0] for point in coords]
    lats = [point[1] for point in coords]
    xs, ys = transformer.transform(lons, lats)

    try:
        _, distances = ox.distance.nearest_edges(
            projected,
            X=list(xs),
            Y=list(ys),
            return_dist=True,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not match GPX points to trail edges: {exc}",
        )

    distances = [float(d) for d in np.atleast_1d(distances)]

    if not distances:
        return {
            "coverage_percent": 0.0,
            "mean_distance_to_trail_m": None,
            "max_distance_to_trail_m": None,
            "points_checked": 0,
            "points_within_tolerance": 0,
        }

    within = sum(
        1
        for distance in distances
        if distance <= GPX_TRAIL_MATCH_TOLERANCE_M
    )

    return {
        "coverage_percent": round(within / len(distances) * 100.0, 1),
        "mean_distance_to_trail_m": round(float(np.mean(distances)), 1),
        "max_distance_to_trail_m": round(float(np.max(distances)), 1),
        "points_checked": len(distances),
        "points_within_tolerance": within,
    }



# ============================================================
# GPX GROUND-TRUTH / GRAPH TRACE DIAGNOSTICS
# ============================================================

def match_gpx_to_graph_edge_runs(G, coords):
    """
    Map dense GPX samples to the nearest allowed graph edges.

    Consecutive samples on the same physical trail segment are collapsed into
    one run. Reverse directed copies of the same OSM segment are treated as the
    same physical edge so bidirectional OSM edges do not create false changes.
    """
    if not coords:
        return {
            "runs": [],
            "sample_count": 0,
            "mean_distance_m": None,
            "max_distance_m": None,
            "continuous_transitions": 0,
            "transition_count": 0,
            "transition_continuity_percent": 0.0,
        }

    projected = ox.projection.project_graph(G)
    projected_crs = projected.graph.get("crs")
    if projected_crs is None:
        raise HTTPException(
            status_code=500,
            detail="Could not determine projected graph CRS for GPX trace matching.",
        )

    transformer = Transformer.from_crs(
        "EPSG:4326",
        projected_crs,
        always_xy=True,
    )

    lons = [float(point[0]) for point in coords]
    lats = [float(point[1]) for point in coords]
    xs, ys = transformer.transform(lons, lats)

    try:
        edge_ids, distances = ox.distance.nearest_edges(
            projected,
            X=list(xs),
            Y=list(ys),
            return_dist=True,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not map GPX samples to graph edges: {exc}",
        )

    edge_ids = list(np.atleast_1d(edge_ids))
    distances = [float(value) for value in np.atleast_1d(distances)]

    runs = []
    for sample_index, edge_id in enumerate(edge_ids):
        try:
            u, v, key = edge_id
        except Exception:
            # Some versions can return a list-like object rather than tuple.
            values = list(edge_id)
            if len(values) < 3:
                continue
            u, v, key = values[:3]

        physical = undirected_edge_key(u, v)
        distance_m = distances[sample_index] if sample_index < len(distances) else None

        if runs and runs[-1]["physical_edge"] == physical:
            runs[-1]["sample_end"] = sample_index
            runs[-1]["samples"] += 1
            if distance_m is not None:
                runs[-1]["max_match_distance_m"] = max(
                    runs[-1]["max_match_distance_m"],
                    distance_m,
                )
            continue

        runs.append({
            "u": int(u),
            "v": int(v),
            "key": int(key) if isinstance(key, (int, np.integer)) else str(key),
            "physical_edge": physical,
            "sample_start": sample_index,
            "sample_end": sample_index,
            "samples": 1,
            "max_match_distance_m": float(distance_m or 0.0),
        })

    transition_count = max(0, len(runs) - 1)
    continuous_transitions = 0
    for left, right in zip(runs, runs[1:]):
        if set(left["physical_edge"]).intersection(right["physical_edge"]):
            continuous_transitions += 1

    continuity = (
        100.0 * continuous_transitions / transition_count
        if transition_count > 0
        else 100.0
    )

    return {
        "runs": runs,
        "sample_count": len(edge_ids),
        "mean_distance_m": round(float(np.mean(distances)), 2) if distances else None,
        "max_distance_m": round(float(np.max(distances)), 2) if distances else None,
        "continuous_transitions": continuous_transitions,
        "transition_count": transition_count,
        "transition_continuity_percent": round(continuity, 1),
    }


def approximate_node_trace_from_edge_runs(G, edge_runs, requested_start_lon, requested_start_lat):
    """
    Convert physical edge runs to an approximate graph-node trace.

    This is exact at graph junctions, but a GPX may begin/end in the middle of
    an OSM edge. In that case we report that the trace is only approximate and
    use the nearest graph endpoint for replay diagnostics.
    """
    if not edge_runs:
        return {
            "nodes": [],
            "start_node": None,
            "start_snap_distance_m": None,
            "continuous": False,
            "closed": False,
            "note": "No matched edge runs were available.",
        }

    start_node = ox.distance.nearest_nodes(
        G,
        X=float(requested_start_lon),
        Y=float(requested_start_lat),
    )
    start_node = int(start_node)

    start_snap_distance_m = haversine_meters(
        float(requested_start_lat),
        float(requested_start_lon),
        float(G.nodes[start_node]["y"]),
        float(G.nodes[start_node]["x"]),
    )

    physical_edges = [tuple(run["physical_edge"]) for run in edge_runs]

    # Shared junctions between consecutive physical edge runs.
    shared_nodes = []
    continuous = True
    for left, right in zip(physical_edges, physical_edges[1:]):
        shared = set(left).intersection(right)
        if len(shared) != 1:
            continuous = False
            shared_nodes.append(None)
        else:
            shared_nodes.append(int(next(iter(shared))))

    nodes = []
    if len(physical_edges) == 1:
        a, b = physical_edges[0]
        nodes = [start_node, b if start_node == a else a]
    elif continuous:
        first_a, first_b = physical_edges[0]
        first_shared = shared_nodes[0]
        first_node = first_b if first_a == first_shared else first_a

        last_a, last_b = physical_edges[-1]
        last_shared = shared_nodes[-1]
        last_node = last_b if last_a == last_shared else last_a

        nodes = [int(first_node)] + [int(n) for n in shared_nodes] + [int(last_node)]

        # If the nearest GPX-start graph node occurs in the trace and this is a
        # graph-closed trace, rotate the closed cycle to that node.
        if len(nodes) >= 2 and nodes[0] == nodes[-1] and start_node in nodes[:-1]:
            index = nodes[:-1].index(start_node)
            cycle = nodes[:-1]
            cycle = cycle[index:] + cycle[:index]
            nodes = cycle + [start_node]

    closed = bool(len(nodes) >= 2 and nodes[0] == nodes[-1])

    note = (
        "Edge transitions are graph-continuous."
        if continuous
        else "Some consecutive nearest-edge runs do not share a graph node; the trace contains map-matching ambiguity."
    )
    if start_snap_distance_m > 20.0:
        note += (
            " The GPX starts noticeably between original OSM graph nodes. "
            "v4 inserts a temporary routing node exactly at the requested GPX start for generation."
        )

    return {
        "nodes": nodes,
        "start_node": start_node,
        "start_snap_distance_m": round(start_snap_distance_m, 1),
        "continuous": continuous,
        "closed": closed,
        "note": note,
    }


def replay_trace_hard_rules(
    G,
    node_trace,
    start_node,
    target_distance_meters,
    limits,
):
    """
    Replay the current hard beam rules against an approximate node trace.

    Elevation is intentionally NOT a hard prune in v3. This checks graph
    continuity, distance overshoot, and shortest-return distance lower bound.
    """
    if not node_trace or len(node_trace) < 2:
        return {
            "status": "not_replayable",
            "first_failure": None,
            "steps_checked": 0,
            "message": "No usable graph-node trace could be reconstructed from the GPX.",
        }

    if int(node_trace[0]) != int(start_node):
        return {
            "status": "not_replayable",
            "first_failure": "gpx_start_is_between_graph_nodes_or_trace_starts_elsewhere",
            "steps_checked": 0,
            "message": (
                "The approximate GPX graph trace does not begin at the snapped beam start node. "
                "This usually means the GPX starts partway along an OSM segment."
            ),
        }

    S = make_simple_routing_graph(G)
    reverse_S = S.reverse(copy=False)
    return_distance = nx.single_source_dijkstra_path_length(
        reverse_S,
        start_node,
        weight="length",
    )

    max_acceptable_distance = target_distance_meters + (
        limits["distance_error_limit_miles"] * METERS_PER_MILE
    )

    cumulative = 0.0
    for step, (u, v) in enumerate(zip(node_trace, node_trace[1:]), start=1):
        if not S.has_edge(u, v):
            return {
                "status": "hard_failure",
                "first_failure": "missing_directed_edge",
                "failure_step": step,
                "steps_checked": step - 1,
                "message": f"Trace step {step} has no directed graph edge {u} -> {v}.",
            }

        cumulative += float(S[u][v].get("length", 0) or 0)

        if cumulative > max_acceptable_distance:
            return {
                "status": "hard_failure",
                "first_failure": "distance_overshoot",
                "failure_step": step,
                "steps_checked": step,
                "cumulative_distance_miles": round(cumulative / METERS_PER_MILE, 3),
                "message": (
                    f"At trace step {step}, cumulative graph distance exceeds the maximum allowed final distance."
                ),
            }

        if v != start_node:
            if v not in return_distance:
                return {
                    "status": "hard_failure",
                    "first_failure": "no_return_path",
                    "failure_step": step,
                    "steps_checked": step,
                    "message": f"At trace step {step}, the state has no path back to the start node.",
                }

            minimum_final = cumulative + float(return_distance[v])
            if minimum_final > max_acceptable_distance:
                return {
                    "status": "hard_failure",
                    "first_failure": "return_distance_bound",
                    "failure_step": step,
                    "steps_checked": step,
                    "cumulative_distance_miles": round(cumulative / METERS_PER_MILE, 3),
                    "minimum_possible_final_miles": round(minimum_final / METERS_PER_MILE, 3),
                    "message": (
                        f"At trace step {step}, current distance plus the shortest possible return exceeds the allowed distance."
                    ),
                }

    return {
        "status": "survives_hard_rules",
        "first_failure": None,
        "steps_checked": len(node_trace) - 1,
        "graph_trace_distance_miles": round(cumulative / METERS_PER_MILE, 3),
        "message": (
            "The reconstructed GPX trace survives the current hard pruning rules. "
            "If the generator still misses it, the likely cause is beam ranking, state deduplication, map-matching/start-edge splitting, or the search budget."
        ),
    }


async def read_and_measure_gpx(file: UploadFile):
    filename = file.filename or "route.gpx"
    if not filename.lower().endswith(".gpx"):
        raise HTTPException(status_code=400, detail="Please upload a .gpx file.")

    gpx_bytes = await file.read()
    if not gpx_bytes:
        raise HTTPException(status_code=400, detail="Uploaded GPX file is empty.")

    raw_coords = parse_gpx_points(gpx_bytes)
    raw_distance_m = polyline_distance_meters(raw_coords)
    dense_coords = densify_polyline(raw_coords, ELEVATION_SAMPLE_SPACING_M)
    raw_dem = elevations_for_coords(dense_coords)
    smoothed = smooth_elevations(raw_dem, radius=ELEVATION_SMOOTHING_RADIUS)
    ascent_m, descent_m = calculate_ascent_descent(smoothed)

    return {
        "filename": filename,
        "raw_coords": raw_coords,
        "dense_coords": dense_coords,
        "distance_m": raw_distance_m,
        "ascent_m": ascent_m,
        "descent_m": descent_m,
    }


@app.post("/test-gpx")
async def test_gpx_against_generator(file: UploadFile = File(...)):
    """
    Ground-truth test:
      1. use the GPX's own start,
      2. use its measured distance and DEM gain as the generator target,
      3. build the allowed graph around that GPX start,
      4. map the GPX to graph edges,
      5. replay hard search rules where possible,
      6. run the actual generator against the same benchmark.
    """
    try:
        measured = await read_and_measure_gpx(file)
        raw_coords = measured["raw_coords"]
        dense_coords = measured["dense_coords"]
        distance_m = measured["distance_m"]
        ascent_m = measured["ascent_m"]
        descent_m = measured["descent_m"]

        start_lon, start_lat = raw_coords[0]
        target_distance_miles = distance_m / METERS_PER_MILE
        target_gain_ft = ascent_m * FEET_PER_METER

        profile = get_route_profile(target_distance_miles)
        limits = get_route_quality_limits(target_distance_miles, target_gain_ft)

        (
            G,
            filtered_edges_removed,
            graph_from_cache,
            unique_elevation_samples,
        ) = download_trail_graph(
            start_lat,
            start_lon,
            profile["search_radius_m"],
        )

        coverage = analyze_gpx_trail_coverage(G, dense_coords)
        edge_match = match_gpx_to_graph_edge_runs(G, dense_coords)
        node_trace = approximate_node_trace_from_edge_runs(
            G,
            edge_match["runs"],
            start_lon,
            start_lat,
        )

        replay_start_node = ox.distance.nearest_nodes(
            G,
            X=start_lon,
            Y=start_lat,
        )
        replay_start_node = int(replay_start_node)

        replay = replay_trace_hard_rules(
            G,
            node_trace["nodes"],
            replay_start_node,
            distance_m,
            limits,
        )

        generator_G, generator_start_node, generator_start_info = insert_exact_routing_point(
            G,
            start_lat,
            start_lon,
        )

        generator = {
            "success": False,
            "error": None,
            "route": [],
        }

        if target_distance_miles < 4.0:
            try:
                route_nodes, metrics, search_steps, states_expanded = beam_search_short_loop(
                    generator_G,
                    generator_start_node,
                    distance_m,
                    ascent_m,
                    limits,
                    profile,
                )

                route_coords = metrics.get("route_coordinates") or route_coordinates(generator_G, route_nodes)
                generator = {
                    "success": True,
                    "error": None,
                    "actual_distance_miles": round(metrics["total_distance_meters"] / METERS_PER_MILE, 3),
                    "actual_gain_ft": round(metrics["actual_gain_meters"] * FEET_PER_METER),
                    "actual_descent_ft": round(metrics.get("actual_descent_meters", 0.0) * FEET_PER_METER),
                    "distance_error_miles": round(metrics["distance_error_meters"] / METERS_PER_MILE, 3),
                    "gain_error_ft": round(metrics["gain_error_meters"] * FEET_PER_METER),
                    "search_steps": search_steps,
                    "states_expanded": states_expanded,
                    "partial_edge_used": bool(metrics.get("partial_edge_used", False)),
                    "route": route_coords,
                }
            except HTTPException as exc:
                generator["error"] = str(exc.detail)
        else:
            generator["error"] = "Ground-truth generator test is currently implemented for routes under 4 miles."

        return {
            "version": APP_VERSION,
            "filename": measured["filename"],
            "benchmark": {
                "start": {"lat": start_lat, "lon": start_lon},
                "distance_miles": round(target_distance_miles, 3),
                "gain_ft": round(target_gain_ft),
                "descent_ft": round(descent_m * FEET_PER_METER),
                "raw_gpx_points": len(raw_coords),
                "dem_sample_points": len(dense_coords),
                "route": [
                    {"lat": float(lat), "lon": float(lon)}
                    for lon, lat in raw_coords
                ],
            },
            "graph": {
                "start_node": generator_start_node,
                "start_snap_distance_m": float(generator_start_info["routing_offset_m"]),
                "exact_start_inserted": bool(generator_start_info["exact_inserted"]),
                "start_trail_offset_m": float(generator_start_info["trail_offset_m"]),
                "original_nearest_node": replay_start_node,
                "original_node_snap_distance_m": node_trace["start_snap_distance_m"],
                "nodes": generator_G.number_of_nodes(),
                "edges": generator_G.number_of_edges(),
                "search_radius_m": profile["search_radius_m"],
                "filtered_edges_removed": filtered_edges_removed,
                "from_cache": graph_from_cache,
                "elevation_samples": unique_elevation_samples,
            },
            "coverage": {
                "percent": coverage["coverage_percent"],
                "mean_distance_m": coverage["mean_distance_to_trail_m"],
                "max_distance_m": coverage["max_distance_to_trail_m"],
                "within_tolerance": coverage["points_within_tolerance"],
                "checked": coverage["points_checked"],
            },
            "edge_trace": {
                "physical_edge_runs": len(edge_match["runs"]),
                "unique_physical_edges": len({tuple(run["physical_edge"]) for run in edge_match["runs"]}),
                "transition_count": edge_match["transition_count"],
                "continuous_transitions": edge_match["continuous_transitions"],
                "transition_continuity_percent": edge_match["transition_continuity_percent"],
                "mean_match_distance_m": edge_match["mean_distance_m"],
                "max_match_distance_m": edge_match["max_distance_m"],
                "approximate_node_count": len(node_trace["nodes"]),
                "approximate_trace_continuous": node_trace["continuous"],
                "approximate_trace_closed": node_trace["closed"],
                "note": node_trace["note"],
                "first_nodes": node_trace["nodes"][:30],
            },
            "hard_rule_replay": replay,
            "generator": generator,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/analyze-gpx")
async def analyze_gpx(
    file: UploadFile = File(...),
    start_lat: float = Query(DEFAULT_LAT),
    start_lon: float = Query(DEFAULT_LON),
    target_distance_miles: float = Query(2.5),
    target_gain_ft: float = Query(200.0),
):
    try:
        filename = file.filename or "route.gpx"

        if not filename.lower().endswith(".gpx"):
            raise HTTPException(
                status_code=400,
                detail="Please upload a .gpx file.",
            )

        gpx_bytes = await file.read()

        if not gpx_bytes:
            raise HTTPException(
                status_code=400,
                detail="Uploaded GPX file is empty.",
            )

        raw_coords = parse_gpx_points(gpx_bytes)

        raw_distance_m = polyline_distance_meters(raw_coords)

        dense_coords = densify_polyline(
            raw_coords,
            ELEVATION_SAMPLE_SPACING_M,
        )

        raw_dem_elevations = elevations_for_coords(dense_coords)
        smoothed_dem_elevations = smooth_elevations(
            raw_dem_elevations,
            radius=ELEVATION_SMOOTHING_RADIUS,
        )

        ascent_m, descent_m = calculate_ascent_descent(
            smoothed_dem_elevations
        )

        closure_distance_m = haversine_meters(
            raw_coords[0][1],
            raw_coords[0][0],
            raw_coords[-1][1],
            raw_coords[-1][0],
        )

        distance_from_requested_start_m = haversine_meters(
            start_lat,
            start_lon,
            raw_coords[0][1],
            raw_coords[0][0],
        )

        profile = get_route_profile(target_distance_miles)

        (
            G,
            filtered_edges_removed,
            graph_from_cache,
            unique_elevation_samples,
        ) = download_trail_graph(
            start_lat,
            start_lon,
            profile["search_radius_m"],
        )

        coverage = analyze_gpx_trail_coverage(
            G,
            dense_coords,
        )

        distance_miles = raw_distance_m / METERS_PER_MILE
        gain_ft = ascent_m * FEET_PER_METER
        descent_ft = descent_m * FEET_PER_METER

        distance_error_miles = abs(
            distance_miles - target_distance_miles
        )

        gain_error_ft = abs(
            gain_ft - target_gain_ft
        )

        limits = get_route_quality_limits(
            target_distance_miles,
            target_gain_ft,
        )

        meets_distance = (
            distance_error_miles
            <= limits["distance_error_limit_miles"]
        )

        meets_gain = (
            gain_error_ft
            <= limits["gain_error_limit_ft"]
        )

        return {
            "filename": filename,
            "raw_gpx_points": len(raw_coords),
            "dem_sample_points": len(dense_coords),
            "distance_miles": round(distance_miles, 3),
            "gain_ft": round(gain_ft),
            "descent_ft": round(descent_ft),
            "closure_distance_m": round(closure_distance_m, 1),
            "distance_from_requested_start_m": round(
                distance_from_requested_start_m,
                1,
            ),
            "start": {
                "lat": raw_coords[0][1],
                "lon": raw_coords[0][0],
            },
            "finish": {
                "lat": raw_coords[-1][1],
                "lon": raw_coords[-1][0],
            },
            "target_distance_miles": target_distance_miles,
            "target_gain_ft": target_gain_ft,
            "distance_error_miles": round(distance_error_miles, 3),
            "gain_error_ft": round(gain_error_ft),
            "meets_distance_limit": meets_distance,
            "meets_gain_limit": meets_gain,
            "meets_both_limits": meets_distance and meets_gain,
            "allowed_distance_error_miles": round(
                limits["distance_error_limit_miles"],
                3,
            ),
            "allowed_gain_error_ft": round(
                limits["gain_error_limit_ft"]
            ),
            "trail_coverage_percent": coverage["coverage_percent"],
            "trail_match_tolerance_m": GPX_TRAIL_MATCH_TOLERANCE_M,
            "mean_distance_to_allowed_trail_m": coverage[
                "mean_distance_to_trail_m"
            ],
            "max_distance_to_allowed_trail_m": coverage[
                "max_distance_to_trail_m"
            ],
            "trail_match_points_checked": coverage["points_checked"],
            "trail_match_points_within_tolerance": coverage[
                "points_within_tolerance"
            ],
            "filtered_edges_removed": filtered_edges_removed,
            "graph_from_cache": graph_from_cache,
            "graph_elevation_samples": unique_elevation_samples,
            "elevation_sample_spacing_m": ELEVATION_SAMPLE_SPACING_M,
            "elevation_smoothing_window_points": 2 * ELEVATION_SMOOTHING_RADIUS + 1,
            "elevation_smoothing_distance_m": (2 * ELEVATION_SMOOTHING_RADIUS + 1) * ELEVATION_SAMPLE_SPACING_M,
            "elevation_source": os.path.basename(DEM_PATH),
            "version": APP_VERSION,
            "route": [
                {
                    "lat": float(lat),
                    "lon": float(lon),
                }
                for lon, lat in raw_coords
            ],
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# ============================================================
# TRAIL NETWORK ENDPOINT
# ============================================================

@app.post("/trail-network")
def trail_network(request: TrailNetworkRequest):
    try:
        profile = get_route_profile(request.target_distance_miles)

        (
            G,
            filtered_edges_removed,
            graph_from_cache,
            unique_elevation_samples,
        ) = download_trail_graph(
            request.start_lat,
            request.start_lon,
            profile["search_radius_m"],
        )

        G, start_node, start_info = insert_exact_routing_point(
            G,
            request.start_lat,
            request.start_lon,
        )

        snapped_lat = float(G.nodes[start_node]["y"])
        snapped_lon = float(G.nodes[start_node]["x"])
        snap_distance = float(start_info["routing_offset_m"])

        segments = graph_debug_segments(G)
        _, master_info = get_master_trail_graph()

        return {
            "allowed_trails": segments,
            "allowed_trail_segments": len(segments),
            "network_nodes": G.number_of_nodes(),
            "network_edges": G.number_of_edges(),
            "master_network_nodes": master_info["nodes"],
            "master_network_edges": master_info["edges"],
            "master_physical_segments": master_info["physical_segments"],
            "master_loaded_from_disk": master_info["loaded_from_disk"],
            "master_tiff": os.path.basename(DEM_PATH),
            "search_radius_m": profile["search_radius_m"],
            "route_profile": profile["name"],
            "version": APP_VERSION,
            "requested_start": {
                "lat": request.start_lat,
                "lon": request.start_lon,
            },
            "snapped_start": {
                "lat": snapped_lat,
                "lon": snapped_lon,
            },
            "snap_distance_m": round(snap_distance, 1),
            "exact_start_inserted": bool(start_info["exact_inserted"]),
            "start_trail_offset_m": float(start_info["trail_offset_m"]),
            "start_source_edge": start_info.get("source_edge"),
            "filtered_edges_removed": filtered_edges_removed,
            "graph_from_cache": graph_from_cache,
            "elevation_samples": unique_elevation_samples,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# ============================================================
# DEM INFO
# ============================================================

@app.get("/dem-info")
def dem_info():
    if not os.path.exists(DEM_PATH):
        raise HTTPException(
            status_code=404,
            detail="DEM file not found.",
        )

    with rasterio.open(DEM_PATH) as src:
        return {
            "file": os.path.basename(DEM_PATH),
            "crs": str(src.crs),
            "width": src.width,
            "height": src.height,
            "bounds": {
                "left": src.bounds.left,
                "bottom": src.bounds.bottom,
                "right": src.bounds.right,
                "top": src.bounds.top,
            },
            "pixel_size": {
                "x": abs(src.transform.a),
                "y": abs(src.transform.e),
            },
            "elevation_sample_spacing_m": ELEVATION_SAMPLE_SPACING_M,
            "elevation_smoothing_window_points": 2 * ELEVATION_SMOOTHING_RADIUS + 1,
            "version": APP_VERSION,
        }


# ============================================================
# GPX EXPORT PROFILE
# ============================================================

def build_gpx_export_points(coords):
    """
    Build the COROS GPX track as coordinates only.

    The route is densified to ~5 m for a faithful track shape, but no <ele>
    values are calculated or embedded. COROS can calculate route elevation
    using its own processing when the file is imported.
    """
    if not coords or len(coords) < 2:
        return []

    lonlat = [
        (float(point["lon"]), float(point["lat"]))
        for point in coords
    ]

    dense = densify_polyline(
        lonlat,
        ELEVATION_SAMPLE_SPACING_M,
    )

    return [
        {
            "lat": float(lat),
            "lon": float(lon),
        }
        for lon, lat in dense
    ]

# ============================================================
# GENERATE ROUTE
# ============================================================

@app.post("/generate-route")
def generate_route(request: RouteRequest):
    try:
        if request.target_distance_miles <= 0:
            raise HTTPException(
                status_code=400,
                detail="Distance must be greater than 0.",
            )

        if request.target_gain_ft < 0:
            raise HTTPException(
                status_code=400,
                detail="Elevation gain cannot be negative.",
            )

        target_distance_meters = (
            request.target_distance_miles * METERS_PER_MILE
        )

        target_gain_meters = (
            request.target_gain_ft / FEET_PER_METER
        )

        profile = get_route_profile(
            request.target_distance_miles
        )

        limits = get_route_quality_limits(
            request.target_distance_miles,
            request.target_gain_ft,
        )

        (
            G,
            filtered_edges_removed,
            graph_from_cache,
            unique_elevation_samples,
        ) = download_trail_graph(
            request.start_lat,
            request.start_lon,
            profile["search_radius_m"],
        )

        same_point = (
            abs(request.start_lat - request.end_lat) < 0.0001
            and abs(request.start_lon - request.end_lon) < 0.0001
        )

        G, start_node, start_info = insert_exact_routing_point(
            G,
            request.start_lat,
            request.start_lon,
        )

        if same_point:
            end_node = start_node
        else:
            end_node = ox.distance.nearest_nodes(
                G,
                X=request.end_lon,
                Y=request.end_lat,
            )

        snapped_start_lat = float(G.nodes[start_node]["y"])
        snapped_start_lon = float(G.nodes[start_node]["x"])
        snap_distance_m = float(start_info["routing_offset_m"])

        if same_point:
            if request.target_distance_miles < 4.0:
                (
                    route_nodes,
                    metrics,
                    search_steps,
                    states_expanded,
                ) = beam_search_short_loop(
                    G,
                    start_node,
                    target_distance_meters,
                    target_gain_meters,
                    limits,
                    profile,
                )

                route_type = "true closed-loop trail beam search"
                search_method = "closed-loop-beam"

            else:
                (
                    route_nodes,
                    metrics,
                    search_steps,
                ) = generate_waypoint_loop(
                    G,
                    start_node,
                    target_distance_meters,
                    target_gain_meters,
                    profile,
                    limits,
                )

                route_type = "adaptive waypoint trail loop"
                search_method = "waypoint"
                states_expanded = None

        else:
            S = make_simple_routing_graph(G)

            try:
                route_nodes = nx.shortest_path(
                    S,
                    start_node,
                    end_node,
                    weight="length",
                )
            except nx.NetworkXNoPath:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "No connected trail route found between start and finish."
                    ),
                )

            _, metrics = route_score(
                G,
                route_nodes,
                target_distance_meters,
                target_gain_meters,
            )

            search_steps = 1
            states_expanded = None
            search_method = "point-to-point"
            route_type = "trail point-to-point"

        route_distance_miles = (
            metrics["total_distance_meters"] / METERS_PER_MILE
        )

        actual_gain_ft = (
            metrics["actual_gain_meters"] * FEET_PER_METER
        )

        distance_error_miles = abs(
            route_distance_miles - request.target_distance_miles
        )

        elevation_error_ft = abs(
            actual_gain_ft - request.target_gain_ft
        )

        repeated_distance_miles = (
            metrics["repeated_distance_meters"] / METERS_PER_MILE
        )

        coords = metrics.get("route_coordinates") or route_coordinates(G, route_nodes)

        # Build the no-elevation COROS GPX track only after the winning route
        # has been selected. This is geometry-only and performs no extra DEM
        # raster sampling.
        gpx_export_points = build_gpx_export_points(coords)

        return {
            "requested_distance_miles": request.target_distance_miles,
            "actual_distance_miles": round(route_distance_miles, 2),
            "distance_error_miles": round(distance_error_miles, 2),
            "requested_gain_ft": request.target_gain_ft,
            "actual_gain_ft": round(actual_gain_ft),
            "actual_descent_ft": round(metrics.get("actual_descent_meters", 0.0) * FEET_PER_METER),
            "elevation_error_ft": round(elevation_error_ft),
            "route_type": route_type,
            "search_method": search_method,
            "route_profile": profile["name"],
            "route": coords,
            "gpx_export_points": gpx_export_points,
            "route_nodes": len(route_nodes),
            "route_geometry_points": len(coords),
            "repeated_edges": metrics["repeated_edges"],
            "repeated_distance_miles": round(
                repeated_distance_miles,
                2,
            ),
            "repeated_nodes": metrics["repeated_nodes"],
            "immediate_reversals": metrics["immediate_reversals"],
            "route_score": round(metrics["score"], 2),
            "max_allowed_distance_error_miles": round(
                limits["distance_error_limit_miles"],
                2,
            ),
            "max_allowed_gain_error_ft": round(
                limits["gain_error_limit_ft"]
            ),
            "search_steps": search_steps,
            "states_expanded": states_expanded,
            "waypoint_unique_candidates": metrics.get("waypoint_unique_candidates"),
            "waypoint_accurate_finalists": metrics.get("waypoint_accurate_finalists"),
            "search_radius_m": profile["search_radius_m"],
            "network_nodes": G.number_of_nodes(),
            "network_edges": G.number_of_edges(),
            "filtered_edges_removed": filtered_edges_removed,
            "graph_from_cache": graph_from_cache,
            "unique_elevation_samples": unique_elevation_samples,
            "elevation_sample_spacing_m": ELEVATION_SAMPLE_SPACING_M,
            "elevation_smoothing_window_points": 2 * ELEVATION_SMOOTHING_RADIUS + 1,
            "elevation_smoothing_distance_m": (2 * ELEVATION_SMOOTHING_RADIUS + 1) * ELEVATION_SAMPLE_SPACING_M,
            "elevation_source": os.path.basename(DEM_PATH),
            "route_elevation_sample_count": metrics.get("route_elevation_sample_count"),
            "partial_edge_used": metrics.get("partial_edge_used", False),
            "partial_added_distance_miles": round(metrics.get("partial_added_distance_meters", 0.0) / METERS_PER_MILE, 3),
            "partial_outward_distance_meters": round(metrics.get("partial_outward_distance_meters", 0.0), 1),
            "version": APP_VERSION,
            "snapped_start_lat": snapped_start_lat,
            "snapped_start_lon": snapped_start_lon,
            "snap_distance_m": round(snap_distance_m, 1),
            "exact_start_inserted": bool(start_info["exact_inserted"]),
            "start_trail_offset_m": float(start_info["trail_offset_m"]),
            "start_source_edge": start_info.get("source_edge"),
            "status": "Route generated",
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# ============================================================
# MAP PAGE
# ============================================================

@app.get("/map", response_class=HTMLResponse)
def route_map():
    return r"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Trail Running Creator</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">

<style>
* {
    box-sizing: border-box;
}

body {
    margin: 0;
    font-family: Arial, sans-serif;
    background: #f5f5f5;
}

#controls {
    padding: 16px;
    background: white;
    border-bottom: 1px solid #ccc;
}

h2 {
    margin: 0 0 14px 0;
}

h3 {
    margin: 16px 0 8px 0;
}

.input-row {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin-bottom: 12px;
}

.input-group {
    display: flex;
    flex-direction: column;
}

label {
    font-size: 13px;
    margin-bottom: 4px;
    font-weight: bold;
}

input[type="number"] {
    width: 170px;
    padding: 8px;
    border: 1px solid #aaa;
    border-radius: 4px;
}

input[type="file"] {
    max-width: 360px;
}

button {
    padding: 10px 18px;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    background: #222;
    color: white;
    font-size: 15px;
    margin-right: 8px;
    margin-bottom: 6px;
}

button:disabled {
    opacity: 0.5;
    cursor: wait;
}

.network-control {
    margin-top: 8px;
}

#gpx-panel {
    border-top: 1px solid #ddd;
    margin-top: 16px;
    padding-top: 4px;
}

#results,
#gpxResults {
    margin-top: 12px;
    line-height: 1.55;
}

#diagnostics {
    margin-top: 8px;
    line-height: 1.45;
    font-size: 13px;
    color: #555;
}

#map {
    height: 650px;
    width: 100%;
}

.error {
    color: #b00020;
}

.success {
    color: #166534;
}

.warning {
    color: #9a6700;
}

.small {
    font-size: 12px;
    color: #666;
}

@media (max-width: 700px) {
    input[type="number"] {
        width: 145px;
    }

    #map {
        height: 60vh;
        min-height: 480px;
    }
}
</style>
</head>

<body>

<div id="controls">

<h2>Trail Running Creator</h2>
<div style="font-size:12px;color:#666;margin-bottom:10px;">Version: 2026-08-09-v7-full-tiff-master-network</div>

<div class="input-row">
    <div class="input-group">
        <label for="start_lat">Start latitude</label>
        <input id="start_lat" type="number" step="any" value="33.586055">
    </div>

    <div class="input-group">
        <label for="start_lon">Start longitude</label>
        <input id="start_lon" type="number" step="any" value="-112.083341">
    </div>

    <div class="input-group">
        <label for="end_lat">End latitude</label>
        <input id="end_lat" type="number" step="any" value="33.586055">
    </div>

    <div class="input-group">
        <label for="end_lon">End longitude</label>
        <input id="end_lon" type="number" step="any" value="-112.083341">
    </div>
</div>

<div class="input-row">
    <div class="input-group">
        <label for="distance">Target distance (miles)</label>
        <input id="distance" type="number" step="0.1" min="0.1" value="2.5">
    </div>

    <div class="input-group">
        <label for="gain">Target elevation gain (ft)</label>
        <input id="gain" type="number" step="25" min="0" value="200">
    </div>
</div>

<button id="generateButton">Generate Trail Route</button>
<button id="downloadGpxButton" disabled>Download GPX for COROS</button>
<button id="networkButton">Reload Allowed Trails</button>

<div class="network-control">
    <label>
        <input id="showNetwork" type="checkbox" checked>
        Show allowed trail network
    </label>
</div>

<div id="results">Ready.</div>
<div id="diagnostics"></div>

<div id="gpx-panel">
    <h3>Analyze Manual GPX</h3>

    <div class="input-row">
        <div class="input-group">
            <label for="gpxFile">GPX file</label>
            <input id="gpxFile" type="file" accept=".gpx,application/gpx+xml,application/xml,text/xml">
        </div>
    </div>

    <button id="analyzeGpxButton">Analyze GPX</button>
    <button id="testGpxButton">Test Generator Against GPX</button>
    <button id="clearGpxButton">Clear GPX</button>

    <div id="gpxResults">
        Upload a manual GPX to compare it with the exact same DEM and allowed trail network.
    </div>
</div>

</div>

<div id="map"></div>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>

<script>
const map = L.map("map").setView(
    [33.586055, -112.083341],
    15
);

L.tileLayer(
    "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    {
        maxZoom: 19,
        attribution: "&copy; OpenStreetMap contributors"
    }
).addTo(map);

let routeLine = null;
let gpxLine = null;
let networkLayer = L.layerGroup();
let lastGeneratedRoute = null;
let requestedStartMarker = null;
let snappedStartMarker = null;
let snapLine = null;

const generateButton = document.getElementById("generateButton");
const downloadGpxButton = document.getElementById("downloadGpxButton");
const networkButton = document.getElementById("networkButton");
const analyzeGpxButton = document.getElementById("analyzeGpxButton");
const testGpxButton = document.getElementById("testGpxButton");
const clearGpxButton = document.getElementById("clearGpxButton");
const showNetworkCheckbox = document.getElementById("showNetwork");


generateButton.addEventListener("click", generateRoute);
downloadGpxButton.addEventListener("click", downloadGeneratedGpx);
networkButton.addEventListener("click", reloadNetwork);
analyzeGpxButton.addEventListener("click", analyzeGpx);
testGpxButton.addEventListener("click", testGpxAgainstGenerator);
clearGpxButton.addEventListener("click", clearGpx);
showNetworkCheckbox.addEventListener("change", updateNetworkVisibility);


function escapeXml(value) {
    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&apos;");
}


function buildGpxXml(points, routeName) {
    const trackPoints = points.map(point => {
        const lat = Number(point.lat).toFixed(7);
        const lon = Number(point.lon).toFixed(7);
        return `      <trkpt lat="${lat}" lon="${lon}"></trkpt>`;
    }).join("\n");

    return `<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="Trail Running Creator" xmlns="http://www.topografix.com/GPX/1/1" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.topografix.com/GPX/1/1 http://www.topografix.com/GPX/1/1/gpx.xsd">
  <metadata>
    <name>${escapeXml(routeName)}</name>
  </metadata>
  <trk>
    <name>${escapeXml(routeName)}</name>
    <trkseg>
${trackPoints}
    </trkseg>
  </trk>
</gpx>
`;
}


function triggerTextDownload(filename, contents, mimeType) {
    const blob = new Blob([contents], {type: mimeType});
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");

    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();

    setTimeout(() => URL.revokeObjectURL(url), 1000);
}


function downloadGeneratedGpx() {
    if (!lastGeneratedRoute) {
        return;
    }

    const points = lastGeneratedRoute.gpx_export_points;

    if (!points || points.length < 2) {
        alert("The generated route does not contain enough points to export.");
        return;
    }

    const distance = Number(lastGeneratedRoute.actual_distance_miles).toFixed(2);
    const gain = Math.round(Number(lastGeneratedRoute.actual_gain_ft));
    const routeName = `Trail Route ${distance} mi - COROS`;
    const filename = `trail-route-${distance}mi-${gain}ft-coros.gpx`;
    const xml = buildGpxXml(points, routeName);

    triggerTextDownload(filename, xml, "application/gpx+xml;charset=utf-8");
}

function updateNetworkVisibility() {
    if (showNetworkCheckbox.checked) {
        if (!map.hasLayer(networkLayer)) {
            networkLayer.addTo(map);
        }
    } else {
        if (map.hasLayer(networkLayer)) {
            map.removeLayer(networkLayer);
        }
    }
}


function getInputData() {
    return {
        start_lat: parseFloat(document.getElementById("start_lat").value),
        start_lon: parseFloat(document.getElementById("start_lon").value),
        end_lat: parseFloat(document.getElementById("end_lat").value),
        end_lon: parseFloat(document.getElementById("end_lon").value),
        target_distance_miles: parseFloat(document.getElementById("distance").value),
        target_gain_ft: parseFloat(document.getElementById("gain").value)
    };
}


async function readJsonResponse(response) {
    const text = await response.text();

    if (!text) {
        throw new Error("Server returned an empty response.");
    }

    let result;

    try {
        result = JSON.parse(text);
    } catch {
        throw new Error("Invalid server response: " + text.substring(0, 500));
    }

    if (!response.ok) {
        throw new Error(result.detail || "Server error.");
    }

    return result;
}


async function loadTrailNetwork(data) {
    const diagnostics = document.getElementById("diagnostics");

    const response = await fetch(
        "/trail-network",
        {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                start_lat: data.start_lat,
                start_lon: data.start_lon,
                target_distance_miles: data.target_distance_miles
            })
        }
    );

    const result = await readJsonResponse(response);

    networkLayer.clearLayers();

    for (const segment of result.allowed_trails) {
        L.polyline(
            segment,
            {
                weight: 3,
                opacity: 0.45,
                color: "#666666"
            }
        ).addTo(networkLayer);
    }

    updateNetworkVisibility();

    if (requestedStartMarker) {
        map.removeLayer(requestedStartMarker);
    }

    if (snappedStartMarker) {
        map.removeLayer(snappedStartMarker);
    }

    if (snapLine) {
        map.removeLayer(snapLine);
    }

    requestedStartMarker = L.circleMarker(
        [
            result.requested_start.lat,
            result.requested_start.lon
        ],
        {
            radius: 6,
            weight: 2,
            color: "#0066cc",
            fillOpacity: 0.8
        }
    )
    .addTo(map)
    .bindPopup("Requested start");

    snappedStartMarker = L.circleMarker(
        [
            result.snapped_start.lat,
            result.snapped_start.lon
        ],
        {
            radius: 6,
            weight: 2,
            color: "#ff9900",
            fillOpacity: 0.8
        }
    )
    .addTo(map)
    .bindPopup(
        "Graph start<br>" +
        result.snap_distance_m +
        " m from requested coordinate"
    );

    if (result.snap_distance_m > 3) {
        snapLine = L.polyline(
            [
                [
                    result.requested_start.lat,
                    result.requested_start.lon
                ],
                [
                    result.snapped_start.lat,
                    result.snapped_start.lon
                ]
            ],
            {
                dashArray: "5,5",
                weight: 2,
                opacity: 0.8,
                color: "#ff9900"
            }
        ).addTo(map);
    }

    diagnostics.innerHTML =
        "<b>Allowed trail network:</b> " +
        result.allowed_trail_segments +
        " physical segments<br>" +
        "<b>Master TIFF trail network:</b> " +
        result.master_physical_segments +
        " physical segments<br>" +
        "<b>Master TIFF:</b> " +
        result.master_tiff +
        "<br>" +
        "<b>Graph nodes:</b> " +
        result.network_nodes +
        "<br>" +
        "<b>Graph edges:</b> " +
        result.network_edges +
        "<br>" +
        "<b>Start snap distance:</b> " +
        result.snap_distance_m +
        " m<br>" +
        "<b>Exact start inserted:</b> " +
        (result.exact_start_inserted ? "YES" : "NO") +
        "<br>" +
        "<b>Requested point → trail:</b> " +
        result.start_trail_offset_m +
        " m<br>" +
        "<b>Search radius:</b> " +
        result.search_radius_m +
        " m<br>" +
        "<b>Profile:</b> " +
        result.route_profile +
        "<br><b>Version:</b> " +
        result.version;

    return result;
}


async function reloadNetwork() {
    const data = getInputData();
    const diagnostics = document.getElementById("diagnostics");

    networkButton.disabled = true;
    diagnostics.innerHTML = '<span class="warning">Loading allowed trail network...</span>';

    try {
        await loadTrailNetwork(data);
    } catch (error) {
        diagnostics.innerHTML = '<span class="error"><b>Error:</b> ' + error.message + '</span>';
    } finally {
        networkButton.disabled = false;
    }
}


async function generateRoute() {
    const results = document.getElementById("results");
    const data = getInputData();

    results.innerHTML = '<span class="warning">Loading allowed trails...</span>';
    generateButton.disabled = true;
    lastGeneratedRoute = null;
    downloadGpxButton.disabled = true;

    try {
        await loadTrailNetwork(data);

        results.innerHTML = '<span class="warning">Searching closed-loop trail combinations...</span>';

        const response = await fetch(
            "/generate-route",
            {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(data)
            }
        );

        const result = await readJsonResponse(response);

        if (!result.route || result.route.length < 2) {
            throw new Error("Server returned an empty route.");
        }

        if (!result.gpx_export_points || result.gpx_export_points.length < 2) {
            throw new Error("Server returned a route but no GPX export profile.");
        }

        lastGeneratedRoute = result;
        downloadGpxButton.disabled = false;

        const coordinates = result.route.map(
            point => [point.lat, point.lon]
        );

        if (routeLine) {
            map.removeLayer(routeLine);
        }

        routeLine = L.polyline(
            coordinates,
            {
                weight: 6,
                opacity: 0.95,
                color: "#d60000"
            }
        ).addTo(map);

        routeLine.bringToFront();

        map.fitBounds(
            routeLine.getBounds(),
            {
                padding: [30, 30]
            }
        );

        const expandedText =
            result.states_expanded === null
            ? "N/A"
            : result.states_expanded;

        results.innerHTML =
            '<span class="success"><b>Route generated</b></span><br>' +
            "<b>Distance target:</b> " + result.requested_distance_miles + " mi<br>" +
            "<b>Actual distance:</b> " + result.actual_distance_miles + " mi<br>" +
            "<b>Distance error:</b> " + result.distance_error_miles + " mi<br><br>" +
            "<b>Elevation target:</b> " + result.requested_gain_ft + " ft<br>" +
            "<b>Actual elevation gain:</b> " + result.actual_gain_ft + " ft<br>" +
            "<b>Actual descent:</b> " + result.actual_descent_ft + " ft<br>" +
            "<b>Elevation error:</b> " + result.elevation_error_ft + " ft<br>" +
            "<b>Partial-edge tuning:</b> " + (result.partial_edge_used ? "YES" : "NO") + "<br>" +
            (result.partial_edge_used
                ? "<b>Partial distance added:</b> " + result.partial_added_distance_miles + " mi<br>" +
                  "<b>Turnaround distance from node:</b> " + result.partial_outward_distance_meters + " m<br>"
                : "") +
            "<br>" +
            "<b>Search method:</b> " + result.search_method + "<br>" +
            "<b>Route profile:</b> " + result.route_profile + "<br>" +
            "<b>Search depth:</b> " + result.search_steps + "<br>" +
            "<b>States expanded:</b> " + expandedText + "<br>" +
            "<b>Start snap distance:</b> " + result.snap_distance_m + " m<br>" +
            "<b>Exact start inserted:</b> " + (result.exact_start_inserted ? "YES" : "NO") + "<br>" +
            "<b>Requested point → trail:</b> " + result.start_trail_offset_m + " m<br><br>" +
            "<b>Repeated trail distance:</b> " + result.repeated_distance_miles + " mi<br>" +
            "<b>Repeated edges:</b> " + result.repeated_edges + "<br>" +
            "<b>Repeated junctions:</b> " + result.repeated_nodes + "<br>" +
            "<b>Immediate reversals:</b> " + result.immediate_reversals + "<br>" +
            "<b>Route score:</b> " + result.route_score + "<br><br>" +
            "<b>Graph cached:</b> " + result.graph_from_cache + "<br>" +
            "<b>Elevation samples:</b> " + result.unique_elevation_samples + "<br>" +
            "<b>Elevation sample spacing:</b> ~" + result.elevation_sample_spacing_m + " m<br>" +
            "<b>Elevation smoothing:</b> ~" + result.elevation_smoothing_distance_m + " m (" + result.elevation_smoothing_window_points + " points)<br>" +
            "<b>Version:</b> " + result.version + "<br>" +
            "<b>GPX export points:</b> " + result.gpx_export_points.length + "<br>" +
            (result.waypoint_accurate_finalists !== null && result.waypoint_accurate_finalists !== undefined
                ? "<b>Accurate waypoint finalists:</b> " + result.waypoint_accurate_finalists + "<br>"
                : "") +
            '<span class="small">Download GPX exports coordinates only for COROS; no elevation is embedded.</span><br>' +
            '<span class="small">Elevation source used for route selection: ' + result.elevation_source + "</span>";

    } catch (error) {
        results.innerHTML =
            '<span class="error"><b>Error:</b> ' +
            error.message +
            "</span><br>" +
            '<span class="small">The gray lines remain visible. If you have a manual GPX that works, upload it below and click Analyze GPX.</span>';
    } finally {
        generateButton.disabled = false;
    }
}


async function analyzeGpx() {
    const gpxResults = document.getElementById("gpxResults");
    const fileInput = document.getElementById("gpxFile");
    const data = getInputData();

    if (!fileInput.files || fileInput.files.length === 0) {
        gpxResults.innerHTML = '<span class="error"><b>Error:</b> Choose a GPX file first.</span>';
        return;
    }

    analyzeGpxButton.disabled = true;
    gpxResults.innerHTML = '<span class="warning">Analyzing GPX with the same DEM and trail network...</span>';

    try {
        await loadTrailNetwork(data);

        const formData = new FormData();
        formData.append("file", fileInput.files[0]);

        const query = new URLSearchParams({
            start_lat: data.start_lat,
            start_lon: data.start_lon,
            target_distance_miles: data.target_distance_miles,
            target_gain_ft: data.target_gain_ft
        });

        const response = await fetch(
            "/analyze-gpx?" + query.toString(),
            {
                method: "POST",
                body: formData
            }
        );

        const result = await readJsonResponse(response);

        const coordinates = result.route.map(
            point => [point.lat, point.lon]
        );

        if (gpxLine) {
            map.removeLayer(gpxLine);
        }

        gpxLine = L.polyline(
            coordinates,
            {
                weight: 6,
                opacity: 0.95,
                color: "#0066cc"
            }
        ).addTo(map);

        gpxLine.bringToFront();

        map.fitBounds(
            gpxLine.getBounds(),
            {
                padding: [30, 30]
            }
        );

        const passText = result.meets_both_limits ? "YES" : "NO";

        gpxResults.innerHTML =
            '<span class="success"><b>GPX analyzed</b></span><br>' +
            "<b>File:</b> " + result.filename + "<br>" +
            "<b>Distance:</b> " + result.distance_miles + " mi<br>" +
            "<b>DEM elevation gain:</b> " + result.gain_ft + " ft<br>" +
            "<b>DEM descent:</b> " + result.descent_ft + " ft<br>" +
            "<b>Distance error:</b> " + result.distance_error_miles + " mi<br>" +
            "<b>Gain error:</b> " + result.gain_error_ft + " ft<br>" +
            "<b>Meets generator quality limits:</b> " + passText + "<br><br>" +
            "<b>Closed-loop gap:</b> " + result.closure_distance_m + " m<br>" +
            "<b>GPX start from requested start:</b> " + result.distance_from_requested_start_m + " m<br><br>" +
            "<b>Allowed-trail coverage:</b> " + result.trail_coverage_percent + "%<br>" +
            "<b>Trail matching tolerance:</b> " + result.trail_match_tolerance_m + " m<br>" +
            "<b>Mean distance to allowed trail:</b> " + result.mean_distance_to_allowed_trail_m + " m<br>" +
            "<b>Max distance to allowed trail:</b> " + result.max_distance_to_allowed_trail_m + " m<br>" +
            "<b>Matched samples:</b> " + result.trail_match_points_within_tolerance + " / " + result.trail_match_points_checked + "<br><br>" +
            "<b>Raw GPX points:</b> " + result.raw_gpx_points + "<br>" +
            "<b>DEM sample points:</b> " + result.dem_sample_points + "<br>" +
            "<b>Elevation sample spacing:</b> ~" + result.elevation_sample_spacing_m + " m<br>" +
            "<b>Elevation smoothing window:</b> " + result.elevation_smoothing_window_points + " points<br>" +
            '<span class="small">Blue = manual GPX, red = generated route, gray = allowed trail network.</span>';

    } catch (error) {
        gpxResults.innerHTML = '<span class="error"><b>Error:</b> ' + error.message + "</span>";
    } finally {
        analyzeGpxButton.disabled = false;
    }
}


async function testGpxAgainstGenerator() {
    const gpxResults = document.getElementById("gpxResults");
    const fileInput = document.getElementById("gpxFile");

    if (!fileInput.files || fileInput.files.length === 0) {
        gpxResults.innerHTML = '<span class="error"><b>Error:</b> Choose a GPX file first.</span>';
        return;
    }

    testGpxButton.disabled = true;
    analyzeGpxButton.disabled = true;
    gpxResults.innerHTML = '<span class="warning">Running ground-truth test from the GPX own start, distance, and DEM gain. This may take about 20 seconds...</span>';

    try {
        const formData = new FormData();
        formData.append("file", fileInput.files[0]);

        const response = await fetch(
            "/test-gpx",
            {
                method: "POST",
                body: formData
            }
        );

        const result = await readJsonResponse(response);
        const benchmark = result.benchmark;
        const trace = result.edge_trace;

        if (benchmark.route && benchmark.route.length >= 2) {
            const benchmarkCoordinates = benchmark.route.map(point => [point.lat, point.lon]);
            if (gpxLine) {
                map.removeLayer(gpxLine);
            }
            gpxLine = L.polyline(
                benchmarkCoordinates,
                {
                    weight: 6,
                    opacity: 0.95,
                    color: "#0066cc"
                }
            ).addTo(map);
            gpxLine.bringToFront();
        }
        const replay = result.hard_rule_replay;
        const generator = result.generator;

        // Put the GPX benchmark start/target into the main form so a normal
        // Generate click immediately repeats the same benchmark.
        document.getElementById("start_lat").value = benchmark.start.lat;
        document.getElementById("start_lon").value = benchmark.start.lon;
        document.getElementById("end_lat").value = benchmark.start.lat;
        document.getElementById("end_lon").value = benchmark.start.lon;
        document.getElementById("distance").value = benchmark.distance_miles;
        document.getElementById("gain").value = benchmark.gain_ft;

        if (generator.success && generator.route && generator.route.length >= 2) {
            const generatedCoordinates = generator.route.map(point => [point.lat, point.lon]);
            if (routeLine) {
                map.removeLayer(routeLine);
            }
            routeLine = L.polyline(
                generatedCoordinates,
                {
                    weight: 6,
                    opacity: 0.95,
                    color: "#d60000"
                }
            ).addTo(map);
            routeLine.bringToFront();
        }

        const generatorText = generator.success
            ? '<span class="success"><b>Generator benchmark result: SUCCESS</b></span><br>' +
              '<b>Generated distance:</b> ' + generator.actual_distance_miles + ' mi<br>' +
              '<b>Generated gain:</b> ' + generator.actual_gain_ft + ' ft<br>' +
              '<b>Distance error:</b> ' + generator.distance_error_miles + ' mi<br>' +
              '<b>Gain error:</b> ' + generator.gain_error_ft + ' ft<br>' +
              '<b>States expanded:</b> ' + generator.states_expanded + '<br>' +
              '<b>Partial-edge tuning:</b> ' + (generator.partial_edge_used ? 'YES' : 'NO')
            : '<span class="error"><b>Generator benchmark result: FAILED</b></span><br>' +
              '<b>Generator error:</b> ' + (generator.error || 'Unknown error');

        gpxResults.innerHTML =
            '<span class="success"><b>GPX ground-truth test complete</b></span><br>' +
            '<b>Version:</b> ' + result.version + '<br><br>' +
            '<b>Benchmark start:</b> ' + benchmark.start.lat.toFixed(6) + ', ' + benchmark.start.lon.toFixed(6) + '<br>' +
            '<b>Benchmark distance:</b> ' + benchmark.distance_miles + ' mi<br>' +
            '<b>Benchmark DEM gain:</b> ' + benchmark.gain_ft + ' ft<br>' +
            '<b>Benchmark descent:</b> ' + benchmark.descent_ft + ' ft<br>' +
            '<b>Graph-start snap:</b> ' + result.graph.start_snap_distance_m + ' m<br>' +
            '<b>Exact GPX start inserted:</b> ' + (result.graph.exact_start_inserted ? 'YES' : 'NO') + '<br>' +
            '<b>GPX start → trail:</b> ' + result.graph.start_trail_offset_m + ' m<br><br>' +
            '<b>Allowed-trail coverage:</b> ' + result.coverage.percent + '%<br>' +
            '<b>Mean trail match distance:</b> ' + result.coverage.mean_distance_m + ' m<br>' +
            '<b>Max trail match distance:</b> ' + result.coverage.max_distance_m + ' m<br><br>' +
            '<b>Matched physical edge runs:</b> ' + trace.physical_edge_runs + '<br>' +
            '<b>Unique physical edges:</b> ' + trace.unique_physical_edges + '<br>' +
            '<b>Edge-transition continuity:</b> ' + trace.transition_continuity_percent + '%<br>' +
            '<b>Approximate graph trace closed:</b> ' + (trace.approximate_trace_closed ? 'YES' : 'NO') + '<br>' +
            '<b>Trace note:</b> ' + trace.note + '<br><br>' +
            '<b>Hard-rule replay:</b> ' + replay.status + '<br>' +
            '<b>Replay explanation:</b> ' + replay.message + '<br><br>' +
            generatorText + '<br><br>' +
            '<span class="small">Blue = uploaded GPX. Red = route independently found by the generator using the GPX own start/distance/gain. The form has been updated to this benchmark automatically.</span>';

        await loadTrailNetwork(getInputData());

    } catch (error) {
        gpxResults.innerHTML = '<span class="error"><b>Error:</b> ' + error.message + '</span>';
    } finally {
        testGpxButton.disabled = false;
        analyzeGpxButton.disabled = false;
    }
}


function clearGpx() {
    if (gpxLine) {
        map.removeLayer(gpxLine);
        gpxLine = null;
    }

    document.getElementById("gpxFile").value = "";
    document.getElementById("gpxResults").innerHTML =
        "Upload a manual GPX to compare it with the exact same DEM and allowed trail network.";
}


// Load the default allowed network when the page opens.
reloadNetwork();
</script>

</body>
</html>
"""
