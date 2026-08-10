from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.responses import HTMLResponse, Response
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel

import math
import os
import json
import random
import time
import threading
import pickle
import sys
import xml.etree.ElementTree as ET

import networkx as nx
import numpy as np
import osmnx as ox
import rasterio
from pyproj import Transformer
from shapely.geometry import LineString, MultiPoint
from rasterio.warp import transform as rio_transform, transform_bounds as rio_transform_bounds
from sklearn.neighbors import BallTree


app = FastAPI()
app.add_middleware(GZipMiddleware, minimum_size=1000)


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

MAX_CACHED_GRAPHS = 10
GRAPH_CACHE = {}

# One filtered OSM natural-trail graph covering the entire DEM/TIFF footprint.
# Walkable roads are NOT stored TIFF-wide. V9 fetches only a few narrow
# connector corridors on demand for long routes, which keeps Render memory low.
MASTER_GRAPH = None
MASTER_GRAPH_INFO = {}
MASTER_GRAPH_LOCK = threading.Lock()
MASTER_GRAPH_GRAPHML_PATH = os.path.join(BASE_DIR, "master_trails.graphml")
MASTER_GRAPH_PICKLE_PATH = os.path.join(BASE_DIR, "master_trails.pkl")
# GraphML is the portable repository file. The pickle is an optional fast cache.
MASTER_GRAPH_PATH = MASTER_GRAPH_GRAPHML_PATH
DEM_BOUNDS_WGS84_CACHE = None

# Cache DEM values by rounded lat/lon. Graph construction already samples
# most trail points, so later route scoring can reuse those values instead
# of reopening/resampling the GeoTIFF for every finalist.
DEM_POINT_CACHE = {}
MAX_DEM_POINT_CACHE = 250000

APP_VERSION = "2026-08-09-v13-big-loop-preference"
MASTER_NETWORK_SCHEMA = "trail-only-v11-offline-precomputed"
ELEVATION_SMOOTHING_RADIUS = 5  # 11 points total ~= 55 m at 5 m spacing
PARTIAL_TUNING_MAX_DEFICIT_M = 0.75 * METERS_PER_MILE
TRAIL_HIGHWAYS = {"path", "track", "steps"}
HARD_TRAIL_SURFACES = {"asphalt", "concrete", "concrete:lanes", "concrete:plates", "paving_stones", "sett", "cobblestone"}
CONNECTOR_HIGHWAYS = {"footway", "pedestrian", "cycleway", "bridleway", "residential", "living_street", "service", "unclassified", "tertiary", "secondary", "primary", "road"}
CONNECTOR_PATH_COST_MULTIPLIER = 2.5
CONNECTOR_FINAL_SCORE_WEIGHT = 120.0
CONNECTOR_CHEAP_SCORE_WEIGHT = 90.0

# V10 returns several materially different successful candidates from the same
# search instead of throwing away every route except the winner.
MAX_ROUTE_OPTIONS = 5
MAX_ROUTE_SHARED_FRACTION = 0.80

# V9 keeps the TIFF-wide master graph trail-only. For long routes it downloads
# only a few narrow walkable-road corridors needed to bridge nearby disconnected
# trail systems. This avoids holding the entire urban street network in memory.
SELECTIVE_CONNECTORS_MIN_RADIUS_M = 8.0 * METERS_PER_MILE
SELECTIVE_CONNECTOR_MAX_COUNT = 4
SELECTIVE_CONNECTOR_MIN_COMPONENT_TRAIL_M = 250.0
SELECTIVE_CONNECTOR_MAX_GAP_M = 7000.0
SELECTIVE_CONNECTOR_ATTACH_MAX_M = 90.0
SELECTIVE_CONNECTOR_CACHE = {}
MAX_SELECTIVE_CONNECTOR_CACHE = 16

# V12 separates the expensive routing-area preparation from distance/elevation
# targets. A workspace is keyed only by the requested start coordinate and
# contains the TIFF-wide trail graph plus the small selective connector set.
# Changing distance/gain reuses this workspace and only creates a cheap in-memory
# radius subgraph for the actual route search.
MAX_CACHED_WORKSPACES = 3
WORKSPACE_CACHE = {}
WORKSPACE_CACHE_LOCK = threading.Lock()

# The gray TIFF-wide trail overlay is serialized once per server process and
# served as compact, gzip-compressed JSON. The browser fetches it once and keeps
# it while distance/elevation targets change.
MASTER_TRAIL_OVERLAY_JSON = None
MASTER_TRAIL_OVERLAY_LOCK = threading.Lock()
CONNECTOR_FILTER = '["highway"~"footway|pedestrian|cycleway|bridleway|residential|living_street|service|unclassified|tertiary|secondary|primary|road"]'


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
    force_reload: bool = False


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
    search_radius_m = max(250, int(float(target_distance_miles) * METERS_PER_MILE))

    if target_distance_miles < 4.0:
        return {
            "name": "short-closed-beam",
            "search_radius_m": search_radius_m,
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
            "search_radius_m": search_radius_m,
            "attempts": 1200,
            "anchor_counts": [2, 3, 3, 3],
            "min_anchor_distance_m": 150,
            "min_anchor_separation_m": 140,
            "accurate_finalists": 24,
            "candidate_pool_multiplier": 3,
        }
    if target_distance_miles < 15.0:
        return {
            "name": "long-waypoint",
            "search_radius_m": search_radius_m,
            "attempts": 900,
            "anchor_counts": [3, 4, 4, 4],
            "min_anchor_distance_m": 300,
            "min_anchor_separation_m": 250,
            "accurate_finalists": 24,
            "candidate_pool_multiplier": 3,
        }
    return {
        "name": "ultra-waypoint",
        "search_radius_m": search_radius_m,
        "attempts": 700,
        "anchor_counts": [4, 4, 5],
        "min_anchor_distance_m": 400,
        "min_anchor_separation_m": 300,
        "accurate_finalists": 24,
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

def edge_routing_cost(data):
    length = float(data.get("length", 0) or 0)
    if str(data.get("route_class", "trail")) == "connector":
        return max(0.0, length) * CONNECTOR_PATH_COST_MULTIPLIER
    return max(0.0, length)


def get_shortest_edge(G, u, v):
    edge_data = G.get_edge_data(u, v)
    if not edge_data:
        return None
    return min(edge_data.values(), key=lambda edge: (edge_routing_cost(edge), float(edge.get("length", float("inf")))))


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
    attrs["routing_cost"] = float(edge_routing_cost(attrs))
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

    snap_graph = trail_only_graph(G)
    if snap_graph.number_of_edges() == 0:
        snap_graph = G

    projected = ox.projection.project_graph(snap_graph)
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
        fallback = int(ox.distance.nearest_nodes(snap_graph, X=lon, Y=lat))
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
        fallback = int(ox.distance.nearest_nodes(snap_graph, X=lon, Y=lat))
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
    """Return natural trail geometry for the gray map overlay; hide connectors."""
    segments = []
    seen = set()
    for u, v, key, data in G.edges(keys=True, data=True):
        if str(data.get("route_class", "trail")) != "trail":
            continue
        physical_key = (min(int(u), int(v)), max(int(u), int(v)), round(float(data.get("length", 0) or 0), 1))
        if physical_key in seen:
            continue
        seen.add(physical_key)
        coords = oriented_edge_coords(G, u, v, data)
        if len(coords) >= 2:
            segments.append([[float(lat), float(lon)] for lon, lat in coords])
    return segments


# ============================================================
# TRAIL FILTER
# ============================================================

def edge_access_allowed(data):
    access = normalize_tag_values(data.get("access"))
    foot = normalize_tag_values(data.get("foot"))
    area = normalize_tag_values(data.get("area"))
    indoor = normalize_tag_values(data.get("indoor"))
    if "yes" in area or "yes" in indoor:
        return False
    if "no" in foot:
        return False
    if access.intersection({"no", "private"}) and not foot.intersection({"yes", "designated", "permissive"}):
        return False
    return True


def classify_walkable_edge(data):
    if not edge_access_allowed(data):
        return None
    highways = normalize_tag_values(data.get("highway"))
    surfaces = normalize_tag_values(data.get("surface"))
    if highways.intersection(TRAIL_HIGHWAYS):
        return "connector" if surfaces.intersection(HARD_TRAIL_SURFACES) else "trail"
    if highways.intersection(CONNECTOR_HIGHWAYS):
        return "connector"
    return None


def edge_is_allowed_trail(data):
    return classify_walkable_edge(data) == "trail"


def trail_edge_ids(G):
    return [(u, v, key) for u, v, key, data in G.edges(keys=True, data=True) if str(data.get("route_class", "trail")) == "trail"]


def trail_only_graph(G):
    edges = trail_edge_ids(G)
    return G.edge_subgraph(edges).copy() if edges else G.__class__()


def graph_route_class_counts(G):
    trail_edges = connector_edges = 0
    for _, _, _, data in G.edges(keys=True, data=True):
        if str(data.get("route_class", "trail")) == "connector":
            connector_edges += 1
        else:
            trail_edges += 1
    return trail_edges, connector_edges


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
        if str(data.get("route_class", "trail")) != "trail":
            G[u][v][key]["ascent_m"] = 0.0
            G[u][v][key]["descent_m"] = 0.0
            G[u][v][key]["elevation_sample_count"] = 0
            continue
        coords = oriented_edge_coords(G, u, v, data)
        samples = densify_polyline(coords, ELEVATION_SAMPLE_SPACING_M)
        if len(samples) < 2:
            samples = [(float(G.nodes[u]["x"]), float(G.nodes[u]["y"])), (float(G.nodes[v]["x"]), float(G.nodes[v]["y"]))]
        edge_samples[(u, v, key)] = samples
        all_points.extend(samples)
    lookup = sample_dem_points(all_points) if all_points else {}
    for (u, v, key), samples in edge_samples.items():
        elevations = [float(lookup[(round(float(lat), 7), round(float(lon), 7))]) for lon, lat in samples]
        elevations = smooth_elevations(elevations, radius=ELEVATION_SMOOTHING_RADIUS)
        ascent, descent = calculate_ascent_descent(elevations)
        G[u][v][key]["ascent_m"] = float(ascent)
        G[u][v][key]["descent_m"] = float(descent)
        G[u][v][key]["elevation_sample_count"] = len(samples)
    return G, len(lookup)


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
        "sidewalk",
        "service",
    ]

    useful_tags = list(ox.settings.useful_tags_way)

    for tag in extra_tags:
        if tag not in useful_tags:
            useful_tags.append(tag)

    ox.settings.useful_tags_way = useful_tags


def master_graph_metadata(G, loaded_from_disk=False):
    physical = set()
    trail_physical = set()

    for u, v, key, data in G.edges(keys=True, data=True):
        pk = (
            min(int(u), int(v)),
            max(int(u), int(v)),
            round(float(data.get("length", 0) or 0), 1),
        )
        physical.add(pk)
        trail_physical.add(pk)

    return {
        "nodes": int(G.number_of_nodes()),
        "edges": int(G.number_of_edges()),
        "physical_segments": len(physical),
        "trail_physical_segments": len(trail_physical),
        "connector_physical_segments": 0,
        "filtered_edges_removed": int(
            float(G.graph.get("master_filtered_edges_removed", 0) or 0)
        ),
        "loaded_from_disk": bool(loaded_from_disk),
        "elevation_precomputed": str(
            G.graph.get("master_elevation_precomputed", "0")
        ) == "1",
        "elevation_samples": int(
            float(G.graph.get("master_elevation_unique_samples", 0) or 0)
        ),
        "bbox": get_dem_bounds_wgs84(),
        "saved_graph": os.path.basename(
            str(G.graph.get("master_loaded_source", MASTER_GRAPH_PATH))
        ),
    }


def _validate_offline_master_graph(G):
    if G is None:
        return False
    if not isinstance(G, (nx.MultiDiGraph, nx.MultiGraph, nx.DiGraph, nx.Graph)):
        return False
    if str(G.graph.get("dem_signature", "")) != get_dem_signature():
        return False
    if str(G.graph.get("master_network_schema", "")) != MASTER_NETWORK_SCHEMA:
        return False
    if str(G.graph.get("master_elevation_precomputed", "0")) != "1":
        return False
    if not G.number_of_nodes() or not G.number_of_edges():
        return False

    # A v11 offline file must already contain elevation heuristics on every
    # natural-trail edge so ordinary requests never resample the full graph.
    for _, _, _, data in G.edges(keys=True, data=True):
        if str(data.get("route_class", "trail")) != "trail":
            continue
        try:
            float(data["ascent_m"])
            float(data["descent_m"])
            float(data.get("routing_cost", data.get("length", 0) or 0))
        except Exception:
            return False
    return True


def try_load_saved_master_graph():
    """
    Load the prebuilt TIFF-wide graph without contacting OpenStreetMap.

    Prefer a local pickle cache because it is fastest. If the pickle was made
    by an incompatible Python/library version, fall back to the portable
    GraphML committed to the repo and refresh the local pickle automatically.
    """
    if os.path.exists(MASTER_GRAPH_PICKLE_PATH):
        try:
            with open(MASTER_GRAPH_PICKLE_PATH, "rb") as f:
                G = pickle.load(f)
            if _validate_offline_master_graph(G):
                G.graph["master_loaded_source"] = MASTER_GRAPH_PICKLE_PATH
                return G
        except Exception:
            pass

    if os.path.exists(MASTER_GRAPH_GRAPHML_PATH):
        try:
            G = ox.io.load_graphml(filepath=MASTER_GRAPH_GRAPHML_PATH)
            if _validate_offline_master_graph(G):
                G.graph["master_loaded_source"] = MASTER_GRAPH_GRAPHML_PATH
                # Best-effort local binary cache. It is not required in Git.
                try:
                    with open(MASTER_GRAPH_PICKLE_PATH + ".tmp", "wb") as f:
                        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
                    os.replace(
                        MASTER_GRAPH_PICKLE_PATH + ".tmp",
                        MASTER_GRAPH_PICKLE_PATH,
                    )
                except Exception:
                    pass
                return G
        except Exception:
            pass

    return None


def save_master_graph(G):
    """
    Save both formats:
      * master_trails.graphml = portable file to commit to GitHub
      * master_trails.pkl     = optional faster local cache
    """
    try:
        ox.io.save_graphml(G, filepath=MASTER_GRAPH_GRAPHML_PATH)
    except Exception:
        return False

    # The pickle is a convenience cache. A GraphML-only repo is fully valid.
    tmp_path = MASTER_GRAPH_PICKLE_PATH + ".tmp"
    try:
        with open(tmp_path, "wb") as f:
            pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, MASTER_GRAPH_PICKLE_PATH)
    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

    return True


def build_master_trail_graph():
    """
    ONE-TIME OFFLINE BUILD.

    Download every allowed natural trail inside the TIFF, filter it, precompute
    the 5 m / ~55 m-smoothed edge elevation heuristics, and save the resulting graph to master_trails.graphml (plus an optional
    fast pickle cache). Render never needs to rebuild this graph.

    This function requires internet access to OpenStreetMap/Overpass and is
    intended to be run on the user's computer before committing the .pkl file.
    """
    configure_osmnx_trail_tags()
    bbox = get_dem_bounds_wgs84()
    trail_filter = '["highway"~"path|track|steps"]'

    print("Building offline master trail graph...")
    print(
        "TIFF bounds: "
        f"west={bbox[0]:.6f}, south={bbox[1]:.6f}, "
        f"east={bbox[2]:.6f}, north={bbox[3]:.6f}"
    )

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
        raise RuntimeError(
            f"Could not download the master TIFF trail network: {exc}"
        ) from exc

    original_edges = G.number_of_edges()
    remove = []

    for u, v, key, data in G.edges(keys=True, data=True):
        if not edge_is_allowed_trail(data):
            remove.append((u, v, key))
            continue
        if not edge_fully_inside_dem(G, u, v, data):
            remove.append((u, v, key))
            continue

        data["route_class"] = "trail"
        data["routing_cost"] = float(data.get("length", 0) or 0)

    G.remove_edges_from(remove)
    G.remove_nodes_from(list(nx.isolates(G)))

    if not G.number_of_edges():
        raise RuntimeError(
            "No usable natural trail network was found inside the TIFF footprint."
        )

    print(
        f"Filtered graph: {G.number_of_nodes()} nodes / "
        f"{G.number_of_edges()} directed edges"
    )
    print("Precomputing trail elevation heuristics from output_USGS10m.tif...")

    # This is deliberately done once during the offline build. It is the same
    # edge-elevation heuristic v10 used at request time.
    G, unique_samples = add_local_dem_edge_elevations(G)

    for _, _, _, data in G.edges(keys=True, data=True):
        data["routing_cost"] = float(edge_routing_cost(data))

    G.graph["dem_signature"] = get_dem_signature()
    G.graph["master_filtered_edges_removed"] = int(
        original_edges - G.number_of_edges()
    )
    G.graph["master_tiff_name"] = os.path.basename(DEM_PATH)
    G.graph["master_network_version"] = APP_VERSION
    G.graph["master_network_schema"] = MASTER_NETWORK_SCHEMA
    G.graph["master_elevation_precomputed"] = "1"
    G.graph["master_elevation_unique_samples"] = int(unique_samples)
    G.graph["master_elevation_spacing_m"] = float(ELEVATION_SAMPLE_SPACING_M)
    G.graph["master_elevation_smoothing_radius"] = int(ELEVATION_SMOOTHING_RADIUS)

    if not save_master_graph(G):
        raise RuntimeError(f"Could not save {MASTER_GRAPH_PATH}")

    graphml_mb = os.path.getsize(MASTER_GRAPH_GRAPHML_PATH) / (1024 * 1024)
    print(f"Saved portable graph: {MASTER_GRAPH_GRAPHML_PATH}")
    print(f"GraphML size: {graphml_mb:.2f} MB")
    if os.path.exists(MASTER_GRAPH_PICKLE_PATH):
        pickle_mb = os.path.getsize(MASTER_GRAPH_PICKLE_PATH) / (1024 * 1024)
        print(f"Saved fast cache: {MASTER_GRAPH_PICKLE_PATH} ({pickle_mb:.2f} MB)")
    print(f"Unique DEM samples baked into build: {unique_samples}")
    print("Commit master_trails.graphml beside main.py. master_trails.pkl is optional.")
    return G


def get_master_trail_graph():
    global MASTER_GRAPH, MASTER_GRAPH_INFO

    if MASTER_GRAPH is not None:
        return MASTER_GRAPH, MASTER_GRAPH_INFO

    with MASTER_GRAPH_LOCK:
        if MASTER_GRAPH is not None:
            return MASTER_GRAPH, MASTER_GRAPH_INFO

        G = try_load_saved_master_graph()
        if G is None:
            reason = (
                "Offline master graph is missing or incompatible. "
                "Put master_trails.graphml beside main.py. To create it once on a "
                "computer with internet access, run: python main.py --build-master"
            )
            raise HTTPException(status_code=503, detail=reason)

        MASTER_GRAPH = G
        MASTER_GRAPH_INFO = master_graph_metadata(
            G,
            loaded_from_disk=True,
        )

    return MASTER_GRAPH, MASTER_GRAPH_INFO


def extract_local_master_subgraph(master_G, lat, lon, radius_meters):
    """
    Return every natural-trail component inside the requested radius/bbox.

    Unlike v7/v8, disconnected trail systems are not discarded here. This is
    why the gray overlay can show all available trails in the large search area.
    The route search later limits anchors to nodes actually reachable from the
    requested start, after selective connector corridors are added.
    """
    if not point_inside_dem(lat, lon):
        left, bottom, right, top = get_dem_bounds_wgs84()
        raise HTTPException(
            status_code=400,
            detail=(
                "Start coordinate is outside TIFF coverage: "
                f"west={left:.6f}, east={right:.6f}, "
                f"south={bottom:.6f}, north={top:.6f}."
            ),
        )

    local_bbox = ox.utils_geo.bbox_from_point(
        (float(lat), float(lon)),
        float(radius_meters),
    )

    dl, db, dr, dt = get_dem_bounds_wgs84()
    left = max(float(local_bbox[0]), dl)
    bottom = max(float(local_bbox[1]), db)
    right = min(float(local_bbox[2]), dr)
    top = min(float(local_bbox[3]), dt)

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

    if not local.number_of_edges():
        raise HTTPException(
            status_code=400,
            detail="No natural trails were found near this start coordinate.",
        )

    return local.copy()


def _component_unique_trail_length(G, nodes):
    nodes = set(nodes)
    seen = set()
    total = 0.0

    for u in nodes:
        if u not in G:
            continue
        for _, v, key, data in G.out_edges(u, keys=True, data=True):
            if v not in nodes:
                continue
            if str(data.get("route_class", "trail")) != "trail":
                continue
            pk = (
                min(int(u), int(v)),
                max(int(u), int(v)),
                round(float(data.get("length", 0) or 0), 1),
            )
            if pk in seen:
                continue
            seen.add(pk)
            total += float(data.get("length", 0) or 0)

    return total


def _closest_node_pair_cross_graph(Ga, nodes_a, Gb, nodes_b):
    a = [n for n in nodes_a if n in Ga]
    b = [n for n in nodes_b if n in Gb]

    if not a or not b:
        return None

    coords_a = np.radians(
        np.array(
            [[float(Ga.nodes[n]["y"]), float(Ga.nodes[n]["x"])] for n in a],
            dtype=float,
        )
    )
    coords_b = np.radians(
        np.array(
            [[float(Gb.nodes[n]["y"]), float(Gb.nodes[n]["x"])] for n in b],
            dtype=float,
        )
    )

    tree = BallTree(coords_a, metric="haversine")
    distances, indices = tree.query(coords_b, k=1)
    row = int(np.argmin(distances[:, 0]))
    ai = int(indices[row, 0])
    distance_m = float(distances[row, 0]) * 6371000.0

    return a[ai], b[row], distance_m


def _closest_node_pair_same_graph(G, nodes_a, nodes_b):
    return _closest_node_pair_cross_graph(G, nodes_a, G, nodes_b)


def _connector_bbox(lat1, lon1, lat2, lon2, padding_m):
    mean_lat = (float(lat1) + float(lat2)) / 2.0
    lat_pad = float(padding_m) / 111320.0
    lon_scale = max(0.2, math.cos(math.radians(mean_lat)))
    lon_pad = float(padding_m) / (111320.0 * lon_scale)

    left = min(float(lon1), float(lon2)) - lon_pad
    right = max(float(lon1), float(lon2)) + lon_pad
    bottom = min(float(lat1), float(lat2)) - lat_pad
    top = max(float(lat1), float(lat2)) + lat_pad

    dl, db, dr, dt = get_dem_bounds_wgs84()
    return (
        max(left, dl),
        max(bottom, db),
        min(right, dr),
        min(top, dt),
    )


def _nodes_in_bbox(G, nodes, bbox):
    left, bottom, right, top = bbox
    result = []

    for n in nodes:
        if n not in G:
            continue
        lat = float(G.nodes[n]["y"])
        lon = float(G.nodes[n]["x"])
        if left <= lon <= right and bottom <= lat <= top:
            result.append(n)

    return result


def _nearest_attachment(local_G, trail_nodes, walk_G, preferred_node, bbox):
    """
    Return (trail_node, walk_node, offset_m).

    Prefer an exact shared OSM node between a trail and the walkable connector
    graph. Only if OSM topology does not share a node do we permit a very short
    <=90 m attachment, which is marked separately as a synthetic attachment.
    """
    trail_nodes = _nodes_in_bbox(local_G, trail_nodes, bbox)
    if not trail_nodes:
        return None

    shared = [n for n in trail_nodes if n in walk_G]

    if shared:
        preferred_lat = float(local_G.nodes[preferred_node]["y"])
        preferred_lon = float(local_G.nodes[preferred_node]["x"])
        best = min(
            shared,
            key=lambda n: haversine_meters(
                preferred_lat,
                preferred_lon,
                float(local_G.nodes[n]["y"]),
                float(local_G.nodes[n]["x"]),
            ),
        )
        return best, best, 0.0

    pair = _closest_node_pair_cross_graph(
        local_G,
        trail_nodes,
        walk_G,
        list(walk_G.nodes),
    )

    if pair is None:
        return None

    trail_node, walk_node, distance_m = pair

    if distance_m > SELECTIVE_CONNECTOR_ATTACH_MAX_M:
        return None

    return trail_node, walk_node, float(distance_m)


def _download_connector_corridor(local_G, source_nodes, target_nodes, source_hint, target_hint):
    source_lat = float(local_G.nodes[source_hint]["y"])
    source_lon = float(local_G.nodes[source_hint]["x"])
    target_lat = float(local_G.nodes[target_hint]["y"])
    target_lon = float(local_G.nodes[target_hint]["x"])

    straight_gap = haversine_meters(
        source_lat,
        source_lon,
        target_lat,
        target_lon,
    )

    padding_m = min(1100.0, max(400.0, straight_gap * 0.16))
    bbox = _connector_bbox(
        source_lat,
        source_lon,
        target_lat,
        target_lon,
        padding_m,
    )

    if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
        return None, "connector corridor falls outside TIFF"

    cache_key = (
        round(source_lat, 4),
        round(source_lon, 4),
        round(target_lat, 4),
        round(target_lon, 4),
        round(padding_m, -1),
    )

    reverse_key = (
        cache_key[2],
        cache_key[3],
        cache_key[0],
        cache_key[1],
        cache_key[4],
    )

    cached = SELECTIVE_CONNECTOR_CACHE.get(cache_key)
    if cached is None:
        cached = SELECTIVE_CONNECTOR_CACHE.get(reverse_key)

    if cached is not None:
        W = cached.copy()
    else:
        try:
            W = ox.graph.graph_from_bbox(
                bbox,
                network_type="walk",
                custom_filter=CONNECTOR_FILTER,
                simplify=True,
                retain_all=True,
                truncate_by_edge=False,
            )
        except Exception as exc:
            return None, f"connector OSM query failed: {exc}"

        remove = []
        for u, v, key, data in W.edges(keys=True, data=True):
            if classify_walkable_edge(data) != "connector":
                remove.append((u, v, key))
                continue
            if not edge_fully_inside_dem(W, u, v, data):
                remove.append((u, v, key))
                continue
            data["route_class"] = "connector"
            data["routing_cost"] = (
                float(data.get("length", 0) or 0)
                * CONNECTOR_PATH_COST_MULTIPLIER
            )

        W.remove_edges_from(remove)
        W.remove_nodes_from(list(nx.isolates(W)))

        if not W.number_of_edges():
            return None, "no usable walkable connector ways in corridor"

        if len(SELECTIVE_CONNECTOR_CACHE) >= MAX_SELECTIVE_CONNECTOR_CACHE:
            oldest = next(iter(SELECTIVE_CONNECTOR_CACHE))
            SELECTIVE_CONNECTOR_CACHE.pop(oldest)

        SELECTIVE_CONNECTOR_CACHE[cache_key] = W.copy()

    source_attach = _nearest_attachment(
        local_G,
        source_nodes,
        W,
        source_hint,
        bbox,
    )
    target_attach = _nearest_attachment(
        local_G,
        target_nodes,
        W,
        target_hint,
        bbox,
    )

    if source_attach is None or target_attach is None:
        return None, "could not attach connector corridor to both trail systems"

    source_trail, source_walk, source_offset = source_attach
    target_trail, target_walk, target_offset = target_attach

    try:
        distances, paths = nx.multi_source_dijkstra(
            W,
            [source_walk],
            weight="length",
        )
    except Exception as exc:
        return None, f"connector shortest-path search failed: {exc}"

    if target_walk not in distances:
        return None, "walkable corridor does not connect the two trail systems"

    path = paths[target_walk]
    path_length = float(distances[target_walk])

    # Reject wildly indirect connector paths. We only want practical bridges
    # between trail systems, not an accidental city-scale detour.
    max_reasonable = max(1800.0, straight_gap * 3.5 + 1200.0)
    if path_length > max_reasonable:
        return None, "walkable connector path is too indirect"

    return {
        "walk_graph": W,
        "path": path,
        "path_length_m": path_length,
        "source_trail": source_trail,
        "source_walk": source_walk,
        "source_offset_m": source_offset,
        "target_trail": target_trail,
        "target_walk": target_walk,
        "target_offset_m": target_offset,
        "straight_gap_m": straight_gap,
    }, None


def _next_unique_edge_key(G, u, v, base):
    key = str(base)
    if not G.has_edge(u, v, key):
        return key

    i = 1
    while G.has_edge(u, v, f"{key}_{i}"):
        i += 1
    return f"{key}_{i}"


def _add_attachment_edge(G, trail_node, walk_node, offset_m, label):
    if trail_node == walk_node or offset_m <= 0.01:
        return 0.0

    lon1 = float(G.nodes[trail_node]["x"])
    lat1 = float(G.nodes[trail_node]["y"])
    lon2 = float(G.nodes[walk_node]["x"])
    lat2 = float(G.nodes[walk_node]["y"])
    length = haversine_meters(lat1, lon1, lat2, lon2)

    attrs_forward = {
        "geometry": LineString([(lon1, lat1), (lon2, lat2)]),
        "length": float(length),
        "highway": "connector_attachment",
        "route_class": "connector",
        "routing_cost": float(length) * CONNECTOR_PATH_COST_MULTIPLIER,
        "synthetic_attachment": True,
        "connector_label": label,
    }
    attrs_reverse = dict(attrs_forward)
    attrs_reverse["geometry"] = LineString([(lon2, lat2), (lon1, lat1)])

    k1 = _next_unique_edge_key(G, trail_node, walk_node, f"v9attach_{label}")
    k2 = _next_unique_edge_key(G, walk_node, trail_node, f"v9attach_{label}")
    G.add_edge(trail_node, walk_node, key=k1, **attrs_forward)
    G.add_edge(walk_node, trail_node, key=k2, **attrs_reverse)
    return float(length)


def _merge_connector_path(local_G, connector_result, connector_index):
    W = connector_result["walk_graph"]
    path = connector_result["path"]

    for n in path:
        if n not in local_G:
            local_G.add_node(n, **dict(W.nodes[n]))

    copied_path_length = 0.0

    for i in range(len(path) - 1):
        a = path[i]
        b = path[i + 1]

        forward = W.get_edge_data(a, b) or {}
        reverse = W.get_edge_data(b, a) or {}

        # Copy every directed edge for this selected physical connector segment.
        for x, y, edge_dict in ((a, b, forward), (b, a, reverse)):
            for key, data in edge_dict.items():
                attrs = dict(data)
                attrs["route_class"] = "connector"
                attrs["routing_cost"] = (
                    float(attrs.get("length", 0) or 0)
                    * CONNECTOR_PATH_COST_MULTIPLIER
                )
                use_key = _next_unique_edge_key(
                    local_G,
                    x,
                    y,
                    f"v9c{connector_index}_{key}",
                )
                local_G.add_edge(x, y, key=use_key, **attrs)

        best = get_shortest_edge(W, a, b)
        if best is not None:
            copied_path_length += float(best.get("length", 0) or 0)

    # If OSM represents the trail/road join with separate nodes, bridge only a
    # very small topology gap. These short edges are explicitly tagged so their
    # mileage is penalized like every other connector.
    copied_path_length += _add_attachment_edge(
        local_G,
        connector_result["source_trail"],
        connector_result["source_walk"],
        connector_result["source_offset_m"],
        f"{connector_index}_a",
    )
    copied_path_length += _add_attachment_edge(
        local_G,
        connector_result["target_trail"],
        connector_result["target_walk"],
        connector_result["target_offset_m"],
        f"{connector_index}_b",
    )

    return copied_path_length


def add_selective_connectors(local_G, start_lat, start_lon, radius_meters):
    """
    Add a small number of real walkable OSM connector paths between useful
    disconnected trail components.

    This function is intentionally conservative:
      * skipped entirely for routes shorter than 8 miles
      * at most four connector corridors
      * at most ~7 km straight-line component gap
      * each corridor is downloaded and immediately reduced to one shortest path
      * all unused streets are discarded before DEM annotation / route search
    """
    stats = {
        "components_before": 0,
        "components_after": 0,
        "connectors_added": 0,
        "connector_queries": 0,
        "connector_path_meters": 0.0,
        "attempted": False,
        "errors": [],
    }

    G = local_G.copy()
    undirected = G.to_undirected(as_view=True)
    components = [set(c) for c in nx.connected_components(undirected)]
    stats["components_before"] = len(components)

    if len(components) <= 1:
        stats["components_after"] = len(components)
        G.graph["selective_connector_stats"] = stats
        return G, stats

    if float(radius_meters) < SELECTIVE_CONNECTORS_MIN_RADIUS_M:
        stats["components_after"] = len(components)
        G.graph["selective_connector_stats"] = stats
        return G, stats

    stats["attempted"] = True

    try:
        trail_graph = trail_only_graph(G)
        start_node = int(
            ox.distance.nearest_nodes(
                trail_graph,
                X=float(start_lon),
                Y=float(start_lat),
            )
        )
    except Exception as exc:
        stats["errors"].append(f"could not locate start trail component: {exc}")
        stats["components_after"] = len(components)
        G.graph["selective_connector_stats"] = stats
        return G, stats

    original_components = components
    component_lengths = {
        i: _component_unique_trail_length(G, nodes)
        for i, nodes in enumerate(original_components)
    }

    start_component_index = next(
        (i for i, nodes in enumerate(original_components) if start_node in nodes),
        None,
    )

    if start_component_index is None:
        stats["components_after"] = len(components)
        G.graph["selective_connector_stats"] = stats
        return G, stats

    failed_components = set()
    max_gap = min(
        SELECTIVE_CONNECTOR_MAX_GAP_M,
        max(1500.0, float(radius_meters) * 0.32),
    )
    max_useful_radial = min(
        float(radius_meters) * 0.55,
        float(radius_meters),
    )

    for connector_index in range(1, SELECTIVE_CONNECTOR_MAX_COUNT + 1):
        reachable = set(
            nx.node_connected_component(
                G.to_undirected(as_view=True),
                start_node,
            )
        )

        candidate_rows = []

        for i, target_nodes in enumerate(original_components):
            if i == start_component_index:
                continue
            if i in failed_components:
                continue
            if reachable.intersection(target_nodes):
                continue
            if component_lengths.get(i, 0.0) < SELECTIVE_CONNECTOR_MIN_COMPONENT_TRAIL_M:
                continue

            radial_pair = _closest_node_pair_same_graph(
                G,
                [start_node],
                target_nodes,
            )
            if radial_pair is None or radial_pair[2] > max_useful_radial:
                continue

            gap_pair = _closest_node_pair_same_graph(
                G,
                reachable,
                target_nodes,
            )
            if gap_pair is None:
                continue

            source_hint, target_hint, gap_m = gap_pair
            if gap_m > max_gap:
                continue

            trail_length = component_lengths.get(i, 0.0)
            # Prefer nearby components, with a modest bonus for substantial
            # trail systems so a 100 m fragment does not beat a preserve.
            score = gap_m / (1.0 + min(trail_length, 8000.0) / 8000.0 * 0.45)
            candidate_rows.append(
                (score, gap_m, -trail_length, i, source_hint, target_hint)
            )

        if not candidate_rows:
            break

        candidate_rows.sort()
        connected_this_round = False

        # Try several candidates; one failed Overpass corridor should not abort
        # the request or kill the server response.
        for _, gap_m, _, component_index, source_hint, target_hint in candidate_rows[:6]:
            stats["connector_queries"] += 1
            result, error = _download_connector_corridor(
                G,
                reachable,
                original_components[component_index],
                source_hint,
                target_hint,
            )

            if result is None:
                failed_components.add(component_index)
                if error:
                    stats["errors"].append(
                        f"component {component_index}: {error}"
                    )
                continue

            copied_length = _merge_connector_path(
                G,
                result,
                connector_index,
            )
            stats["connectors_added"] += 1
            stats["connector_path_meters"] += float(copied_length)
            connected_this_round = True
            break

        if not connected_this_round:
            break

    stats["components_after"] = nx.number_connected_components(
        G.to_undirected(as_view=True)
    )
    stats["connector_path_meters"] = round(
        stats["connector_path_meters"],
        1,
    )
    G.graph["selective_connector_stats"] = stats
    return G, stats



def add_reachable_dem_edge_elevations(G, start_lat, start_lon):
    """
    V11 fast path.

    Natural trail edges already contain ascent/descent from master_trails.pkl,
    so ordinary requests do not resample every reachable trail edge from the
    TIFF. Selective connector edges keep zero heuristic ascent here; final route
    scoring still samples the complete route geometry against the TIFF, so the
    authoritative distance/gain result remains unchanged.
    """
    trail_graph = trail_only_graph(G)
    if not trail_graph.number_of_edges():
        raise HTTPException(
            status_code=400,
            detail="No natural trails were found near this start coordinate.",
        )

    start_node = int(
        ox.distance.nearest_nodes(
            trail_graph,
            X=float(start_lon),
            Y=float(start_lat),
        )
    )
    reachable = set(
        nx.node_connected_component(
            G.to_undirected(as_view=True),
            start_node,
        )
    )

    H = G.copy()
    routeable_edges = 0

    for u, v, key, data in H.edges(keys=True, data=True):
        is_reachable = u in reachable and v in reachable
        if is_reachable:
            routeable_edges += 1

        if str(data.get("route_class", "trail")) == "trail":
            # Loaded from the offline master. Validation guarantees these exist.
            data["ascent_m"] = float(data.get("ascent_m", 0) or 0)
            data["descent_m"] = float(data.get("descent_m", 0) or 0)
            data["elevation_sample_count"] = int(
                float(data.get("elevation_sample_count", 0) or 0)
            )
        else:
            # Same heuristic treatment connectors had in v10. The final full
            # route DEM pass captures their actual climbing/descending.
            data["ascent_m"] = float(data.get("ascent_m", 0) or 0)
            data["descent_m"] = float(data.get("descent_m", 0) or 0)
            data["elevation_sample_count"] = int(
                float(data.get("elevation_sample_count", 0) or 0)
            )

        data["routing_cost"] = float(edge_routing_cost(data))

    H.graph["routeable_component_nodes"] = len(reachable)
    H.graph["routeable_component_edges"] = routeable_edges
    H.graph["offline_master_elevation_used"] = True
    return H, 0


def workspace_cache_key(lat, lon):
    return (
        round(float(lat), 5),
        round(float(lon), 5),
        os.path.basename(DEM_PATH),
        "start-workspace-v12",
    )


def workspace_max_radius_meters(lat, lon):
    """Radius from the requested start that fully covers the TIFF footprint."""
    left, bottom, right, top = get_dem_bounds_wgs84()
    corners = [
        (bottom, left),
        (bottom, right),
        (top, left),
        (top, right),
    ]
    farthest = max(
        haversine_meters(float(lat), float(lon), float(clat), float(clon))
        for clat, clon in corners
    )
    return float(farthest + 150.0)


def physical_trail_segment_count(G):
    seen = set()
    for u, v, key, data in G.edges(keys=True, data=True):
        if str(data.get("route_class", "trail")) != "trail":
            continue
        pk = (
            min(int(u), int(v)),
            max(int(u), int(v)),
            round(float(data.get("length", 0) or 0), 1),
        )
        seen.add(pk)
    return len(seen)


def finalize_workspace_graph(G, start_node, start_lat, start_lon):
    """
    Prepare routing metadata once for a start-specific workspace.

    Natural-trail elevation heuristics are already baked into the offline
    master. Connector heuristics remain cheap here; authoritative route gain is
    still measured from the full 5 m DEM profile for finalists.
    """
    for u, v, key, data in G.edges(keys=True, data=True):
        data["ascent_m"] = float(data.get("ascent_m", 0) or 0)
        data["descent_m"] = float(data.get("descent_m", 0) or 0)
        data["elevation_sample_count"] = int(
            float(data.get("elevation_sample_count", 0) or 0)
        )
        data["routing_cost"] = float(edge_routing_cost(data))

    radial = {}
    for node, data in G.nodes(data=True):
        try:
            radial[node] = haversine_meters(
                float(start_lat),
                float(start_lon),
                float(data["y"]),
                float(data["x"]),
            )
        except Exception:
            radial[node] = float("inf")

    radial[start_node] = 0.0

    try:
        reachable = set(
            nx.node_connected_component(
                G.to_undirected(as_view=True),
                start_node,
            )
        )
    except Exception:
        reachable = {start_node}

    routeable_edges = sum(
        1
        for u, v in G.edges()
        if u in reachable and v in reachable
    )

    G.graph["workspace_node_radial_m"] = radial
    G.graph["routeable_component_nodes"] = len(reachable)
    G.graph["routeable_component_edges"] = routeable_edges
    G.graph["offline_master_elevation_used"] = True
    return G


def get_start_workspace(lat, lon, force_rebuild=False):
    """
    Build/load the expensive routing workspace keyed ONLY by start coordinate.

    Distance and elevation are intentionally absent from the key. The workspace
    contains the entire offline TIFF trail graph plus the small selective
    connector set, so target changes never trigger OSM/connector preparation.
    """
    if not point_inside_dem(lat, lon):
        left, bottom, right, top = get_dem_bounds_wgs84()
        raise HTTPException(
            status_code=400,
            detail=(
                "Start coordinate is outside TIFF coverage: "
                f"west={left:.6f}, east={right:.6f}, "
                f"south={bottom:.6f}, north={top:.6f}."
            ),
        )

    key = workspace_cache_key(lat, lon)

    if not force_rebuild and key in WORKSPACE_CACHE:
        return WORKSPACE_CACHE[key], True

    with WORKSPACE_CACHE_LOCK:
        if not force_rebuild and key in WORKSPACE_CACHE:
            return WORKSPACE_CACHE[key], True

        started = time.perf_counter()
        master_G, master_info = get_master_trail_graph()

        max_radius_m = workspace_max_radius_meters(lat, lon)

        # add_selective_connectors only considers target systems out to ~55% of
        # its radius. Doubling the TIFF-covering radius makes every trail system
        # in the TIFF eligible during this one start-specific preparation pass.
        connector_radius_m = max(
            SELECTIVE_CONNECTORS_MIN_RADIUS_M,
            max_radius_m * 2.0,
        )

        G, connector_stats = add_selective_connectors(
            master_G,
            float(lat),
            float(lon),
            connector_radius_m,
        )

        # Insert/split the exact requested start only once per workspace.
        G, start_node, start_info = insert_exact_routing_point(
            G,
            float(lat),
            float(lon),
        )

        G = finalize_workspace_graph(
            G,
            start_node,
            float(lat),
            float(lon),
        )

        build_seconds = time.perf_counter() - started

        G.graph["selective_connector_stats"] = connector_stats
        G.graph["workspace_start_node"] = start_node
        G.graph["workspace_start_info"] = dict(start_info)
        G.graph["workspace_start_lat"] = float(lat)
        G.graph["workspace_start_lon"] = float(lon)
        G.graph["workspace_max_radius_m"] = float(max_radius_m)
        G.graph["workspace_connector_radius_m"] = float(connector_radius_m)
        G.graph["workspace_build_seconds"] = float(build_seconds)
        G.graph["workspace_master_file"] = master_info.get(
            "saved_graph",
            os.path.basename(MASTER_GRAPH_PATH),
        )

        workspace = {
            "graph": G,
            "start_node": start_node,
            "start_info": dict(start_info),
            "max_radius_m": float(max_radius_m),
            "connector_radius_m": float(connector_radius_m),
            "build_seconds": float(build_seconds),
            "filtered_edges_removed": master_info["filtered_edges_removed"],
            "master_info": master_info,
        }

        if force_rebuild:
            WORKSPACE_CACHE.pop(key, None)

        while len(WORKSPACE_CACHE) >= MAX_CACHED_WORKSPACES:
            oldest = next(iter(WORKSPACE_CACHE))
            WORKSPACE_CACHE.pop(oldest, None)

        WORKSPACE_CACHE[key] = workspace
        return workspace, False


def extract_route_graph_from_workspace(workspace, radius_meters):
    """
    Cheap target-specific graph extraction from an already prepared workspace.

    This is a true radial filter around the requested start. No OSM request,
    connector download, master-graph parsing, DEM edge annotation, or gray-map
    serialization happens here.
    """
    G = workspace["graph"]
    start_node = workspace["start_node"]
    requested_radius = max(100.0, float(radius_meters))
    max_radius = float(workspace["max_radius_m"])

    if requested_radius >= max_radius:
        H = G
    else:
        radial = G.graph.get("workspace_node_radial_m", {})
        # Small buffer keeps edges whose endpoints sit just outside the nominal
        # radius from being needlessly severed by coordinate rounding.
        cutoff = requested_radius + 35.0
        nodes = [
            node
            for node, distance in radial.items()
            if float(distance) <= cutoff
        ]
        if start_node not in nodes:
            nodes.append(start_node)
        H = G.subgraph(nodes).copy()

    if start_node not in H or H.number_of_edges() == 0:
        raise HTTPException(
            status_code=400,
            detail="No routeable trail network exists inside this distance radius.",
        )

    H.graph["workspace_start_node"] = start_node
    H.graph["workspace_start_info"] = dict(workspace["start_info"])
    H.graph["workspace_start_lat"] = float(G.graph.get("workspace_start_lat"))
    H.graph["workspace_start_lon"] = float(G.graph.get("workspace_start_lon"))
    H.graph["workspace_max_radius_m"] = max_radius
    H.graph["search_radius_m"] = requested_radius
    H.graph["workspace_build_seconds"] = float(workspace["build_seconds"])
    H.graph["workspace_connector_radius_m"] = float(workspace["connector_radius_m"])
    H.graph["selective_connector_stats"] = G.graph.get(
        "selective_connector_stats",
        {},
    )

    try:
        reachable = set(
            nx.node_connected_component(
                H.to_undirected(as_view=True),
                start_node,
            )
        )
        H.graph["routeable_component_nodes"] = len(reachable)
        H.graph["routeable_component_edges"] = sum(
            1 for u, v in H.edges() if u in reachable and v in reachable
        )
    except Exception:
        H.graph["routeable_component_nodes"] = 1
        H.graph["routeable_component_edges"] = 0

    return H


def download_trail_graph(lat, lon, radius_meters):
    """
    V12 compatibility wrapper used by route/GPX endpoints.

    The expensive workspace is keyed only by start. Target distance changes
    merely slice that in-memory workspace by radius.
    """
    workspace, workspace_from_cache = get_start_workspace(
        float(lat),
        float(lon),
        force_rebuild=False,
    )

    G = extract_route_graph_from_workspace(
        workspace,
        float(radius_meters),
    )

    G.graph["workspace_from_cache"] = bool(workspace_from_cache)
    return (
        G,
        int(workspace["filtered_edges_removed"]),
        bool(workspace_from_cache),
        0,
    )


def get_master_trail_overlay_json():
    """Serialize the TIFF-wide gray trail overlay exactly once per process."""
    global MASTER_TRAIL_OVERLAY_JSON

    if MASTER_TRAIL_OVERLAY_JSON is not None:
        return MASTER_TRAIL_OVERLAY_JSON

    with MASTER_TRAIL_OVERLAY_LOCK:
        if MASTER_TRAIL_OVERLAY_JSON is not None:
            return MASTER_TRAIL_OVERLAY_JSON

        master_G, master_info = get_master_trail_graph()
        segments = graph_debug_segments(master_G)
        payload = {
            "allowed_trails": segments,
            "allowed_trail_segments": len(segments),
            "master_network_nodes": master_info["nodes"],
            "master_network_edges": master_info["edges"],
            "master_graph_file": master_info.get(
                "saved_graph",
                os.path.basename(MASTER_GRAPH_PATH),
            ),
            "master_tiff": os.path.basename(DEM_PATH),
            "version": APP_VERSION,
        }
        MASTER_TRAIL_OVERLAY_JSON = json.dumps(
            payload,
            separators=(",", ":"),
        )
        return MASTER_TRAIL_OVERLAY_JSON

# ============================================================
# SIMPLE ROUTING GRAPH
# ============================================================

def make_simple_routing_graph(G):
    S = nx.DiGraph()
    S.add_nodes_from(G.nodes(data=True))
    for u, v, data in G.edges(data=True):
        length = float(data.get("length", 0) or 0)
        if length <= 0:
            continue
        ascent = float(data.get("ascent_m", 0) or 0)
        route_class = str(data.get("route_class", "trail"))
        routing_cost = float(data.get("routing_cost", edge_routing_cost(data)))
        if not S.has_edge(u, v) or routing_cost < float(S[u][v].get("routing_cost", float("inf"))):
            S.add_edge(u, v, length=length, ascent_m=ascent, route_class=route_class, routing_cost=routing_cost)
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


def connector_distance_meters(G, route_nodes):
    total = 0.0
    for i in range(len(route_nodes) - 1):
        edge = get_shortest_edge(G, route_nodes[i], route_nodes[i + 1])
        if edge is not None and str(edge.get("route_class", "trail")) == "connector":
            total += float(edge.get("length", 0) or 0)
    return total


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



def route_topology_metrics(route_nodes):
    """
    Describe the topology created by the physical edges actually used by a route.

    A clean single loop has cycle_rank == 1 and normally no branch points.
    Figure-eights / compound loops create additional independent cycles and/or
    branch points. Repeated physical edges are handled separately by the normal
    repetition penalty.
    """
    unique_edges = set()
    degrees = {}

    for i in range(len(route_nodes) - 1):
        u = route_nodes[i]
        v = route_nodes[i + 1]
        key = undirected_edge_key(u, v)
        if key in unique_edges:
            continue
        unique_edges.add(key)
        degrees[u] = degrees.get(u, 0) + 1
        degrees[v] = degrees.get(v, 0) + 1

    if not unique_edges:
        return {
            "cycle_rank": 0,
            "extra_cycles": 0,
            "branch_points": 0,
            "branch_excess": 0,
            "route_topology_nodes": 0,
            "route_topology_edges": 0,
        }

    # The used-edge graph comes from one continuous walk, so it is connected.
    # Cyclomatic number E - V + 1 therefore equals the number of independent
    # cycles. A clean simple loop is exactly one cycle.
    edge_count = len(unique_edges)
    node_count = len(degrees)
    cycle_rank = max(0, edge_count - node_count + 1)
    branch_points = sum(1 for degree in degrees.values() if degree > 2)
    branch_excess = sum(max(0, degree - 2) for degree in degrees.values())

    return {
        "cycle_rank": int(cycle_rank),
        "extra_cycles": int(max(0, cycle_rank - 1)),
        "branch_points": int(branch_points),
        "branch_excess": int(branch_excess),
        "route_topology_nodes": int(node_count),
        "route_topology_edges": int(edge_count),
    }


def route_max_radial_meters_from_nodes(G, route_nodes):
    if not route_nodes:
        return 0.0

    start = route_nodes[0]
    if start not in G:
        return 0.0

    start_lat = float(G.nodes[start]["y"])
    start_lon = float(G.nodes[start]["x"])
    farthest = 0.0

    for node in set(route_nodes):
        if node not in G:
            continue
        radial = haversine_meters(
            start_lat,
            start_lon,
            float(G.nodes[node]["y"]),
            float(G.nodes[node]["x"]),
        )
        farthest = max(farthest, radial)

    return float(farthest)


def route_max_radial_meters_from_coords(coords):
    if not coords:
        return 0.0

    start_lat = float(coords[0]["lat"])
    start_lon = float(coords[0]["lon"])
    farthest = 0.0

    for point in coords:
        radial = haversine_meters(
            start_lat,
            start_lon,
            float(point["lat"]),
            float(point["lon"]),
        )
        farthest = max(farthest, radial)

    return float(farthest)


def route_convex_hull_area_m2(coords):
    """Approximate route footprint in local meters using a convex hull."""
    if len(coords) < 3:
        return 0.0

    lat0 = float(coords[0]["lat"])
    lon0 = float(coords[0]["lon"])
    cos_lat = max(0.01, math.cos(math.radians(lat0)))

    points_xy = []
    for point in coords:
        lat = float(point["lat"])
        lon = float(point["lon"])
        x = (lon - lon0) * 111320.0 * cos_lat
        y = (lat - lat0) * 110540.0
        points_xy.append((x, y))

    try:
        return float(MultiPoint(points_xy).convex_hull.area)
    except Exception:
        return 0.0


def big_loop_shape_penalty(
    target_distance_meters,
    topology,
    max_radial_meters,
    footprint_area_m2=None,
    cheap=False,
):
    """
    Soft preference for one geographically large loop.

    It never makes compound loops illegal. It simply makes a clean, broad loop
    win when distance/elevation quality is otherwise comparable.
    """
    target_miles = target_distance_meters / METERS_PER_MILE
    if target_miles < 4.0:
        return 0.0, {
            "max_radial_ratio": 0.0,
            "footprint_ratio": 0.0,
            "shape_penalty": 0.0,
        }

    radial_ratio = max_radial_meters / max(target_distance_meters, 1.0)
    extra_cycles = int(topology.get("extra_cycles", 0))
    branch_excess = int(topology.get("branch_excess", 0))

    if target_miles < 8.0:
        target_radial_ratio = 0.27
        cycle_weight = 12.0 if cheap else 18.0
        branch_weight = 2.5 if cheap else 4.0
        spread_weight = 14.0 if cheap else 22.0
        footprint_target = 0.016
        footprint_weight = 0.0 if cheap else 10.0
    elif target_miles < 15.0:
        target_radial_ratio = 0.30
        cycle_weight = 28.0 if cheap else 40.0
        branch_weight = 5.0 if cheap else 7.0
        spread_weight = 30.0 if cheap else 48.0
        footprint_target = 0.020
        footprint_weight = 0.0 if cheap else 22.0
    else:
        target_radial_ratio = 0.31
        cycle_weight = 34.0 if cheap else 48.0
        branch_weight = 6.0 if cheap else 8.0
        spread_weight = 36.0 if cheap else 58.0
        footprint_target = 0.022
        footprint_weight = 0.0 if cheap else 28.0

    spread_shortfall = max(0.0, target_radial_ratio - radial_ratio) / max(
        target_radial_ratio,
        0.001,
    )

    footprint_ratio = 0.0
    footprint_shortfall = 0.0
    if footprint_area_m2 is not None:
        footprint_ratio = float(footprint_area_m2) / max(
            target_distance_meters * target_distance_meters,
            1.0,
        )
        footprint_shortfall = max(0.0, footprint_target - footprint_ratio) / max(
            footprint_target,
            0.001,
        )

    penalty = (
        extra_cycles * cycle_weight
        + branch_excess * branch_weight
        + spread_shortfall * spread_weight
        + footprint_shortfall * footprint_weight
    )

    return float(penalty), {
        "max_radial_ratio": float(radial_ratio),
        "footprint_ratio": float(footprint_ratio),
        "shape_penalty": float(penalty),
    }


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
    connector_distance = connector_distance_meters(G, route_nodes)
    connector_ratio = connector_distance / max(total_distance, 1.0)

    topology = route_topology_metrics(route_nodes)
    max_radial_meters = route_max_radial_meters_from_coords(coords)
    footprint_area_m2 = route_convex_hull_area_m2(coords)
    shape_penalty, shape_metrics = big_loop_shape_penalty(
        target_distance_meters,
        topology,
        max_radial_meters,
        footprint_area_m2=footprint_area_m2,
        cheap=False,
    )

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
        + connector_ratio * CONNECTOR_FINAL_SCORE_WEIGHT
        + shape_penalty
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
            "connector_distance_meters": connector_distance,
            "connector_ratio": connector_ratio,
            "trail_fraction": max(0.0, 1.0 - connector_ratio),
            "cycle_rank": topology["cycle_rank"],
            "extra_cycles": topology["extra_cycles"],
            "branch_points": topology["branch_points"],
            "branch_excess": topology["branch_excess"],
            "max_radial_meters": max_radial_meters,
            "max_radial_ratio": shape_metrics["max_radial_ratio"],
            "footprint_area_m2": footprint_area_m2,
            "footprint_ratio": shape_metrics["footprint_ratio"],
            "shape_penalty": shape_metrics["shape_penalty"],
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


def route_unique_edge_lengths(G, route_nodes):
    """Return unique physical trail/connector edge lengths used by a route."""
    result = {}
    for i in range(len(route_nodes) - 1):
        u = route_nodes[i]
        v = route_nodes[i + 1]
        data = get_shortest_edge(G, u, v)
        if data is None:
            continue
        key = undirected_edge_key(u, v)
        length = float(data.get("length", 0) or 0)
        if length > 0:
            result[key] = max(result.get(key, 0.0), length)
    return result


def route_shared_fraction(G, route_a, route_b):
    """
    Fraction of the smaller route's unique physical-edge distance shared with
    the other route. A value of 1 means essentially the same physical route.
    """
    a = route_unique_edge_lengths(G, route_a)
    b = route_unique_edge_lengths(G, route_b)
    if not a or not b:
        return 1.0

    total_a = sum(a.values())
    total_b = sum(b.values())
    denominator = min(total_a, total_b)
    if denominator <= 0:
        return 1.0

    shared = 0.0
    for key in set(a).intersection(b):
        shared += min(a[key], b[key])
    return shared / denominator


def select_diverse_accurate_candidates(
    G,
    scored_candidates,
    max_routes=MAX_ROUTE_OPTIONS,
    max_shared_fraction=MAX_ROUTE_SHARED_FRACTION,
):
    """
    Pick the best accurately scored routes while rejecting near-duplicates.
    scored_candidates contains (score, route_nodes, metrics).
    """
    ordered = sorted(scored_candidates, key=lambda item: item[0])
    selected = []

    for candidate in ordered:
        _, route_nodes, _ = candidate
        if any(
            route_shared_fraction(G, route_nodes, existing[1]) > max_shared_fraction
            for existing in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) >= max_routes:
            break

    return selected


def copy_internal_route_options(candidates):
    """Make non-circular internal copies safe to attach to winning metrics."""
    packaged = []
    for score, route_nodes, metrics in candidates:
        clean_metrics = {
            key: value
            for key, value in metrics.items()
            if key != "_route_options_candidates"
        }
        packaged.append({
            "score": float(score),
            "route_nodes": list(route_nodes),
            "metrics": clean_metrics,
        })
    return packaged


def build_route_option_payload(
    G,
    route_nodes,
    metrics,
    request,
    option_index,
):
    coords = metrics.get("route_coordinates") or route_coordinates(G, route_nodes)
    route_distance_miles = metrics["total_distance_meters"] / METERS_PER_MILE
    actual_gain_ft = metrics["actual_gain_meters"] * FEET_PER_METER
    actual_descent_ft = metrics.get("actual_descent_meters", 0.0) * FEET_PER_METER
    repeated_distance_miles = metrics["repeated_distance_meters"] / METERS_PER_MILE

    return {
        "index": int(option_index),
        "name": f"Route {int(option_index) + 1}",
        "actual_distance_miles": round(route_distance_miles, 2),
        "distance_error_miles": round(abs(route_distance_miles - request.target_distance_miles), 2),
        "actual_gain_ft": round(actual_gain_ft),
        "actual_descent_ft": round(actual_descent_ft),
        "elevation_error_ft": round(abs(actual_gain_ft - request.target_gain_ft)),
        "route": coords,
        "gpx_export_points": build_gpx_export_points(coords),
        "route_nodes": len(route_nodes),
        "route_geometry_points": len(coords),
        "repeated_edges": metrics["repeated_edges"],
        "repeated_distance_miles": round(repeated_distance_miles, 2),
        "repeated_nodes": metrics["repeated_nodes"],
        "immediate_reversals": metrics["immediate_reversals"],
        "connector_distance_miles": round(metrics.get("connector_distance_meters", 0.0) / METERS_PER_MILE, 2),
        "trail_percent": round(metrics.get("trail_fraction", 1.0) * 100.0, 1),
        "independent_loops": int(metrics.get("cycle_rank", 0)),
        "extra_subloops": int(metrics.get("extra_cycles", 0)),
        "branch_points": int(metrics.get("branch_points", 0)),
        "max_reach_miles": round(metrics.get("max_radial_meters", 0.0) / METERS_PER_MILE, 2),
        "footprint_sq_miles": round(metrics.get("footprint_area_m2", 0.0) / (METERS_PER_MILE ** 2), 2),
        "shape_penalty": round(metrics.get("shape_penalty", 0.0), 2),
        "route_score": round(metrics["score"], 2),
        "partial_edge_used": bool(metrics.get("partial_edge_used", False)),
        "partial_added_distance_miles": round(metrics.get("partial_added_distance_meters", 0.0) / METERS_PER_MILE, 3),
        "partial_outward_distance_meters": round(metrics.get("partial_outward_distance_meters", 0.0), 1),
    }


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
            if str(data.get("route_class", "trail")) != "trail":
                continue
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
        "connector_distance": 0.0,
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
        connector_ratio = state.get("connector_distance", 0.0) / max(state["distance"], 1.0)
        reversal_penalty = state["reversals"] * 0.02
        connector_penalty = connector_ratio * 1.5

        return {
            "balanced": distance_error * 3.0 + density_error * 0.85 + repeat_ratio * 0.25 + reversal_penalty + connector_penalty,
            "gain": density_error * 2.5 + distance_error * 1.0 + repeat_ratio * 0.20 + reversal_penalty + connector_penalty,
            "distance": distance_error * 5.0 + density_error * 0.15 + repeat_ratio * 0.15 + reversal_penalty + connector_penalty,
            "flat": gain_density * 8.0 + distance_error * 2.0 + repeat_ratio * 0.15 + reversal_penalty + connector_penalty,
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
                new_connector_distance = state.get("connector_distance", 0.0) + (
                    edge_length if str(edge.get("route_class", "trail")) == "connector" else 0.0
                )

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
                            connector_ratio = new_connector_distance / max(new_distance, 1.0)

                            closed_candidates.append({
                                "route": list(new_route),
                                "distance": new_distance,
                                "gain": new_gain,
                                "distance_ratio": distance_ratio,
                                "gain_ratio": gain_ratio,
                                "repeat_ratio": repeat_ratio,
                                "gain_density": gain_density,
                                "connector_ratio": connector_ratio,
                                "cheap_balanced": distance_ratio * 3.0 + gain_ratio * 1.2 + repeat_ratio * 0.25 + connector_ratio * 1.5,
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
                        "connector_distance": new_connector_distance,
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
                int(state.get("connector_distance", 0.0) / 50.0),
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
    acceptable_candidates = []
    for score, route_nodes, metrics in accurately_scored:
        distance_error_miles = metrics["distance_error_meters"] / METERS_PER_MILE
        gain_error_ft = metrics["gain_error_meters"] * FEET_PER_METER
        acceptable = (
            distance_error_miles <= limits["distance_error_limit_miles"]
            and gain_error_ft <= limits["gain_error_limit_ft"]
        )

        if best_any is None or score < best_any[0]:
            best_any = (score, route_nodes, metrics)
        if acceptable:
            acceptable_candidates.append((score, route_nodes, metrics))

    if acceptable_candidates:
        diverse = select_diverse_accurate_candidates(G, acceptable_candidates)
        if not diverse:
            diverse = [min(acceptable_candidates, key=lambda item: item[0])]
        _, route_nodes, metrics = diverse[0]
        metrics = dict(metrics)
        metrics["_route_options_candidates"] = copy_internal_route_options(diverse)
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
        cost = float(data.get("routing_cost", data.get("length", 1.0)))
        if undirected_edge_key(u, v) in used_edges:
            cost *= 40.0
        return cost

    def heuristic(a, b):
        try:
            return haversine_meters(
                float(S.nodes[a]["y"]),
                float(S.nodes[a]["x"]),
                float(S.nodes[b]["y"]),
                float(S.nodes[b]["x"]),
            )
        except Exception:
            return 0.0

    return nx.astar_path(
        S,
        source,
        target,
        heuristic=heuristic,
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
    connector_distance = connector_distance_meters(G, route_nodes)
    connector_ratio = connector_distance / max(total_distance, 1.0)

    topology = route_topology_metrics(route_nodes)
    max_radial_meters = route_max_radial_meters_from_nodes(G, route_nodes)
    shape_penalty, shape_metrics = big_loop_shape_penalty(
        target_distance_meters,
        topology,
        max_radial_meters,
        footprint_area_m2=None,
        cheap=True,
    )

    # Distance and approximate elevation are the primary exploratory goals.
    # Repetition remains a meaningful but secondary penalty. V13 also gives
    # clean, broad single-loop candidates a soft advantage so they survive
    # into the small set of expensive DEM finalists.
    score = (
        distance_ratio * 190.0
        + gain_ratio * 150.0
        + repeat_ratio * 170.0
        + repeated_nodes * 12.0
        + immediate_reversals * 10.0
        + connector_ratio * CONNECTOR_CHEAP_SCORE_WEIGHT
        + shape_penalty
    )

    return score, {
        "total_distance_meters": total_distance,
        "approximate_gain_meters": approximate_gain,
        "repeated_edges": repeated_edges,
        "repeated_distance_meters": repeated_distance,
        "repeated_nodes": repeated_nodes,
        "immediate_reversals": immediate_reversals,
        "connector_distance_meters": connector_distance,
        "connector_ratio": connector_ratio,
        "cycle_rank": topology["cycle_rank"],
        "extra_cycles": topology["extra_cycles"],
        "branch_points": topology["branch_points"],
        "branch_excess": topology["branch_excess"],
        "max_radial_meters": max_radial_meters,
        "max_radial_ratio": shape_metrics["max_radial_ratio"],
        "shape_penalty": shape_metrics["shape_penalty"],
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

    # For a closed route, ~50% of total distance is the practical maximum
    # straight-line reach because the route still has to return to the start.
    max_radial_distance = min(
        profile["search_radius_m"] * 0.95,
        target_distance_meters * 0.48,
    )

    # Waypoint anchors are chosen on natural trails, never on connector streets.
    # The gray overlay may contain disconnected trail systems. Only components
    # actually reachable from this start (including v9 selective connectors)
    # are eligible as route anchors.
    reachable_nodes = set(
        nx.node_connected_component(
            S.to_undirected(as_view=True),
            start_node,
        )
    )

    trail_nodes = set()
    for u, v, data in G.edges(data=True):
        if str(data.get("route_class", "trail")) == "trail":
            if u in reachable_nodes:
                trail_nodes.add(u)
            if v in reachable_nodes:
                trail_nodes.add(v)

    for node in trail_nodes:
        if node == start_node or node not in S:
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
        cycle_bucket = int(cheap_metrics.get("extra_cycles", 0))
        radial_bucket = int(cheap_metrics.get("max_radial_meters", 0.0) / 500.0)
        bucket = (distance_bucket, gain_bucket, repeat_bucket, cycle_bucket, radial_bucket)

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

    best_any_route = None
    best_any_metrics = None
    best_any_score = float("inf")
    acceptable_candidates = []

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
            acceptable_candidates.append((score, route, metrics))

    if acceptable_candidates:
        diverse = select_diverse_accurate_candidates(G, acceptable_candidates)
        if not diverse:
            diverse = [min(acceptable_candidates, key=lambda item: item[0])]
        _, best_route, best_metrics = diverse[0]
        best_metrics = dict(best_metrics)
        best_metrics["waypoint_accurate_finalists"] = accurately_scored
        best_metrics["_route_options_candidates"] = copy_internal_route_options(diverse)
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

        generator_G = G
        generator_start_node = G.graph.get("workspace_start_node")
        generator_start_info = G.graph.get("workspace_start_info") or {}
        if generator_start_node is None or generator_start_node not in generator_G:
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

@app.get("/trail-overlay")
def trail_overlay():
    """
    TIFF-wide natural-trail visualization, independent of route targets.

    The JSON string is created once and GZipMiddleware compresses it in transit.
    The browser fetches this endpoint once per page session instead of receiving
    gray trail geometry before every route search.
    """
    return Response(
        content=get_master_trail_overlay_json(),
        media_type="application/json",
        headers={
            "Cache-Control": "public, max-age=3600",
        },
    )


@app.post("/trail-network")
def trail_network(request: TrailNetworkRequest):
    try:
        profile = get_route_profile(request.target_distance_miles)

        workspace, workspace_from_cache = get_start_workspace(
            request.start_lat,
            request.start_lon,
            force_rebuild=bool(request.force_reload),
        )

        # This cheap slice is only for diagnostics. It does not rebuild the
        # workspace and does not serialize any gray-map geometry.
        G = extract_route_graph_from_workspace(
            workspace,
            profile["search_radius_m"],
        )

        start_node = workspace["start_node"]
        start_info = workspace["start_info"]
        workspace_G = workspace["graph"]

        snapped_lat = float(workspace_G.nodes[start_node]["y"])
        snapped_lon = float(workspace_G.nodes[start_node]["x"])
        snap_distance = float(start_info["routing_offset_m"])

        local_trail_edges, local_connector_edges = graph_route_class_counts(G)
        connector_stats = workspace_G.graph.get("selective_connector_stats", {}) or {}
        master_info = workspace["master_info"]

        return {
            # Gray geometry intentionally lives at /trail-overlay now.
            "allowed_trail_segments": physical_trail_segment_count(G),
            "network_nodes": G.number_of_nodes(),
            "network_edges": G.number_of_edges(),
            "workspace_nodes": workspace_G.number_of_nodes(),
            "workspace_edges": workspace_G.number_of_edges(),
            "workspace_max_radius_m": round(float(workspace["max_radius_m"]), 1),
            "workspace_connector_radius_m": round(float(workspace["connector_radius_m"]), 1),
            "workspace_build_seconds": round(float(workspace["build_seconds"]), 3),
            "workspace_from_cache": bool(workspace_from_cache),
            "master_network_nodes": master_info["nodes"],
            "master_network_edges": master_info["edges"],
            "master_physical_segments": master_info["physical_segments"],
            "master_trail_segments": master_info.get("trail_physical_segments", 0),
            "master_connector_segments": master_info.get("connector_physical_segments", 0),
            "local_trail_directed_edges": local_trail_edges,
            "local_connector_directed_edges": local_connector_edges,
            "selective_connectors_attempted": bool(connector_stats.get("attempted", False)),
            "selective_connectors_added": int(connector_stats.get("connectors_added", 0) or 0),
            "selective_connector_queries": int(connector_stats.get("connector_queries", 0) or 0),
            "selective_connector_path_meters": float(connector_stats.get("connector_path_meters", 0.0) or 0.0),
            "trail_components_before": int(connector_stats.get("components_before", 0) or 0),
            "trail_components_after": int(connector_stats.get("components_after", 0) or 0),
            "routeable_component_nodes": int(float(G.graph.get("routeable_component_nodes", 0) or 0)),
            "routeable_component_edges": int(float(G.graph.get("routeable_component_edges", 0) or 0)),
            "master_loaded_from_disk": master_info["loaded_from_disk"],
            "master_elevation_precomputed": master_info.get("elevation_precomputed", False),
            "master_graph_file": master_info.get("saved_graph", os.path.basename(MASTER_GRAPH_PATH)),
            "request_edge_dem_samples": 0,
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
            "filtered_edges_removed": int(workspace["filtered_edges_removed"]),
            "graph_from_cache": bool(workspace_from_cache),
            "elevation_samples": 0,
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

        start_node = G.graph.get("workspace_start_node")
        start_info = G.graph.get("workspace_start_info") or {}
        if start_node is None or start_node not in G:
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

        internal_options = metrics.get("_route_options_candidates") or [
            {
                "score": float(metrics.get("score", 0.0)),
                "route_nodes": list(route_nodes),
                "metrics": {
                    key: value
                    for key, value in metrics.items()
                    if key != "_route_options_candidates"
                },
            }
        ]

        route_options = []
        for option_index, option in enumerate(internal_options[:MAX_ROUTE_OPTIONS]):
            route_options.append(
                build_route_option_payload(
                    G,
                    option["route_nodes"],
                    option["metrics"],
                    request,
                    option_index,
                )
            )

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
            "route_options": route_options,
            "route_options_count": len(route_options),
            "route_option_max_shared_fraction": MAX_ROUTE_SHARED_FRACTION,
            "route_nodes": len(route_nodes),
            "route_geometry_points": len(coords),
            "repeated_edges": metrics["repeated_edges"],
            "repeated_distance_miles": round(
                repeated_distance_miles,
                2,
            ),
            "repeated_nodes": metrics["repeated_nodes"],
            "immediate_reversals": metrics["immediate_reversals"],
            "connector_distance_miles": round(metrics.get("connector_distance_meters", 0.0) / METERS_PER_MILE, 2),
            "trail_percent": round(metrics.get("trail_fraction", 1.0) * 100.0, 1),
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

.route-choice-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 10px 0 12px 0;
}

.route-choice {
    background: #f3f4f6;
    color: #111;
    border: 1px solid #bbb;
    margin: 0;
    padding: 8px 10px;
    font-size: 13px;
}

.route-choice.selected {
    background: #b91c1c;
    color: white;
    border-color: #991b1b;
}

#selectedRouteDetails {
    margin-top: 8px;
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
<div style="font-size:12px;color:#666;margin-bottom:10px;">Version: 2026-08-09-v12-start-workspace-overlay-cache</div>

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
<button id="networkButton">Load / Refresh Start Area</button>

<div class="network-control">
    <label>
        <input id="showNetwork" type="checkbox" checked>
        Show TIFF-wide trail network
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
const map = L.map("map", {preferCanvas: true}).setView(
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
let routeOptionLines = [];
let selectedRouteOptionIndex = 0;
let gpxLine = null;
let networkLayer = L.layerGroup();
let lastGeneratedRoute = null;
let requestedStartMarker = null;
let snappedStartMarker = null;
let snapLine = null;
let loadedWorkspaceStartKey = null;
let lastWorkspaceResult = null;
let masterTrailOverlayLoaded = false;
let masterTrailOverlayPromise = null;

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


function getSelectedRouteOption() {
    if (!lastGeneratedRoute) {
        return null;
    }
    const options = lastGeneratedRoute.route_options || [];
    if (options.length > 0) {
        return options[Math.max(0, Math.min(selectedRouteOptionIndex, options.length - 1))];
    }
    return lastGeneratedRoute;
}


function downloadGeneratedGpx() {
    const selected = getSelectedRouteOption();
    if (!selected) {
        return;
    }

    const points = selected.gpx_export_points;

    if (!points || points.length < 2) {
        alert("The selected route does not contain enough points to export.");
        return;
    }

    const distance = Number(selected.actual_distance_miles).toFixed(2);
    const gain = Math.round(Number(selected.actual_gain_ft));
    const routeNumber = Number(selected.index ?? selectedRouteOptionIndex) + 1;
    const routeName = `Trail Route ${routeNumber} - ${distance} mi - COROS`;
    const filename = `trail-route-${routeNumber}-${distance}mi-${gain}ft-coros.gpx`;
    const xml = buildGpxXml(points, routeName);

    triggerTextDownload(filename, xml, "application/gpx+xml;charset=utf-8");
}


function clearGeneratedRouteLines() {
    for (const line of routeOptionLines) {
        if (map.hasLayer(line)) {
            map.removeLayer(line);
        }
    }
    routeOptionLines = [];

    if (routeLine && map.hasLayer(routeLine)) {
        map.removeLayer(routeLine);
    }
    routeLine = null;
}


function renderSelectedRouteDetails(option) {
    const details = document.getElementById("selectedRouteDetails");
    if (!details || !option) {
        return;
    }

    details.innerHTML =
        "<b>Selected:</b> " + option.name + "<br>" +
        "<b>Actual distance:</b> " + option.actual_distance_miles + " mi " +
        "(error " + option.distance_error_miles + " mi)<br>" +
        "<b>Elevation gain:</b> " + option.actual_gain_ft + " ft " +
        "(error " + option.elevation_error_ft + " ft)<br>" +
        "<b>Descent:</b> " + option.actual_descent_ft + " ft<br>" +
        "<b>Trail:</b> " + option.trail_percent + "% · " +
        "<b>Connector:</b> " + option.connector_distance_miles + " mi<br>" +
        "<b>Repeated trail:</b> " + option.repeated_distance_miles + " mi · " +
        "<b>Score:</b> " + option.route_score + "<br>" +
        "<b>Max reach from start:</b> " + (option.max_reach_miles ?? 0) + " mi · " +
        "<b>Independent loops:</b> " + (option.independent_loops ?? 0) + " · " +
        "<b>Extra subloops:</b> " + (option.extra_subloops ?? 0) + "<br>" +
        "<b>Route footprint:</b> " + (option.footprint_sq_miles ?? 0) + " sq mi · " +
        "<b>Big-loop penalty:</b> " + (option.shape_penalty ?? 0) + "<br>" +
        "<b>Partial-edge tuning:</b> " + (option.partial_edge_used ? "YES" : "NO");
}


function selectRouteOption(index, fitMap = true) {
    if (!lastGeneratedRoute || !lastGeneratedRoute.route_options) {
        return;
    }

    const options = lastGeneratedRoute.route_options;
    if (index < 0 || index >= options.length) {
        return;
    }

    selectedRouteOptionIndex = index;

    routeOptionLines.forEach((line, lineIndex) => {
        if (lineIndex === index) {
            line.setStyle({weight: 7, opacity: 0.96, color: "#d60000"});
            line.bringToFront();
            routeLine = line;
        } else {
            line.setStyle({weight: 4, opacity: 0.30, color: "#4455aa"});
        }
    });

    document.querySelectorAll(".route-choice").forEach(button => {
        button.classList.toggle("selected", Number(button.dataset.routeIndex) === index);
    });

    const selected = options[index];
    renderSelectedRouteDetails(selected);
    downloadGpxButton.disabled = false;

    if (fitMap && routeLine) {
        map.fitBounds(routeLine.getBounds(), {padding: [30, 30]});
    }
}


function drawRouteOptions(result) {
    clearGeneratedRouteLines();
    const options = result.route_options || [];

    for (const option of options) {
        const coordinates = option.route.map(point => [point.lat, point.lon]);
        const line = L.polyline(coordinates, {
            weight: 4,
            opacity: 0.30,
            color: "#4455aa"
        }).addTo(map);
        routeOptionLines.push(line);
    }

    if (options.length > 0) {
        selectRouteOption(0, true);
    }
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


function getWorkspaceStartKey(data) {
    const lat = Number(data.start_lat);
    const lon = Number(data.start_lon);
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
        return "";
    }
    return lat.toFixed(5) + "," + lon.toFixed(5);
}


async function loadMasterTrailOverlayOnce() {
    if (masterTrailOverlayLoaded) {
        return;
    }

    if (masterTrailOverlayPromise) {
        return masterTrailOverlayPromise;
    }

    masterTrailOverlayPromise = (async () => {
        const response = await fetch("/trail-overlay");
        const result = await readJsonResponse(response);

        networkLayer.clearLayers();

        if (result.allowed_trails && result.allowed_trails.length > 0) {
            // One canvas-backed multi-polyline is much cheaper for Leaflet than
            // creating thousands of individual line-layer objects.
            L.polyline(
                result.allowed_trails,
                {
                    weight: 3,
                    opacity: 0.42,
                    color: "#666666",
                    interactive: false
                }
            ).addTo(networkLayer);
        }

        masterTrailOverlayLoaded = true;
        updateNetworkVisibility();
        return result;
    })();

    try {
        return await masterTrailOverlayPromise;
    } finally {
        masterTrailOverlayPromise = null;
    }
}


async function loadTrailNetwork(data) {
    const diagnostics = document.getElementById("diagnostics");
    const startKey = getWorkspaceStartKey(data);

    // Start the gray-overlay fetch in parallel. It is visualization only and
    // must never block routing-area preparation or route search.
    loadMasterTrailOverlayOnce().catch(error => {
        console.warn("Trail overlay load failed:", error);
    });

    // V12: the expensive workspace depends only on start. Distance/elevation
    // changes reuse it and never call this endpoint again from normal Generate.
    if (
        startKey &&
        loadedWorkspaceStartKey === startKey &&
        lastWorkspaceResult
    ) {
        return lastWorkspaceResult;
    }

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
                target_distance_miles: data.target_distance_miles,
                force_reload: false
            })
        }
    );

    const result = await readJsonResponse(response);

    loadedWorkspaceStartKey = startKey;
    lastWorkspaceResult = result;

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
        "<b>Routing area ready for this start:</b> YES<br>" +
        "<b>Distance/elevation changes reload trails:</b> NO<br>" +
        "<b>Gray overlay:</b> TIFF-wide · loaded once in browser<br>" +
        "<b>Current radius trail segments:</b> " +
        result.allowed_trail_segments +
        " physical segments<br>" +
        "<b>Master TIFF trail network:</b> " +
        result.master_physical_segments +
        " physical segments<br>" +
        "<b>Master TIFF:</b> " +
        result.master_tiff +
        "<br>" +
        "<b>Master graph file:</b> " +
        result.master_graph_file +
        "<br>" +
        "<b>Offline master loaded:</b> " +
        (result.master_loaded_from_disk ? "YES" : "NO") +
        "<br>" +
        "<b>Trail elevation precomputed:</b> " +
        (result.master_elevation_precomputed ? "YES" : "NO") +
        "<br>" +
        "<b>Workspace from cache:</b> " +
        (result.workspace_from_cache ? "YES" : "NO") +
        "<br>" +
        "<b>Workspace build:</b> " +
        result.workspace_build_seconds +
        " s<br>" +
        "<b>Workspace graph:</b> " +
        result.workspace_nodes + " nodes / " + result.workspace_edges + " edges<br>" +
        "<b>Current search graph:</b> " +
        result.network_nodes + " nodes / " + result.network_edges + " edges<br>" +
        "<b>Trail components:</b> " +
        result.trail_components_before +
        " → " +
        result.trail_components_after +
        "<br>" +
        "<b>Selective connectors:</b> " +
        result.selective_connectors_added +
        " added (" + result.selective_connector_queries + " corridor queries)<br>" +
        "<b>Routeable component:</b> " +
        result.routeable_component_nodes +
        " nodes / " + result.routeable_component_edges + " edges<br>" +
        "<b>Start snap distance:</b> " +
        result.snap_distance_m +
        " m<br>" +
        "<b>Exact start inserted:</b> " +
        (result.exact_start_inserted ? "YES" : "NO") +
        "<br>" +
        "<b>Requested point → trail:</b> " +
        result.start_trail_offset_m +
        " m<br>" +
        "<b>Current route search radius:</b> " +
        result.search_radius_m +
        " m<br>" +
        "<b>Workspace TIFF-covering radius:</b> " +
        result.workspace_max_radius_m +
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
    const startKey = getWorkspaceStartKey(data);

    networkButton.disabled = true;

    if (loadedWorkspaceStartKey === startKey && lastWorkspaceResult) {
        diagnostics.innerHTML = '<span class="success">Routing area is already loaded for this start. Distance/elevation changes reuse it.</span>';
        networkButton.disabled = false;
        return;
    }

    diagnostics.innerHTML = '<span class="warning">Preparing routing area for this start...</span>';

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
    selectedRouteOptionIndex = 0;
    downloadGpxButton.disabled = true;
    clearGeneratedRouteLines();

    try {
        const workspaceKey = getWorkspaceStartKey(data);
        const needsWorkspace = loadedWorkspaceStartKey !== workspaceKey || !lastWorkspaceResult;

        if (needsWorkspace) {
            results.innerHTML = '<span class="warning">Preparing routing area for this start...</span>';
            await loadTrailNetwork(data);
        }

        results.innerHTML = '<span class="warning">Searching route alternatives...</span>';

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
        const options = result.route_options || [];

        if (!result.route || result.route.length < 2) {
            throw new Error("Server returned an empty route.");
        }

        if (options.length === 0) {
            result.route_options = [{
                index: 0,
                name: "Route 1",
                actual_distance_miles: result.actual_distance_miles,
                distance_error_miles: result.distance_error_miles,
                actual_gain_ft: result.actual_gain_ft,
                actual_descent_ft: result.actual_descent_ft,
                elevation_error_ft: result.elevation_error_ft,
                route: result.route,
                gpx_export_points: result.gpx_export_points,
                repeated_edges: result.repeated_edges,
                repeated_distance_miles: result.repeated_distance_miles,
                repeated_nodes: result.repeated_nodes,
                immediate_reversals: result.immediate_reversals,
                connector_distance_miles: result.connector_distance_miles,
                trail_percent: result.trail_percent,
                independent_loops: result.independent_loops,
                extra_subloops: result.extra_subloops,
                branch_points: result.branch_points,
                max_reach_miles: result.max_reach_miles,
                footprint_sq_miles: result.footprint_sq_miles,
                shape_penalty: result.shape_penalty,
                route_score: result.route_score,
                partial_edge_used: result.partial_edge_used,
                partial_added_distance_miles: result.partial_added_distance_miles,
                partial_outward_distance_meters: result.partial_outward_distance_meters
            }];
        }

        lastGeneratedRoute = result;

        const routeButtons = result.route_options.map((option, index) => {
            return '<button type="button" class="route-choice" data-route-index="' + index + '">' +
                option.name + ' · ' + option.actual_distance_miles + ' mi · ' + option.actual_gain_ft + ' ft' +
                '</button>';
        }).join("");

        const expandedText = result.states_expanded === null ? "N/A" : result.states_expanded;

        results.innerHTML =
            '<span class="success"><b>Route search complete</b></span><br>' +
            '<b>Found route choices:</b> ' + result.route_options.length + '<br>' +
            '<span class="small">Routes are filtered to avoid near-duplicates; fewer than 5 may be shown when the trail network does not provide 5 materially different matches.</span>' +
            '<div class="route-choice-grid">' + routeButtons + '</div>' +
            '<div id="selectedRouteDetails"></div><br>' +
            '<b>Distance target:</b> ' + result.requested_distance_miles + ' mi<br>' +
            '<b>Elevation target:</b> ' + result.requested_gain_ft + ' ft<br><br>' +
            '<b>Search method:</b> ' + result.search_method + '<br>' +
            '<b>Route profile:</b> ' + result.route_profile + '<br>' +
            '<b>Search depth:</b> ' + result.search_steps + '<br>' +
            '<b>States expanded:</b> ' + expandedText + '<br>' +
            '<b>Start snap distance:</b> ' + result.snap_distance_m + ' m<br>' +
            '<b>Exact start inserted:</b> ' + (result.exact_start_inserted ? 'YES' : 'NO') + '<br>' +
            '<b>Requested point → trail:</b> ' + result.start_trail_offset_m + ' m<br><br>' +
            '<b>Start workspace cached:</b> ' + result.graph_from_cache + '<br>' +
            '<b>Elevation samples:</b> ' + result.unique_elevation_samples + '<br>' +
            '<b>Elevation sample spacing:</b> ~' + result.elevation_sample_spacing_m + ' m<br>' +
            '<b>Elevation smoothing:</b> ~' + result.elevation_smoothing_distance_m + ' m (' + result.elevation_smoothing_window_points + ' points)<br>' +
            (result.waypoint_accurate_finalists !== null && result.waypoint_accurate_finalists !== undefined
                ? '<b>Accurate waypoint finalists:</b> ' + result.waypoint_accurate_finalists + '<br>'
                : '') +
            '<b>Version:</b> ' + result.version + '<br>' +
            '<span class="small">Selected route is red. Other choices are faint blue. Click a route above to switch. Download GPX exports the selected route with coordinates only for COROS.</span>';

        document.querySelectorAll(".route-choice").forEach(button => {
            button.addEventListener("click", () => {
                selectRouteOption(Number(button.dataset.routeIndex), true);
            });
        });

        drawRouteOptions(result);

    } catch (error) {
        results.innerHTML =
            '<span class="error"><b>Error:</b> ' +
            error.message +
            '</span><br>' +
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


// Load the TIFF-wide gray overlay and prepare the default start workspace once.
reloadNetwork();
</script>

</body>
</html>
"""


# ============================================================
# ONE-TIME OFFLINE MASTER BUILD CLI
# ============================================================

if __name__ == "__main__":
    if "--build-master" in sys.argv:
        try:
            build_master_trail_graph()
        except Exception as exc:
            print(f"MASTER BUILD FAILED: {exc}", file=sys.stderr)
            raise SystemExit(1)
        raise SystemExit(0)

    print(
        "This file is the FastAPI app. Start it with uvicorn, or create the "
        "offline master graph with: python main.py --build-master"
    )
