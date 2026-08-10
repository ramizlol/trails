from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.responses import HTMLResponse, Response
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field

import math
import os
import json
import random
import time
import threading
import pickle
import sys
import subprocess
import tempfile
import shutil
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
LOCAL_OSM_PBF_PATH = os.path.join(BASE_DIR, "phoenix-tiff.osm.pbf")

METERS_PER_MILE = 1609.344
FEET_PER_METER = 3.28084

DEFAULT_LAT = 33.586055
DEFAULT_LON = -112.083341

ELEVATION_SAMPLE_SPACING_M = 5.0
GPX_TRAIL_MATCH_TOLERANCE_M = 25.0

MAX_CACHED_GRAPHS = 10
GRAPH_CACHE = {}

MASTER_GRAPH = None
MASTER_GRAPH_INFO = {}
MASTER_GRAPH_LOCK = threading.Lock()
MASTER_GRAPH_GRAPHML_PATH = os.path.join(BASE_DIR, "master_trails.graphml")
MASTER_GRAPH_PICKLE_PATH = os.path.join(BASE_DIR, "master_trails.pkl")
MASTER_GRAPH_PATH = MASTER_GRAPH_GRAPHML_PATH

MASTER_ROUTING_GRAPH = None
MASTER_ROUTING_INFO = {}
MASTER_ROUTING_LOCK = threading.Lock()
MASTER_ROUTING_GRAPHML_PATH = os.path.join(BASE_DIR, "master_routing.graphml")
MASTER_ROUTING_PICKLE_PATH = os.path.join(BASE_DIR, "master_routing.pkl")
ROUTING_NETWORK_SCHEMA = "trail-plus-sparse-connectors-v15-local-pbf"

DEM_BOUNDS_WGS84_CACHE = None

DEM_POINT_CACHE = {}
MAX_DEM_POINT_CACHE = 250000

APP_VERSION = "2026-08-09-v16-required-pass-through"
MASTER_NETWORK_SCHEMA = "trail-only-v15-local-pbf-precomputed"
ELEVATION_SMOOTHING_RADIUS = 5  
PARTIAL_TUNING_MAX_DEFICIT_M = 0.75 * METERS_PER_MILE
TRAIL_HIGHWAYS = {"path", "track", "steps"}
HARD_TRAIL_SURFACES = {"asphalt", "concrete", "concrete:lanes", "concrete:plates", "paving_stones", "sett", "cobblestone"}
CONNECTOR_HIGHWAYS = {"footway", "pedestrian", "cycleway", "bridleway", "residential", "living_street", "service", "unclassified", "tertiary", "secondary", "primary", "road"}
CONNECTOR_PATH_COST_MULTIPLIER = 2.5
CONNECTOR_FINAL_SCORE_WEIGHT = 120.0
CONNECTOR_CHEAP_SCORE_WEIGHT = 90.0

MAX_ROUTE_OPTIONS = 5
MAX_ROUTE_SHARED_FRACTION = 0.80

SELECTIVE_CONNECTORS_MIN_RADIUS_M = 8.0 * METERS_PER_MILE
SELECTIVE_CONNECTOR_MAX_COUNT = 4
SELECTIVE_CONNECTOR_MIN_COMPONENT_TRAIL_M = 250.0
SELECTIVE_CONNECTOR_MAX_GAP_M = 7000.0
SELECTIVE_CONNECTOR_ATTACH_MAX_M = 90.0
SELECTIVE_CONNECTOR_CACHE = {}
MAX_SELECTIVE_CONNECTOR_CACHE = 16

OFFLINE_CONNECTOR_MAX_COUNT = 28
OFFLINE_CONNECTOR_NEIGHBORS_PER_COMPONENT = 8
OFFLINE_CONNECTOR_MAX_GAP_M = 7000.0
OFFLINE_CONNECTOR_MIN_COMPONENT_TRAIL_M = 250.0

WAYPOINT_LEG_CACHE_MAX = 4096

DEFAULT_PASS_THROUGH_TOLERANCE_MILES = 0.25
MAX_REQUIRED_PASS_POINTS = 5
PASS_POINT_CANDIDATES_PER_ZONE = 18

MAX_CACHED_WORKSPACES = 3
WORKSPACE_CACHE = {}
WORKSPACE_CACHE_LOCK = threading.Lock()

MASTER_TRAIL_OVERLAY_JSON = None
MASTER_TRAIL_OVERLAY_LOCK = threading.Lock()
CONNECTOR_FILTER = '["highway"~"path|track|steps|footway|pedestrian|cycleway|bridleway|residential|living_street|service|unclassified|tertiary|secondary|primary|road"]'


# ============================================================
# REQUEST MODELS
# ============================================================

class RequiredPassPoint(BaseModel):
    lat: float
    lon: float
    tolerance_miles: float = DEFAULT_PASS_THROUGH_TOLERANCE_MILES


class RouteRequest(BaseModel):
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float
    target_distance_miles: float
    target_gain_ft: float
    pass_points: list[RequiredPassPoint] = Field(default_factory=list)


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

    # ENHANCEMENT: Dynamic minimum anchors scale with route size to prevent spaghetti routing
    dynamic_min_anchor_m = int((target_distance_miles * METERS_PER_MILE) * 0.35)
    dynamic_min_sep_m = int((target_distance_miles * METERS_PER_MILE) * 0.15)

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
            "min_anchor_distance_m": max(150, dynamic_min_anchor_m),
            "min_anchor_separation_m": max(140, dynamic_min_sep_m),
            "accurate_finalists": 14,
            "candidate_pool_multiplier": 3,
        }
    if target_distance_miles < 15.0:
        return {
            "name": "long-waypoint",
            "search_radius_m": search_radius_m,
            "attempts": 900,
            "anchor_counts": [3, 4, 4, 4],
            "min_anchor_distance_m": max(300, dynamic_min_anchor_m),
            "min_anchor_separation_m": max(250, dynamic_min_sep_m),
            "accurate_finalists": 14,
            "candidate_pool_multiplier": 3,
        }
    return {
        "name": "ultra-waypoint",
        "search_radius_m": search_radius_m,
        "attempts": 700,
        "anchor_counts": [4, 4, 5],
        "min_anchor_distance_m": max(400, dynamic_min_anchor_m),
        "min_anchor_separation_m": max(300, dynamic_min_sep_m),
        "accurate_finalists": 14,
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


# ENHANCEMENT: Spatial scoring to reward routes that expand outwards
def calculate_spatial_score(G, route_nodes, start_node):
    """
    Calculates a spatial footprint score (0.0 to ~0.5+). 
    Higher score means the route travels further away from the origin 
    rather than looping tightly on itself.
    """
    if not route_nodes:
        return 0.0
        
    start_lat = float(G.nodes[start_node]['y'])
    start_lon = float(G.nodes[start_node]['x'])
    
    max_dist = 0.0
    for node in route_nodes:
        dist = haversine_meters(
            start_lat, start_lon,
            float(G.nodes[node]['y']), float(G.nodes[node]['x'])
        )
        if dist > max_dist:
            max_dist = dist
            
    total_dist = path_distance_meters(G, route_nodes)
    return max_dist / max(1.0, total_dist)


# ============================================================
# EDGE HELPERS
# ============================================================

def edge_routing_cost(data, target_distance_miles=None):
    length = float(data.get("length", 0) or 0)
    
    # ENHANCEMENT: Lower the connector penalty dynamically for longer runs 
    # so the algorithm will cross streets to find more trail networks.
    if str(data.get("route_class", "trail")) == "connector":
        multiplier = CONNECTOR_PATH_COST_MULTIPLIER
        if target_distance_miles is not None and target_distance_miles >= 8.0:
            multiplier = 1.5 
        return max(0.0, length) * multiplier
        
    return max(0.0, length)


def get_shortest_edge(G, u, v, target_distance_miles=None):
    edge_data = G.get_edge_data(u, v)
    if not edge_data:
        return None
    return min(edge_data.values(), key=lambda edge: (edge_routing_cost(edge, target_distance_miles), float(edge.get("length", float("inf")))))


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
    # Assume polyline_distance_meters, densify_polyline, elevations_for_coords, 
    # smooth_elevations, calculate_ascent_descent are defined elsewhere.
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
# REQUIRED PASS-THROUGH POINTS
# ============================================================

def _next_virtual_node_id(G):
    node = -1
    while node in G:
        node -= 1
    return node


def insert_required_pass_point(G, lat, lon, tolerance_meters):
    lat = float(lat)
    lon = float(lon)
    tolerance_meters = float(tolerance_meters)

    if tolerance_meters <= 0:
        raise HTTPException(status_code=400, detail="Pass-through tolerance must be greater than 0.")

    snap_graph = trail_only_graph(G)
    if snap_graph.number_of_edges() == 0:
        raise HTTPException(status_code=400, detail="No natural trails are available near the required pass-through point.")

    projected = ox.projection.project_graph(snap_graph)
    projected_crs = projected.graph.get("crs")
    if projected_crs is None:
        raise HTTPException(status_code=500, detail="Could not project trail graph for required pass-through point.")

    transformer = Transformer.from_crs("EPSG:4326", projected_crs, always_xy=True)
    x, y = transformer.transform(lon, lat)

    edge_id, edge_distance_m = ox.distance.nearest_edges(
        projected, X=float(x), Y=float(y), return_dist=True
    )
    values = list(edge_id) if not isinstance(edge_id, tuple) else list(edge_id)
    if len(values) < 3:
        raise HTTPException(status_code=500, detail="Nearest trail edge returned an invalid identifier for pass-through point.")

    u, v, key = values[:3]
    edge_distance_m = float(np.atleast_1d(edge_distance_m)[0])
    if edge_distance_m > tolerance_meters:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Required pass-through point is {edge_distance_m:.0f} m from the nearest natural trail, "
                f"outside its {tolerance_meters:.0f} m tolerance."
            ),
        )

    selected_data = G.get_edge_data(u, v, key)
    if selected_data is None:
        selected_data = get_shortest_edge(G, u, v)
    if selected_data is None:
        raise HTTPException(status_code=500, detail="Could not read nearest trail edge for pass-through point.")

    selected_coords = oriented_edge_coords(G, u, v, selected_data)
    nearest = nearest_position_on_polyline(selected_coords, lon, lat)
    if nearest is None:
        raise HTTPException(status_code=500, detail="Could not project required pass-through point onto trail geometry.")

    split_lon = float(nearest["projected_lon"])
    split_lat = float(nearest["projected_lat"])

    endpoint_candidates = []
    for endpoint in (u, v):
        d = haversine_meters(
            split_lat, split_lon,
            float(G.nodes[endpoint]["y"]), float(G.nodes[endpoint]["x"]),
        )
        endpoint_candidates.append((d, endpoint))
    endpoint_candidates.sort(key=lambda item: item[0])
    if endpoint_candidates and endpoint_candidates[0][0] <= 1.0:
        node = endpoint_candidates[0][1]
        return G, node, {
            "requested_lat": lat,
            "requested_lon": lon,
            "routing_lat": float(G.nodes[node]["y"]),
            "routing_lon": float(G.nodes[node]["x"]),
            "trail_offset_m": round(edge_distance_m, 2),
            "tolerance_m": float(tolerance_meters),
            "virtual_inserted": False,
            "source_edge": [int(u), int(v), str(key)],
        }

    H = G.copy()
    virtual_node = _next_virtual_node_id(H)
    elevation = elevations_for_coords([(split_lon, split_lat)])[0]
    H.add_node(
        virtual_node,
        x=split_lon,
        y=split_lat,
        elevation=float(elevation),
        virtual_pass_point=True,
    )

    candidates = []
    for a, b in [(u, v), (v, u)]:
        edge_dict = G.get_edge_data(a, b) or {}
        for candidate_key, data in edge_dict.items():
            if str(data.get("route_class", "trail")) != "trail":
                continue
            coords = oriented_edge_coords(G, a, b, data)
            position = nearest_position_on_polyline(coords, split_lon, split_lat)
            if position is None:
                continue
            if position["distance_m"] <= 1.5:
                candidates.append((a, b, candidate_key, data, coords, position))

    if not candidates:
        candidates.append((u, v, key, selected_data, selected_coords, nearest))

    split_count = 0
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

        if polyline_distance_meters(left) > 0.25:
            H.add_edge(a, virtual_node, key=candidate_key, **edge_attributes_for_split_part(data, left))
            split_count += 1
        if polyline_distance_meters(right) > 0.25:
            H.add_edge(virtual_node, b, key=candidate_key, **edge_attributes_for_split_part(data, right))
            split_count += 1

    if H.degree(virtual_node) == 0:
        H.remove_node(virtual_node)
        raise HTTPException(status_code=500, detail="Could not insert required pass-through point on the trail graph.")

    return H, virtual_node, {
        "requested_lat": lat,
        "requested_lon": lon,
        "routing_lat": split_lat,
        "routing_lon": split_lon,
        "trail_offset_m": round(edge_distance_m, 2),
        "tolerance_m": float(tolerance_meters),
        "virtual_inserted": True,
        "source_edge": [int(u), int(v), str(key)],
        "split_directed_pieces": split_count,
    }


def resolve_required_pass_points(G, pass_points):
    """Insert required zones and build small candidate-node sets for each."""
    pass_points = list(pass_points or [])
    if len(pass_points) > MAX_REQUIRED_PASS_POINTS:
        raise HTTPException(
            status_code=400,
            detail=f"A maximum of {MAX_REQUIRED_PASS_POINTS} required pass-through points is supported.",
        )

    H = G
    resolved = []

    # FIX: Completing the truncated loop logic
    for index, point in enumerate(pass_points):
        lat = float(point.lat)
        lon = float(point.lon)
        tolerance_meters = float(point.tolerance_miles) * METERS_PER_MILE
        
        # Insert the pass points incrementally into the graph copy (H)
        H, v_node, details = insert_required_pass_point(H, lat, lon, tolerance_meters)
        resolved.append({"node": v_node, "details": details})
        
    return H, resolved
