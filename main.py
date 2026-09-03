from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.responses import HTMLResponse, Response
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field
from array import array

import math
import gc
import os
import json
import hashlib
import random
import time
import threading
import pickle
import sys
import subprocess
import tempfile
import shutil
import xml.etree.ElementTree as ET
from collections import OrderedDict

import networkx as nx
import rustworkx as rx
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
DEM_PATH = os.path.join(BASE_DIR, "output_hh.tif")
LOCAL_OSM_PBF_PATH = os.path.join(BASE_DIR, "central-az.osm.pbf")

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
# V15 keeps the runtime fully offline. Natural trails plus a sparse set of
# useful prebuilt walking connectors are loaded from disk; normal requests never
# contact OpenStreetMap/Overpass.
MASTER_GRAPH = None
MASTER_GRAPH_INFO = {}
MASTER_GRAPH_LOCK = threading.Lock()
MASTER_GRAPH_GRAPHML_PATH = os.path.join(BASE_DIR, "master_trails.graphml")
MASTER_GRAPH_PICKLE_PATH = os.path.join(BASE_DIR, "master_trails.pkl")
# GraphML is the portable repository file. The pickle is an optional fast cache.
MASTER_GRAPH_PATH = MASTER_GRAPH_GRAPHML_PATH

# Prebuilt runtime routing graph: trail master + only the useful connector paths
# selected during the one-time offline build. This is the production graph.
MASTER_ROUTING_GRAPH = None
MASTER_ROUTING_INFO = {}
MASTER_ROUTING_LOCK = threading.Lock()
MASTER_ROUTING_GRAPHML_PATH = os.path.join(BASE_DIR, "master_routing.graphml")
MASTER_ROUTING_PICKLE_PATH = os.path.join(BASE_DIR, "master_routing.pkl")
ROUTING_NETWORK_SCHEMA = "trail-plus-sparse-connectors-v15-local-pbf"


# V37 runtime geographic routing tiles. The raw OSM/PBF is NEVER parsed by the
# production server. Codespaces prebuilds compact pickled NetworkX graphs in
# routing_tiles/, and Render loads only tiles intersecting the requested start
# workspace. This keeps runtime memory bounded as geographic coverage grows.
ROUTING_TILE_DIR = os.path.join(BASE_DIR, "routing_tiles")
ROUTING_TILE_MANIFEST_PATH = os.path.join(ROUTING_TILE_DIR, "manifest.json")
ROUTING_TILE_MANIFEST = None
ROUTING_TILE_MANIFEST_LOCK = threading.Lock()

# Keep only a few decompressed tile graphs resident. Pickle files are small,
# but NetworkX's in-memory representation is much larger than the file itself.
MAX_ROUTING_TILE_CACHE = 0
ROUTING_TILE_CACHE = OrderedDict()
ROUTING_TILE_CACHE_LOCK = threading.Lock()

# Extra graph coverage around the requested workspace prevents a route from
# being severed exactly at a tile boundary before the final radial truncation.
ROUTING_TILE_SELECTION_BUFFER_M = 1500.0
ROUTING_TILE_SCHEMA = "trail-routing-tile-v1"
ROUTING_TILE_MANIFEST_SCHEMA = "trail-routing-tile-manifest-v1"

# V39 lightweight map-overlay tiles. These contain only gray trail coordinates
# and are served without ever unpickling a NetworkX routing graph.
OVERLAY_TILE_DIR = os.path.join(BASE_DIR, "overlay_tiles")
OVERLAY_TILE_MANIFEST_PATH = os.path.join(OVERLAY_TILE_DIR, "manifest.json")
OVERLAY_TILE_MANIFEST = None
OVERLAY_TILE_MANIFEST_LOCK = threading.Lock()
OVERLAY_TILE_MANIFEST_SCHEMA = "trail-overlay-tile-manifest-v1"

DEM_BOUNDS_WGS84_CACHE = None

# Cache DEM values by rounded lat/lon. Graph construction already samples
# most trail points, so later route scoring can reuse those values instead
# of reopening/resampling the GeoTIFF for every finalist.
DEM_POINT_CACHE = {}
MAX_DEM_POINT_CACHE = 50000

APP_VERSION = "2026-08-20-v52-compact-elevation-profiles"
MASTER_NETWORK_SCHEMA = "trail-only-v15-local-pbf-precomputed"
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
# V20 exposes every accurately-scored finalist we keep instead of hiding
# routes behind strict target-quality gates. Exact route duplicates are already
# removed during exploration, so similarity is not used as a rejection rule.
MAX_ROUTE_OPTIONS = 14
MAX_ROUTE_SHARED_FRACTION = 1.0

# V15 builds only a sparse connector backbone OFFLINE. It never stores the full
# city street network, and normal route requests never make live connector calls.
SELECTIVE_CONNECTORS_MIN_RADIUS_M = 8.0 * METERS_PER_MILE
SELECTIVE_CONNECTOR_MAX_COUNT = 4
SELECTIVE_CONNECTOR_MIN_COMPONENT_TRAIL_M = 250.0
SELECTIVE_CONNECTOR_MAX_GAP_M = 7000.0
SELECTIVE_CONNECTOR_ATTACH_MAX_M = 90.0
SELECTIVE_CONNECTOR_CACHE = {}
MAX_SELECTIVE_CONNECTOR_CACHE = 16

# One-time sparse offline connector builder. These values intentionally keep the
# stored routing graph small: only practical bridges between meaningful trail
# components are saved, never the complete Phoenix street network.
OFFLINE_CONNECTOR_MAX_COUNT = 28
OFFLINE_CONNECTOR_NEIGHBORS_PER_COMPONENT = 8
OFFLINE_CONNECTOR_MAX_GAP_M = 7000.0
OFFLINE_CONNECTOR_MIN_COMPONENT_TRAIL_M = 250.0

# Per-search baseline waypoint-leg cache. A cached unpenalized leg is reused when
# it does not touch any edge already used by the current candidate loop.
WAYPOINT_LEG_CACHE_MAX = 4096

# V18 route-quality tuning.
#
# A single sensible out-and-back is cheap. Reusing the same corridor a third
# time is expensive, which prevents the search from solving mileage by stacking
# repeated laps. Literal graph bridges get even more lenient second-traversal
# treatment because there may truly be no alternative way back.
REUSED_EDGE_SECOND_COST_MULTIPLIER = 1.50
REUSED_EDGE_THIRD_PLUS_COST_MULTIPLIER = 10.0
NECESSARY_REUSED_EDGE_SECOND_COST_MULTIPLIER = 1.05
NECESSARY_REUSED_EDGE_THIRD_PLUS_COST_MULTIPLIER = 3.0

# Final/cheap repeated-distance scoring distinguishes one return traversal from
# third-and-later traversals. This is the key difference between a natural
# out-and-back stem and spaghetti/lap behavior.
OPTIONAL_SECOND_RETRACE_SCORE_FACTOR = 0.35
NECESSARY_SECOND_RETRACE_SCORE_FACTOR = 0.05
OPTIONAL_THIRD_PLUS_RETRACE_SCORE_FACTOR = 2.00
NECESSARY_THIRD_PLUS_RETRACE_SCORE_FACTOR = 0.50
NECESSARY_RETRACE_SCORE_FACTOR = 0.10
LONG_REPEAT_SCORE_WEIGHT = 70.0
LONG_REPEATED_NODE_WEIGHT = 4.0
LONG_IMMEDIATE_REVERSAL_WEIGHT = 3.0
CHEAP_REPEAT_SCORE_WEIGHT = 55.0
CHEAP_REPEATED_NODE_WEIGHT = 4.0
CHEAP_IMMEDIATE_REVERSAL_WEIGHT = 3.0

# Explicitly discourage small independent loops that exist only to avoid a
# reasonable retrace. The threshold scales with the target but is capped so a
# legitimate large secondary loop is not treated as "tiny".
SMALL_SUBLOOP_MIN_M = 120.0
SMALL_SUBLOOP_MAX_M = 1.50 * METERS_PER_MILE

# V16 required pass-through zones. A user point is not treated as an off-trail
# routing coordinate: it is snapped to a natural trail inside the tolerance,
# then every returned route must pass through at least one natural-trail
# candidate inside that zone. Multiple points are supported.
DEFAULT_PASS_THROUGH_TOLERANCE_MILES = 0.25
MAX_REQUIRED_PASS_POINTS = 10
PASS_POINT_CANDIDATES_PER_ZONE = 18

# V25 avoid zones. These are hard exclusions: routing edges that touch an
# avoid circle are removed from the per-request search graph before candidate
# generation. The saved offline routing graph remains unchanged.
DEFAULT_AVOID_RADIUS_MILES = 0.25
MAX_AVOID_AREAS = 5

# V27 point-selected trail controls. Clicking a trail in the browser stores a
# point on that physical trail segment; each route request resolves that point
# against the current request-local graph. Avoided segments are hard exclusions.
# Preferred segments are only a soft routing/scoring preference.
MAX_AVOID_SEGMENTS = 12
MAX_PREFERRED_SEGMENTS = 12
TRAIL_SEGMENT_SELECTION_MAX_DISTANCE_M = 45.0
PREFERRED_SEGMENT_ROUTING_COST_MULTIPLIER = 0.35
PREFERRED_SEGMENT_FINAL_REWARD = 18.0
PREFERRED_SEGMENT_CHEAP_REWARD = 12.0
DEFAULT_ROUTE_DIVERSITY = 50.0


# V15 separates the start workspace from distance/elevation
# targets. A workspace is keyed only by the requested start coordinate and
# contains the TIFF-wide trail graph plus the small selective connector set.
# Changing distance/gain reuses this workspace and only creates a cheap in-memory
# radius subgraph for the actual route search.
MAX_CACHED_WORKSPACES = 1

# The region graph is now large (~44k nodes). Previously the workspace covered
# the ENTIRE DEM footprint from any start, which meant copying/projecting the
# whole region graph for every new start point -- fine for the old small
# single-city build, but a major memory cost now. Cap the workspace to a
# generous straight-line radius instead. Routes wanting trails farther than
# this from their start will not find them; raise this if memory allows.
WORKSPACE_RADIUS_CAP_M = 25.0 * METERS_PER_MILE
WORKSPACE_CACHE = {}
WORKSPACE_CACHE_LOCK = threading.Lock()

# V38 gray trail overlay is viewport-scoped. Never serialize the whole regional
# network into one Python list/JSON string: that can exceed a 512 MB Render
# instance even though the on-disk routing tiles are compact.
TRAIL_OVERLAY_MAX_TILES = 12
TRAIL_OVERLAY_BOUNDS_PAD_DEG = 0.002
TRAIL_OVERLAY_LOCK = threading.Lock()
CONNECTOR_FILTER = '["highway"~"path|track|steps|footway|pedestrian|cycleway|bridleway|residential|living_street|service|unclassified|tertiary|secondary|primary|road"]'


# ============================================================
# REQUEST MODELS
# ============================================================

class RequiredPassPoint(BaseModel):
    lat: float
    lon: float
    tolerance_miles: float = DEFAULT_PASS_THROUGH_TOLERANCE_MILES


class AvoidArea(BaseModel):
    lat: float
    lon: float
    radius_miles: float = DEFAULT_AVOID_RADIUS_MILES


class TrailSegmentPoint(BaseModel):
    lat: float
    lon: float
    geometry: list[list[float]] = Field(default_factory=list)

    # V41 replacement selections carry the exact edge drawn in the gray overlay.
    # These stay optional so existing avoid/prefer segment controls still work.
    tile_id: str | None = None
    edge_u: int | None = None
    edge_v: int | None = None
    edge_key: str | None = None


class RouteRequest(BaseModel):
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float
    target_distance_miles: float
    target_gain_ft: float
    pass_points: list[RequiredPassPoint] = Field(default_factory=list)
    # Kept for backwards compatibility with v28 requests. The v29 browser no
    # longer uses ambiguous drag-to-reroute handles; it uses explicit section
    # replacement instead.
    route_edit_points: list[RequiredPassPoint] = Field(default_factory=list)
    avoid_areas: list[AvoidArea] = Field(default_factory=list)
    avoid_segments: list[TrailSegmentPoint] = Field(default_factory=list)
    prefer_segments: list[TrailSegmentPoint] = Field(default_factory=list)
    route_diversity: float = DEFAULT_ROUTE_DIVERSITY
    search_seed: int | None = None


class RouteCoordinatePoint(BaseModel):
    lat: float
    lon: float


class RouteSectionReplacementRequest(BaseModel):
    start_lat: float
    start_lon: float
    target_distance_miles: float
    target_gain_ft: float
    current_route: list[RouteCoordinatePoint]
    cut_start_index: int
    cut_end_index: int
    replacement_segments: list[TrailSegmentPoint] = Field(default_factory=list)
    avoid_areas: list[AvoidArea] = Field(default_factory=list)
    avoid_segments: list[TrailSegmentPoint] = Field(default_factory=list)
    prefer_segments: list[TrailSegmentPoint] = Field(default_factory=list)


class RouteRecalculateRequest(BaseModel):
    start_lat: float
    start_lon: float
    target_distance_miles: float
    target_gain_ft: float
    current_route: list[RouteCoordinatePoint]
    avoid_areas: list[AvoidArea] = Field(default_factory=list)
    avoid_segments: list[TrailSegmentPoint] = Field(default_factory=list)
    prefer_segments: list[TrailSegmentPoint] = Field(default_factory=list)


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
        "routing_engine": FAST_ROUTING_ENGINE,
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
    target_distance_m = float(target_distance_miles) * METERS_PER_MILE
    search_radius_m = max(250, int(target_distance_m))

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
            "min_anchor_distance_m": max(120, int(target_distance_m * 0.05)),
            "min_anchor_separation_m": max(100, int(target_distance_m * 0.04)),
            "far_anchor_min_ratio": 0.24,
            "far_anchor_attempt_probability": 0.45,
            "accurate_finalists": 14,
            "candidate_pool_multiplier": 4,
        }
    if target_distance_miles < 15.0:
        return {
            "name": "long-waypoint",
            "search_radius_m": search_radius_m,
            "attempts": 900,
            "anchor_counts": [3, 4, 4, 4],
            # V20: outward travel is encouraged, not required. Keep anchor
            # spacing loose enough that real trail topology can dictate shape.
            "min_anchor_distance_m": max(200, int(target_distance_m * 0.06)),
            "min_anchor_separation_m": max(180, int(target_distance_m * 0.05)),
            "far_anchor_min_ratio": 0.30,
            "far_anchor_attempt_probability": 0.55,
            "accurate_finalists": 14,
            "candidate_pool_multiplier": 4,
        }
    return {
        "name": "ultra-waypoint",
        "search_radius_m": search_radius_m,
        "attempts": 700,
        "anchor_counts": [4, 4, 5],
        "min_anchor_distance_m": max(250, int(target_distance_m * 0.07)),
        "min_anchor_separation_m": max(200, int(target_distance_m * 0.05)),
        "far_anchor_min_ratio": 0.32,
        "far_anchor_attempt_probability": 0.60,
        "accurate_finalists": 14,
        "candidate_pool_multiplier": 4,
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

    # V52 fix: this helper does not have a graph/u/v context. Reconstruct
    # the original 10 m sample positions directly from the stored edge geometry
    # and pair them with the compact float32 elevation array.
    original_elev = [
        float(value)
        for value in (original_data.get("dem_elevations_f32") or [])
    ]

    original_geometry = original_data.get("geometry")
    if original_geometry is not None:
        original_line_coords = [
            (float(lon), float(lat))
            for lon, lat in original_geometry.coords
        ]
    else:
        original_line_coords = list(coords)

    spacing = float(
        original_data.get(
            "dem_sample_spacing_m",
            ELEVATION_SAMPLE_SPACING_M,
        )
        or ELEVATION_SAMPLE_SPACING_M
    )

    original_coords = densify_polyline(
        original_line_coords,
        spacing,
    )

    if len(original_coords) != len(original_elev) and original_elev:
        dense = densify_polyline(
            original_line_coords,
            max(spacing / 2.0, 1.0),
        )

        if len(original_elev) == 1:
            original_coords = [dense[0]]
        elif dense:
            original_coords = []
            last = len(dense) - 1
            for i in range(len(original_elev)):
                idx = int(
                    round(
                        i * last / (len(original_elev) - 1)
                    )
                )
                original_coords.append(dense[idx])

    if original_coords and original_elev and len(original_coords) == len(original_elev):
        elevations = []
        for lon, lat in samples:
            nearest = nearest_position_on_polyline(
                original_coords,
                float(lon),
                float(lat),
            )
            if nearest is None:
                elevations.append(float(original_elev[0]))
                continue
            total_m = max(float(nearest.get("total_m", 0.0)), 1e-9)
            frac = max(
                0.0,
                min(1.0, float(nearest["along_m"]) / total_m),
            )
            pos = frac * (len(original_elev) - 1)
            lo = int(math.floor(pos))
            hi = min(lo + 1, len(original_elev) - 1)
            t = pos - lo
            elevations.append(
                float(original_elev[lo]) * (1.0 - t)
                + float(original_elev[hi]) * t
            )
    else:
        eu = original_data.get("elevation_start_m")
        ev = original_data.get("elevation_end_m")
        if eu is None:
            eu = 0.0
        if ev is None:
            ev = eu
        elevations = [
            float(eu) + (float(ev) - float(eu))
            * (i / max(len(samples) - 1, 1))
            for i in range(len(samples))
        ]

    attrs["dem_elevations_f32"] = array(
        "f",
        (float(value) for value in elevations),
    )
    attrs["dem_sample_spacing_m"] = float(ELEVATION_SAMPLE_SPACING_M)

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

    source_profile_coords, source_profile_elev = baked_edge_profile(
        H,
        u,
        v,
        selected_data,
    )
    elevation = 0.0
    if source_profile_coords and source_profile_elev:
        nearest_profile = nearest_position_on_polyline(
            source_profile_coords,
            split_lon,
            split_lat,
        )
        if nearest_profile is not None:
            frac = max(
                0.0,
                min(
                    1.0,
                    float(nearest_profile["along_m"])
                    / max(float(nearest_profile.get("total_m", 0.0)), 1e-9),
                ),
            )
            pos = frac * (len(source_profile_elev) - 1)
            lo = int(math.floor(pos))
            hi = min(lo + 1, len(source_profile_elev) - 1)
            t = pos - lo
            elevation = (
                float(source_profile_elev[lo]) * (1.0 - t)
                + float(source_profile_elev[hi]) * t
            )
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
# AVOID AREAS
# ============================================================

def _local_xy_meters(lon, lat, origin_lon, origin_lat):
    """Approximate lon/lat as local planar meters around one avoid-zone center."""
    lat_scale = 110540.0
    lon_scale = 111320.0 * max(0.01, math.cos(math.radians(float(origin_lat))))
    return (
        (float(lon) - float(origin_lon)) * lon_scale,
        (float(lat) - float(origin_lat)) * lat_scale,
    )


def _point_to_segment_distance_local_m(cx, cy, ax, ay, bx, by):
    dx = bx - ax
    dy = by - ay
    denom = dx * dx + dy * dy
    if denom <= 1e-12:
        return math.hypot(cx - ax, cy - ay)
    t = ((cx - ax) * dx + (cy - ay) * dy) / denom
    t = max(0.0, min(1.0, t))
    px = ax + t * dx
    py = ay + t * dy
    return math.hypot(cx - px, cy - py)


def _polyline_intersects_avoid_circle(coords, center_lat, center_lon, radius_m):
    """Return True when any polyline segment touches the avoid circle."""
    coords = list(coords or [])
    if not coords:
        return False

    radius_m = float(radius_m)
    center_lat = float(center_lat)
    center_lon = float(center_lon)

    local = [
        _local_xy_meters(lon, lat, center_lon, center_lat)
        for lon, lat in coords
    ]

    r2 = radius_m * radius_m
    for x, y in local:
        if x * x + y * y <= r2:
            return True

    for i in range(1, len(local)):
        ax, ay = local[i - 1]
        bx, by = local[i]
        if _point_to_segment_distance_local_m(0.0, 0.0, ax, ay, bx, by) <= radius_m:
            return True

    return False


def normalize_avoid_areas(avoid_areas):
    avoid_areas = list(avoid_areas or [])
    if len(avoid_areas) > MAX_AVOID_AREAS:
        raise HTTPException(
            status_code=400,
            detail=f"A maximum of {MAX_AVOID_AREAS} avoid areas is supported.",
        )

    normalized = []
    for index, area in enumerate(avoid_areas):
        lat = float(area.lat)
        lon = float(area.lon)
        radius_miles = float(area.radius_miles)
        if not math.isfinite(lat) or not math.isfinite(lon):
            raise HTTPException(status_code=400, detail=f"Avoid area {index + 1} has invalid coordinates.")
        if radius_miles <= 0 or radius_miles > 10.0:
            raise HTTPException(
                status_code=400,
                detail=f"Avoid area {index + 1} radius must be between 0 and 10 miles.",
            )
        normalized.append({
            "index": index,
            "lat": lat,
            "lon": lon,
            "radius_miles": radius_miles,
            "radius_m": radius_miles * METERS_PER_MILE,
        })
    return normalized


def requested_point_inside_avoid_area(lat, lon, normalized_areas):
    for area in normalized_areas:
        distance_m = haversine_meters(
            float(lat), float(lon), area["lat"], area["lon"]
        )
        if distance_m <= area["radius_m"]:
            return area, distance_m
    return None, None


def apply_avoid_areas_to_graph(G, avoid_areas, start_node=None, start_lat=None, start_lon=None, end_lat=None, end_lon=None):
    """
    Remove every routing edge that intersects an avoid circle, then retain only
    the undirected component reachable from the selected start.

    This is intentionally request-local: MASTER_ROUTING_GRAPH and the cached
    start workspace are never mutated.
    """
    normalized = normalize_avoid_areas(avoid_areas)
    if not normalized:
        return G, []

    if start_lat is not None and start_lon is not None:
        area, _ = requested_point_inside_avoid_area(start_lat, start_lon, normalized)
        if area is not None:
            raise HTTPException(
                status_code=400,
                detail=f"The start point is inside avoid area {area['index'] + 1}. Move or resize that avoid area.",
            )

    if end_lat is not None and end_lon is not None:
        # Same start/end is already covered above; this is mainly for point-to-point routes.
        area, _ = requested_point_inside_avoid_area(end_lat, end_lon, normalized)
        if area is not None:
            raise HTTPException(
                status_code=400,
                detail=f"The end point is inside avoid area {area['index'] + 1}. Move or resize that avoid area.",
            )

    H = G.copy()
    removals = []

    for u, v, key, data in list(H.edges(keys=True, data=True)):
        coords = oriented_edge_coords(H, u, v, data)
        if not coords:
            coords = [
                (float(H.nodes[u]["x"]), float(H.nodes[u]["y"])),
                (float(H.nodes[v]["x"]), float(H.nodes[v]["y"])),
            ]

        blocked = False
        for area in normalized:
            # Cheap endpoint/edge bounding check happens naturally inside the
            # local segment test; this graph is already radius-limited.
            if _polyline_intersects_avoid_circle(
                coords, area["lat"], area["lon"], area["radius_m"]
            ):
                blocked = True
                break
        if blocked:
            removals.append((u, v, key))

    for u, v, key in removals:
        if H.has_edge(u, v, key):
            H.remove_edge(u, v, key)

    # Remove dead nodes but keep the exact start long enough to provide a useful
    # blocked-start error below.
    protected = {start_node} if start_node is not None else set()
    dead_nodes = [node for node in H.nodes if H.degree(node) == 0 and node not in protected]
    if dead_nodes:
        H.remove_nodes_from(dead_nodes)

    if start_node is not None:
        if start_node not in H or H.degree(start_node) == 0:
            raise HTTPException(
                status_code=400,
                detail="The avoid areas block all usable trails from the selected start.",
            )

        undirected = H.to_undirected(as_view=True)
        component = set(nx.node_connected_component(undirected, start_node))
        H = H.subgraph(component).copy()

    if H.number_of_edges() == 0:
        raise HTTPException(status_code=400, detail="The avoid areas remove all usable routing trails in this search area.")

    return H, normalized


# ============================================================
# V27 CLICKED TRAIL-SEGMENT CONTROLS
# ============================================================

def _normalize_segment_points(points, max_count, label):
    points = list(points or [])
    if len(points) > max_count:
        raise HTTPException(
            status_code=400,
            detail=f"A maximum of {max_count} {label} trail segments is supported.",
        )
    normalized = []
    for index, point in enumerate(points):
        lat = float(point.lat)
        lon = float(point.lon)
        if not math.isfinite(lat) or not math.isfinite(lon):
            raise HTTPException(
                status_code=400,
                detail=f"{label.title()} trail segment {index + 1} has invalid coordinates.",
            )
        normalized.append({"index": index, "lat": lat, "lon": lon})
    return normalized


def _nearest_natural_edge_to_selected_point(G, lat, lon, max_distance_m=TRAIL_SEGMENT_SELECTION_MAX_DISTANCE_M):
    """Resolve a browser-selected trail point against the current request graph."""
    best = None
    seen = set()
    for u, v, key, data in G.edges(keys=True, data=True):
        if str(data.get("route_class", "trail")) != "trail":
            continue
        physical = undirected_edge_key(u, v)
        # Reverse-direction copies normally share the same physical geometry.
        if physical in seen:
            continue
        seen.add(physical)
        coords = oriented_edge_coords(G, u, v, data)
        if len(coords) < 2:
            continue
        nearest = nearest_position_on_polyline(coords, float(lon), float(lat))
        if nearest is None:
            continue
        distance_m = float(nearest["distance_m"])
        if best is None or distance_m < best["distance_m"]:
            best = {
                "u": u,
                "v": v,
                "key": key,
                "physical_key": physical,
                "distance_m": distance_m,
                "routing_lat": float(nearest["projected_lat"]),
                "routing_lon": float(nearest["projected_lon"]),
            }
    if best is None or best["distance_m"] > float(max_distance_m):
        return None
    return best


def apply_trail_segment_controls(G, avoid_segments, prefer_segments, start_node=None):
    """
    Resolve clicked trail points against this request graph.

    Avoid selections delete the selected physical edge in both directions.
    Preferred selections remain fully optional: they receive a lower routing
    cost and a modest scoring reward, but no candidate is required to use them.
    """
    avoid_points = _normalize_segment_points(avoid_segments, MAX_AVOID_SEGMENTS, "avoided")
    prefer_points = _normalize_segment_points(prefer_segments, MAX_PREFERRED_SEGMENTS, "preferred")
    if not avoid_points and not prefer_points:
        return G, [], []

    resolved_avoid = []
    resolved_prefer = []
    avoid_keys = set()
    prefer_keys = set()

    # Resolve against the unchanged graph first so one avoided segment does not
    # cause a later clicked point to snap to a different neighboring trail.
    for point in avoid_points:
        match = _nearest_natural_edge_to_selected_point(G, point["lat"], point["lon"])
        if match is None:
            continue
        avoid_keys.add(match["physical_key"])
        resolved_avoid.append({
            "lat": point["lat"],
            "lon": point["lon"],
            "routing_lat": match["routing_lat"],
            "routing_lon": match["routing_lon"],
            "distance_m": round(match["distance_m"], 2),
        })

    for point in prefer_points:
        match = _nearest_natural_edge_to_selected_point(G, point["lat"], point["lon"])
        if match is None:
            continue
        # Avoid always wins if the same physical segment is in both lists.
        if match["physical_key"] not in avoid_keys:
            prefer_keys.add(match["physical_key"])
        resolved_prefer.append({
            "lat": point["lat"],
            "lon": point["lon"],
            "routing_lat": match["routing_lat"],
            "routing_lon": match["routing_lon"],
            "distance_m": round(match["distance_m"], 2),
        })

    if not avoid_keys and not prefer_keys:
        return G, resolved_avoid, resolved_prefer

    H = G.copy()
    removals = []
    for u, v, key, data in list(H.edges(keys=True, data=True)):
        physical = undirected_edge_key(u, v)
        if physical in avoid_keys:
            removals.append((u, v, key))
            continue
        if physical in prefer_keys:
            data["preferred_segment"] = True
            base_cost = float(data.get("routing_cost", edge_routing_cost(data)))
            data["routing_cost"] = max(
                0.01,
                base_cost * PREFERRED_SEGMENT_ROUTING_COST_MULTIPLIER,
            )

    for u, v, key in removals:
        if H.has_edge(u, v, key):
            H.remove_edge(u, v, key)

    protected = {start_node} if start_node is not None else set()
    dead_nodes = [node for node in H.nodes if H.degree(node) == 0 and node not in protected]
    if dead_nodes:
        H.remove_nodes_from(dead_nodes)

    if start_node is not None:
        if start_node not in H or H.degree(start_node) == 0:
            raise HTTPException(
                status_code=400,
                detail="The avoided trail segments block all usable trails from the selected start.",
            )
        component = set(nx.node_connected_component(H.to_undirected(as_view=True), start_node))
        H = H.subgraph(component).copy()

    if H.number_of_edges() == 0:
        raise HTTPException(
            status_code=400,
            detail="The avoided trail segments remove all usable routing trails in this search area.",
        )

    return H, resolved_avoid, resolved_prefer


def preferred_segment_metrics(G, route_nodes):
    """Return traversed preferred mileage and number of distinct preferred edges hit."""
    total = 0.0
    hits = set()
    for i in range(len(route_nodes) - 1):
        u = route_nodes[i]
        v = route_nodes[i + 1]
        edge = get_shortest_edge(G, u, v)
        if edge is None or not bool(edge.get("preferred_segment", False)):
            continue
        total += float(edge.get("length", 0) or 0)
        hits.add(undirected_edge_key(u, v))
    return float(total), len(hits)


def normalized_route_diversity(value):
    try:
        value = float(value)
    except Exception:
        value = DEFAULT_ROUTE_DIVERSITY
    return max(0.0, min(100.0, value))


# ============================================================
# REQUIRED PASS-THROUGH POINTS
# ============================================================

def _next_virtual_node_id(G):
    node = -1
    while node in G:
        node -= 1
    return node


def insert_required_pass_point(G, lat, lon, tolerance_meters):
    """
    Snap a required user marker to the nearest NATURAL trail inside its
    tolerance and insert a temporary routing node on that trail.

    Unlike the exact start, the temporary node is placed on the projected
    trail position rather than at the user's possibly off-trail coordinate.
    This guarantees that the suggested route stays on the trail while still
    satisfying the user's pass-near requirement.
    """
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

    # If projection lands almost exactly at an existing endpoint, reuse it.
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
    source_profile_coords, source_profile_elev = baked_edge_profile(
        H,
        u,
        v,
        selected_data,
    )
    elevation = 0.0
    if source_profile_coords and source_profile_elev:
        nearest_profile = nearest_position_on_polyline(
            source_profile_coords,
            split_lon,
            split_lat,
        )
        if nearest_profile is not None:
            frac = max(
                0.0,
                min(
                    1.0,
                    float(nearest_profile["along_m"])
                    / max(float(nearest_profile.get("total_m", 0.0)), 1e-9),
                ),
            )
            pos = frac * (len(source_profile_elev) - 1)
            lo = int(math.floor(pos))
            hi = min(lo + 1, len(source_profile_elev) - 1)
            t = pos - lo
            elevation = (
                float(source_profile_elev[lo]) * (1.0 - t)
                + float(source_profile_elev[hi]) * t
            )
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

    for index, point in enumerate(pass_points):
        lat = float(point.lat)
        lon = float(point.lon)
        tolerance_miles = float(point.tolerance_miles)
        if not math.isfinite(lat) or not math.isfinite(lon):
            raise HTTPException(status_code=400, detail=f"Pass-through point {index + 1} has invalid coordinates.")
        if tolerance_miles <= 0 or tolerance_miles > 5.0:
            raise HTTPException(
                status_code=400,
                detail=f"Pass-through point {index + 1} tolerance must be between 0 and 5 miles.",
            )

        tolerance_m = tolerance_miles * METERS_PER_MILE
        H, primary_node, info = insert_required_pass_point(H, lat, lon, tolerance_m)
        info["index"] = index
        info["tolerance_miles"] = tolerance_miles
        info["primary_node"] = primary_node
        resolved.append(info)

    if not resolved:
        return H, []

    # Candidate alternatives let the search use any natural trail inside the
    # zone, rather than forcing only the single closest trail segment.
    trail_nodes = set()
    for u, v, data in H.edges(data=True):
        if str(data.get("route_class", "trail")) == "trail":
            trail_nodes.add(u)
            trail_nodes.add(v)

    for info in resolved:
        nearby = []
        for node in trail_nodes:
            d = haversine_meters(
                info["requested_lat"], info["requested_lon"],
                float(H.nodes[node]["y"]), float(H.nodes[node]["x"]),
            )
            if d <= info["tolerance_m"] + 0.5:
                nearby.append((float(d), node))

        # The projected primary node should always be present, but explicitly
        # include it in case numerical filtering omitted it.
        primary = info["primary_node"]
        primary_distance = haversine_meters(
            info["requested_lat"], info["requested_lon"],
            float(H.nodes[primary]["y"]), float(H.nodes[primary]["x"]),
        )
        nearby.append((float(primary_distance), primary))

        best_by_node = {}
        for d, node in nearby:
            if node not in best_by_node or d < best_by_node[node]:
                best_by_node[node] = d
        nearby = sorted((d, node) for node, d in best_by_node.items())
        nearby = nearby[:PASS_POINT_CANDIDATES_PER_ZONE]

        info["candidate_nodes"] = [node for _, node in nearby]
        info["candidate_offsets_m"] = [round(d, 1) for d, _ in nearby]

    return H, resolved


def required_pass_metrics_for_coords(coords, required_pass_points):
    if not required_pass_points:
        return []
    lonlat = [(float(p["lon"]), float(p["lat"])) for p in coords]
    result = []
    for info in required_pass_points:
        nearest = nearest_position_on_polyline(
            lonlat, info["requested_lon"], info["requested_lat"]
        )
        distance_m = float(nearest["distance_m"]) if nearest else float("inf")
        result.append({
            "index": int(info["index"]),
            "requested_lat": float(info["requested_lat"]),
            "requested_lon": float(info["requested_lon"]),
            "tolerance_miles": round(float(info["tolerance_miles"]), 3),
            "tolerance_m": round(float(info["tolerance_m"]), 1),
            "nearest_route_distance_m": round(distance_m, 1),
            "nearest_route_distance_miles": round(distance_m / METERS_PER_MILE, 3),
            "satisfied": bool(distance_m <= float(info["tolerance_m"]) + 2.0),
            "snapped_trail_lat": float(info["routing_lat"]),
            "snapped_trail_lon": float(info["routing_lon"]),
            "marker_to_nearest_trail_m": round(float(info["trail_offset_m"]), 1),
        })
    return result


def required_waypoint_profile(profile, target_distance_miles):
    """Give sub-4-mile constrained routes a waypoint-search profile."""
    if "attempts" in profile:
        return dict(profile)
    return {
        "name": "short-required-waypoint",
        "search_radius_m": profile["search_radius_m"],
        "attempts": 1000,
        "anchor_counts": [1, 2, 2, 3],
        "min_anchor_distance_m": 60,
        "min_anchor_separation_m": 70,
        "accurate_finalists": 14,
        "candidate_pool_multiplier": 3,
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
    _ensure_data_file_downloaded(DEM_PATH, "DEM_TIF_URL")

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
        raw_elevations = [
            float(lookup[(round(float(lat), 7), round(float(lon), 7))])
            for lon, lat in samples
        ]

        # V52: keep only compact float32 elevations. Sample coordinates are
        # reconstructed from the edge geometry at runtime using the same
        # ELEVATION_SAMPLE_SPACING_M. This avoids millions of Python
        # lon/lat tuples and float objects in production memory.
        G[u][v][key]["dem_elevations_f32"] = array(
            "f",
            (float(value) for value in raw_elevations),
        )
        G[u][v][key]["dem_sample_spacing_m"] = float(
            ELEVATION_SAMPLE_SPACING_M
        )

        elevations = smooth_elevations(
            raw_elevations,
            radius=ELEVATION_SMOOTHING_RADIUS,
        )
        ascent, descent = calculate_ascent_descent(elevations)
        G[u][v][key]["ascent_m"] = float(ascent)
        G[u][v][key]["descent_m"] = float(descent)
        G[u][v][key]["elevation_sample_count"] = len(samples)
    return G, len(lookup)


# ============================================================
# MASTER TIFF TRAIL GRAPH + LOCAL GRAPH CACHE
# ============================================================

def get_dem_bounds_wgs84():
    """Return coverage bounds in EPSG:4326.

    Build machines use the real TIFF. V49 production can run without the TIFF
    and falls back to the routing-tile manifest coverage baked from that DEM.
    """
    global DEM_BOUNDS_WGS84_CACHE

    if DEM_BOUNDS_WGS84_CACHE is not None:
        return DEM_BOUNDS_WGS84_CACHE

    if os.path.exists(DEM_PATH):
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

    # DEM-free production fallback.
    manifest = load_routing_tile_manifest()
    coverage = manifest.get("coverage_nominal") or {}
    try:
        DEM_BOUNDS_WGS84_CACHE = (
            float(coverage["west"]),
            float(coverage["south"]),
            float(coverage["east"]),
            float(coverage["north"]),
        )
    except Exception:
        raise HTTPException(
            status_code=500,
            detail=(
                "DEM is not present and routing_tiles/manifest.json does not "
                "contain usable coverage_nominal bounds."
            ),
        )

    return DEM_BOUNDS_WGS84_CACHE


def get_dem_signature():
    """Return the build DEM signature without requiring the TIFF at runtime."""
    if not os.path.exists(DEM_PATH):
        try:
            manifest = load_routing_tile_manifest()
            signature = str(manifest.get("dem_signature", "") or "")
            if signature:
                return signature
        except Exception:
            pass
        return "dem-free-runtime"

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
    lat = float(lat)
    lon = float(lon)

    if os.path.exists(DEM_PATH):
        left, bottom, right, top = get_dem_bounds_wgs84()
        return left <= lon <= right and bottom <= lat <= top

    try:
        manifest = load_routing_tile_manifest()
        coverage = manifest.get("coverage_nominal") or {}
        left = float(coverage["west"])
        bottom = float(coverage["south"])
        right = float(coverage["east"])
        top = float(coverage["north"])
        return left <= lon <= right and bottom <= lat <= top
    except Exception:
        return True


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
    connector_physical = set()

    for u, v, key, data in G.edges(keys=True, data=True):
        pk = (
            min(int(u), int(v)),
            max(int(u), int(v)),
            round(float(data.get("length", 0) or 0), 1),
        )
        physical.add(pk)
        if str(data.get("route_class", "trail")) == "connector":
            connector_physical.add(pk)
        else:
            trail_physical.add(pk)

    return {
        "nodes": int(G.number_of_nodes()),
        "edges": int(G.number_of_edges()),
        "physical_segments": len(physical),
        "trail_physical_segments": len(trail_physical),
        "connector_physical_segments": len(connector_physical),
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
            str(
                G.graph.get(
                    "master_loaded_source",
                    G.graph.get("routing_loaded_source", MASTER_GRAPH_PATH),
                )
            )
        ),
        "offline_connector_count": int(
            float(G.graph.get("offline_connector_count", 0) or 0)
        ),
        "offline_connector_path_meters": float(
            G.graph.get("offline_connector_path_meters", 0.0) or 0.0
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


def _ensure_data_file_downloaded(local_path, env_var_name):
    """
    If local_path is missing, try downloading it from the URL in the given
    environment variable (set this on Render to a GitHub Release asset URL).
    No-ops if the file already exists locally, or if the env var isn't set --
    so local/dev behavior with the file already present is unchanged.
    """
    if os.path.exists(local_path):
        return

    url = os.environ.get(env_var_name)
    if not url:
        return

    import urllib.request

    print(f"{os.path.basename(local_path)} not found locally; downloading from {env_var_name}...")
    tmp = local_path + ".downloading"
    try:
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, local_path)
        print(f"  saved {local_path} ({os.path.getsize(local_path) / (1024*1024):.1f} MB)")
    except Exception as exc:
        if os.path.exists(tmp):
            os.remove(tmp)
        print(f"  download failed: {exc}")


def try_load_saved_master_graph():
    """
    Load the prebuilt TIFF-wide graph without contacting OpenStreetMap.

    Prefer a local pickle cache because it is fastest. If the pickle was made
    by an incompatible Python/library version, fall back to the portable
    GraphML committed to the repo and refresh the local pickle automatically.
    """
    _ensure_data_file_downloaded(MASTER_GRAPH_PICKLE_PATH, "MASTER_TRAILS_PICKLE_URL")
    _ensure_data_file_downloaded(MASTER_GRAPH_GRAPHML_PATH, "MASTER_TRAILS_GRAPHML_URL")

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



def _run_osmium_command(args):
    """Run osmium-tool for the one-time local PBF build."""
    executable = shutil.which("osmium")
    if not executable:
        raise RuntimeError(
            "osmium-tool is required only for the one-time local build. "
            "In GitHub Codespaces run: sudo apt-get update && "
            "sudo apt-get install -y osmium-tool"
        )

    command = [executable] + list(args)
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=BASE_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - started

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "unknown osmium error").strip()
        raise RuntimeError(
            f"osmium command failed after {elapsed:.1f}s: {' '.join(command)}\n{detail}"
        )

    return elapsed


def load_local_highway_graph_from_pbf(pbf_path=None):
    """
    Read one local OSM PBF (a full region, or one tile of it) without Overpass.

    osmium-tool first filters to highway ways (while retaining referenced nodes),
    converts that temporary extract to OSM XML, and OSMnx builds the graph from
    the local XML. The temporary files disappear automatically.
    """
    configure_osmnx_trail_tags()

    pbf_path = pbf_path or LOCAL_OSM_PBF_PATH

    if not os.path.exists(pbf_path):
        raise RuntimeError(
            f"Missing {os.path.basename(pbf_path)} beside main.py. "
            "Commit the cropped local-area OSM PBF to the repo first."
        )

    pbf_mb = os.path.getsize(pbf_path) / (1024 * 1024)
    print(
        f"Loading local OSM source: {os.path.basename(pbf_path)} "
        f"({pbf_mb:.2f} MB)"
    )

    with tempfile.TemporaryDirectory(prefix="trail_pbf_build_") as tmpdir:
        highways_pbf = os.path.join(tmpdir, "highways.osm.pbf")
        highways_xml = os.path.join(tmpdir, "highways.osm")

        elapsed_filter = _run_osmium_command([
            "tags-filter",
            "-O",
            "-o", highways_pbf,
            pbf_path,
            "w/highway",
        ])
        print(f"  osmium highway filter: {elapsed_filter:.1f}s")

        elapsed_xml = _run_osmium_command([
            "cat",
            "-O",
            "-o", highways_xml,
            highways_pbf,
        ])
        print(f"  osmium PBF -> XML: {elapsed_xml:.1f}s")

        started = time.perf_counter()
        try:
            G = ox.graph.graph_from_xml(
                highways_xml,
                bidirectional=True,
                simplify=True,
                retain_all=True,
            )
        except Exception as exc:
            raise RuntimeError(f"Could not build local OSMnx graph from PBF extract: {exc}") from exc
        print(
            f"  OSMnx local graph parse: {time.perf_counter() - started:.1f}s "
            f"({G.number_of_nodes()} nodes / {G.number_of_edges()} directed edges)"
        )

    return G


def load_local_highway_graph_tiled(tile_paths):
    """
    Memory-bounded alternative to load_local_highway_graph_from_pbf(): runs the
    same osmium->XML->osmnx pipeline once per tile (so peak memory is bounded
    by the single largest tile, not the whole region), immediately narrows each
    tile down to just trail+connector-relevant edges (discarding everything
    else -- motorway, aeroway, etc. -- right away), frees the tile's raw graph,
    then unions the small narrowed tiles together.
    """
    import gc

    kept_parts = []
    for i, path in enumerate(tile_paths, start=1):
        print(f"--- Tile {i}/{len(tile_paths)}: {os.path.basename(path)} ---")
        raw_tile_G = load_local_highway_graph_from_pbf(pbf_path=path)

        trail_part = _classified_subgraph_from_local_osm(raw_tile_G, "trail")
        connector_part = _classified_subgraph_from_local_osm(raw_tile_G, "connector")

        del raw_tile_G
        gc.collect()

        print(
            f"  kept: {trail_part.number_of_edges()} trail edges, "
            f"{connector_part.number_of_edges()} connector edges"
        )
        kept_parts.append(trail_part)
        kept_parts.append(connector_part)

    merged = nx.compose_all(kept_parts)
    print(
        f"Merged all tiles: {merged.number_of_nodes()} nodes / "
        f"{merged.number_of_edges()} directed edges"
    )
    return merged


def _classified_subgraph_from_local_osm(source_G, route_class):
    """Create either the natural-trail graph or connector graph from local OSM."""
    H = nx.MultiDiGraph()
    H.graph.update(dict(source_G.graph))

    kept = 0
    removed = 0
    for u, v, key, data in source_G.edges(keys=True, data=True):
        classification = classify_walkable_edge(data)
        if classification != route_class:
            removed += 1
            continue
        if not edge_fully_inside_dem(source_G, u, v, data):
            removed += 1
            continue

        if u not in H:
            H.add_node(u, **dict(source_G.nodes[u]))
        if v not in H:
            H.add_node(v, **dict(source_G.nodes[v]))

        attrs = dict(data)
        attrs["route_class"] = route_class
        length = float(attrs.get("length", 0) or 0)
        attrs["routing_cost"] = (
            length * CONNECTOR_PATH_COST_MULTIPLIER
            if route_class == "connector"
            else length
        )
        H.add_edge(u, v, key=key, **attrs)
        kept += 1

    H.remove_nodes_from(list(nx.isolates(H)))
    H.graph["local_source_edges_kept"] = int(kept)
    H.graph["local_source_edges_removed"] = int(removed)
    H.graph["local_osm_source"] = os.path.basename(LOCAL_OSM_PBF_PATH)
    return H


def _build_walk_node_index(W):
    nodes = list(W.nodes)
    if not nodes:
        return None
    coords = np.asarray(
        [[float(W.nodes[n]["y"]), float(W.nodes[n]["x"])] for n in nodes],
        dtype=float,
    )
    tree = BallTree(np.radians(coords), metric="haversine")
    return {"nodes": nodes, "coords": coords, "tree": tree}


def _nearest_walk_candidates(W, walk_index, trail_G, trail_node, k=6):
    if walk_index is None or trail_node not in trail_G:
        return []

    lat = float(trail_G.nodes[trail_node]["y"])
    lon = float(trail_G.nodes[trail_node]["x"])
    k = max(1, min(int(k), len(walk_index["nodes"])))
    distances, indices = walk_index["tree"].query(
        np.radians(np.asarray([[lat, lon]], dtype=float)),
        k=k,
    )

    rows = []
    for angular, idx in zip(distances[0], indices[0]):
        node = walk_index["nodes"][int(idx)]
        distance_m = float(angular) * 6371000.0
        if distance_m <= SELECTIVE_CONNECTOR_ATTACH_MAX_M:
            rows.append((node, distance_m))
    return rows


def _find_local_connector_path(
    trail_G,
    walk_G,
    walk_index,
    source_hint,
    target_hint,
    straight_gap,
):
    """Find one useful connector path entirely inside the local PBF graph."""
    source_candidates = _nearest_walk_candidates(
        walk_G, walk_index, trail_G, source_hint, k=8
    )
    target_candidates = _nearest_walk_candidates(
        walk_G, walk_index, trail_G, target_hint, k=8
    )

    if not source_candidates or not target_candidates:
        return None, "no walkable network attachment within 90 m of both trail systems"

    source_nodes = [node for node, _ in source_candidates]
    source_offsets = {node: offset for node, offset in source_candidates}
    target_offsets = {node: offset for node, offset in target_candidates}

    max_reasonable = max(1800.0, float(straight_gap) * 3.5 + 1200.0)

    try:
        distances, paths = nx.multi_source_dijkstra(
            walk_G,
            source_nodes,
            cutoff=max_reasonable,
            weight="length",
        )
    except Exception as exc:
        return None, f"local connector shortest-path search failed: {exc}"

    reachable_targets = []
    for target_walk, target_offset in target_candidates:
        if target_walk not in distances:
            continue
        path = paths[target_walk]
        if not path:
            continue
        source_walk = path[0]
        source_offset = source_offsets.get(source_walk)
        if source_offset is None:
            continue
        total = float(distances[target_walk]) + float(source_offset) + float(target_offset)
        reachable_targets.append(
            (total, target_walk, source_walk, path, source_offset, target_offset)
        )

    if not reachable_targets:
        return None, "local walkable network does not connect the two trail systems"

    reachable_targets.sort(key=lambda row: row[0])
    _, target_walk, source_walk, path, source_offset, target_offset = reachable_targets[0]
    path_length = float(distances[target_walk])

    if path_length > max_reasonable:
        return None, "local walkable connector path is too indirect"

    return {
        "walk_graph": walk_G,
        "path": path,
        "path_length_m": path_length,
        "source_trail": source_hint,
        "source_walk": source_walk,
        "source_offset_m": float(source_offset),
        "target_trail": target_hint,
        "target_walk": target_walk,
        "target_offset_m": float(target_offset),
        "straight_gap_m": float(straight_gap),
    }, None


def build_master_trail_graph(local_source_graph=None):
    """
    ONE-TIME LOCAL BUILD from phoenix-tiff.osm.pbf.

    No Overpass/API requests are made. Natural trails are selected from the
    already-cropped local PBF, then the trail elevation heuristics are baked in
    from output_USGS10m.tif and saved to master_trails.graphml.
    """
    bbox = get_dem_bounds_wgs84()
    print("Building offline master trail graph from LOCAL PBF...")
    print(
        "TIFF bounds: "
        f"west={bbox[0]:.6f}, south={bbox[1]:.6f}, "
        f"east={bbox[2]:.6f}, north={bbox[3]:.6f}"
    )

    source_G = local_source_graph or load_local_highway_graph_from_pbf()
    G = _classified_subgraph_from_local_osm(source_G, "trail")

    if not G.number_of_edges():
        raise RuntimeError(
            "No usable natural trail network was found inside the TIFF footprint."
        )

    print(
        f"Filtered trail graph: {G.number_of_nodes()} nodes / "
        f"{G.number_of_edges()} directed edges"
    )
    print("Precomputing trail elevation heuristics from output_USGS10m.tif...")

    G, unique_samples = add_local_dem_edge_elevations(G)

    for _, _, _, data in G.edges(keys=True, data=True):
        data["route_class"] = "trail"
        data["routing_cost"] = float(edge_routing_cost(data))

    G.graph["dem_signature"] = get_dem_signature()
    G.graph["master_filtered_edges_removed"] = int(
        source_G.number_of_edges() - G.number_of_edges()
    )
    G.graph["master_tiff_name"] = os.path.basename(DEM_PATH)
    G.graph["master_network_version"] = APP_VERSION
    G.graph["master_network_schema"] = MASTER_NETWORK_SCHEMA
    G.graph["master_elevation_precomputed"] = "1"
    G.graph["master_elevation_unique_samples"] = int(unique_samples)
    G.graph["master_elevation_spacing_m"] = float(ELEVATION_SAMPLE_SPACING_M)
    G.graph["master_elevation_smoothing_radius"] = int(ELEVATION_SMOOTHING_RADIUS)
    G.graph["local_osm_source"] = os.path.basename(LOCAL_OSM_PBF_PATH)
    G.graph["overpass_used"] = "0"

    if not save_master_graph(G):
        raise RuntimeError(f"Could not save {MASTER_GRAPH_PATH}")

    graphml_mb = os.path.getsize(MASTER_GRAPH_GRAPHML_PATH) / (1024 * 1024)
    print(f"Saved portable trail graph: {MASTER_GRAPH_GRAPHML_PATH} ({graphml_mb:.2f} MB)")
    if os.path.exists(MASTER_GRAPH_PICKLE_PATH):
        pickle_mb = os.path.getsize(MASTER_GRAPH_PICKLE_PATH) / (1024 * 1024)
        print(f"Saved optional trail pickle: {MASTER_GRAPH_PICKLE_PATH} ({pickle_mb:.2f} MB)")
    print(f"Unique DEM samples baked into trail build: {unique_samples}")
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




def _validate_offline_routing_graph(G):
    if G is None:
        return False
    if not isinstance(G, (nx.MultiDiGraph, nx.MultiGraph, nx.DiGraph, nx.Graph)):
        return False
    if str(G.graph.get("dem_signature", "")) != get_dem_signature():
        return False
    if str(G.graph.get("routing_network_schema", "")) != ROUTING_NETWORK_SCHEMA:
        return False
    if str(G.graph.get("master_elevation_precomputed", "0")) != "1":
        return False
    if not G.number_of_nodes() or not G.number_of_edges():
        return False

    # Trail heuristics must be baked into the offline file. Connector heuristic
    # elevation may remain zero because finalists are still scored against the
    # continuous 5 m DEM profile.
    for _, _, _, data in G.edges(keys=True, data=True):
        route_class = str(data.get("route_class", "trail"))
        try:
            float(data.get("length", 0) or 0)
            float(data.get("routing_cost", edge_routing_cost(data)))
            if route_class == "trail":
                float(data["ascent_m"])
                float(data["descent_m"])
        except Exception:
            return False
    return True


def _normalize_loaded_routing_graph(G):
    """Normalize numeric edge attributes once when the runtime graph loads."""
    for _, _, _, data in G.edges(keys=True, data=True):
        data["length"] = float(data.get("length", 0) or 0)
        data["ascent_m"] = float(data.get("ascent_m", 0) or 0)
        data["descent_m"] = float(data.get("descent_m", 0) or 0)
        data["elevation_sample_count"] = int(
            float(data.get("elevation_sample_count", 0) or 0)
        )
        data["routing_cost"] = float(edge_routing_cost(data))
    return G


def try_load_saved_routing_graph():
    """Load the sparse trail+connector production graph entirely from disk."""
    _ensure_data_file_downloaded(MASTER_ROUTING_PICKLE_PATH, "MASTER_ROUTING_PICKLE_URL")
    _ensure_data_file_downloaded(MASTER_ROUTING_GRAPHML_PATH, "MASTER_ROUTING_GRAPHML_URL")

    if os.path.exists(MASTER_ROUTING_PICKLE_PATH):
        try:
            with open(MASTER_ROUTING_PICKLE_PATH, "rb") as f:
                G = pickle.load(f)
            if _validate_offline_routing_graph(G):
                G = _normalize_loaded_routing_graph(G)
                G.graph["routing_loaded_source"] = MASTER_ROUTING_PICKLE_PATH
                G.graph["master_loaded_source"] = MASTER_ROUTING_PICKLE_PATH
                return G
        except Exception:
            pass

    if os.path.exists(MASTER_ROUTING_GRAPHML_PATH):
        try:
            G = ox.io.load_graphml(filepath=MASTER_ROUTING_GRAPHML_PATH)
            if _validate_offline_routing_graph(G):
                G = _normalize_loaded_routing_graph(G)
                G.graph["routing_loaded_source"] = MASTER_ROUTING_GRAPHML_PATH
                G.graph["master_loaded_source"] = MASTER_ROUTING_GRAPHML_PATH
                # Best-effort local binary cache. Commit the GraphML; the pickle
                # is optional because Python/library versions can invalidate it.
                try:
                    tmp = MASTER_ROUTING_PICKLE_PATH + ".tmp"
                    with open(tmp, "wb") as f:
                        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
                    os.replace(tmp, MASTER_ROUTING_PICKLE_PATH)
                except Exception:
                    pass
                return G
        except Exception:
            pass
    return None


def save_master_routing_graph(G):
    try:
        ox.io.save_graphml(G, filepath=MASTER_ROUTING_GRAPHML_PATH)
    except Exception:
        return False

    tmp = MASTER_ROUTING_PICKLE_PATH + ".tmp"
    try:
        with open(tmp, "wb") as f:
            pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, MASTER_ROUTING_PICKLE_PATH)
    except Exception:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
    return True


def _component_centroid(G, nodes):
    coords = [
        (float(G.nodes[n]["y"]), float(G.nodes[n]["x"]))
        for n in nodes
        if n in G
    ]
    if not coords:
        return None
    arr = np.asarray(coords, dtype=float)
    return float(arr[:, 0].mean()), float(arr[:, 1].mean())


def _offline_connector_candidate_pairs(G, components, component_lengths):
    """
    Build a compact candidate list between meaningful trail components.

    For small component counts we inspect every pair. For larger sets we use
    centroid-nearest neighbors first, then calculate the true closest node pair
    only for those candidates. This is one-time builder work, not request work.
    """
    useful = [
        i for i, nodes in enumerate(components)
        if component_lengths.get(i, 0.0) >= OFFLINE_CONNECTOR_MIN_COMPONENT_TRAIL_M
    ]
    if len(useful) < 2:
        return []

    pair_ids = set()
    if len(useful) <= 40:
        for ai in range(len(useful)):
            for bi in range(ai + 1, len(useful)):
                pair_ids.add((useful[ai], useful[bi]))
    else:
        centroids = []
        centroid_ids = []
        for i in useful:
            c = _component_centroid(G, components[i])
            if c is not None:
                centroids.append(c)
                centroid_ids.append(i)

        if centroids:
            radians = np.radians(np.asarray(centroids, dtype=float))
            tree = BallTree(radians, metric="haversine")
            k = min(
                len(centroids),
                OFFLINE_CONNECTOR_NEIGHBORS_PER_COMPONENT + 1,
            )
            _, indices = tree.query(radians, k=k)
            for row, neighbors in enumerate(indices):
                a = centroid_ids[row]
                for idx in neighbors[1:]:
                    b = centroid_ids[int(idx)]
                    if a == b:
                        continue
                    pair_ids.add(tuple(sorted((a, b))))

    rows = []
    print(f"Evaluating {len(pair_ids)} possible sparse connector pairs...")
    for count, (a, b) in enumerate(sorted(pair_ids), start=1):
        pair = _closest_node_pair_same_graph(G, components[a], components[b])
        if pair is None:
            continue
        source_hint, target_hint, gap_m = pair
        if gap_m > OFFLINE_CONNECTOR_MAX_GAP_M:
            continue

        trail_bonus = 1.0 + min(
            component_lengths.get(a, 0.0) + component_lengths.get(b, 0.0),
            16000.0,
        ) / 16000.0 * 0.35
        score = float(gap_m) / trail_bonus
        rows.append((score, float(gap_m), a, b, source_hint, target_hint))

        if count % 100 == 0:
            print(f"  checked {count}/{len(pair_ids)} component pairs")

    rows.sort(key=lambda row: (row[0], row[1]))
    return rows


def build_master_routing_graph(rebuild_trails=True, source_G=None):
    """
    ONE-TIME V15 LOCAL ROUTING BUILD.

    Everything comes from the local OSM PBF (either the single-file path, or a
    pre-merged tiled graph passed in via source_G). The full local walkable
    graph is held only during this build. We save natural trails plus a sparse
    set of useful connector paths to master_routing.graphml. No Overpass
    requests are made during the build or during normal FastAPI use.
    """
    build_started = time.perf_counter()
    configure_osmnx_trail_tags()

    # Parse the local PBF once, then derive both trails and walkable connectors
    # from the same node/way source so topology and OSM IDs stay consistent.
    # If a pre-built source_G was passed in (e.g. from the tiled loader), reuse
    # it instead of loading LOCAL_OSM_PBF_PATH again.
    source_G = source_G or load_local_highway_graph_from_pbf()

    if rebuild_trails:
        trail_G = build_master_trail_graph(local_source_graph=source_G)
    else:
        trail_G = try_load_saved_master_graph()
        if trail_G is None:
            trail_G = build_master_trail_graph(local_source_graph=source_G)

    walk_G = _classified_subgraph_from_local_osm(source_G, "connector")
    if not walk_G.number_of_edges():
        raise RuntimeError("No walkable connector network was found in phoenix-tiff.osm.pbf")

    print(
        f"Local connector graph: {walk_G.number_of_nodes()} nodes / "
        f"{walk_G.number_of_edges()} directed edges"
    )
    walk_index = _build_walk_node_index(walk_G)

    G = trail_G.copy()
    trail_base = trail_only_graph(G)
    components = [
        set(c)
        for c in nx.connected_components(trail_base.to_undirected(as_view=True))
    ]
    component_lengths = {
        i: _component_unique_trail_length(G, nodes)
        for i, nodes in enumerate(components)
    }

    useful_count = sum(
        1 for i in range(len(components))
        if component_lengths.get(i, 0.0) >= OFFLINE_CONNECTOR_MIN_COMPONENT_TRAIL_M
    )

    print("Building sparse connector backbone from LOCAL PBF...")
    print(f"Trail components: {len(components)} total / {useful_count} useful")

    candidate_rows = _offline_connector_candidate_pairs(
        G,
        components,
        component_lengths,
    )

    union = nx.utils.UnionFind(range(len(components)))
    connector_count = 0
    connector_checks = 0
    connector_path_meters = 0.0
    errors = []

    for _, gap_m, a, b, source_hint, target_hint in candidate_rows:
        if connector_count >= OFFLINE_CONNECTOR_MAX_COUNT:
            break
        if union[a] == union[b]:
            continue

        connector_checks += 1
        print(
            f"  local connector {connector_checks}: components {a}<->{b}, "
            f"straight gap {gap_m:.0f} m"
        )

        result, error = _find_local_connector_path(
            G,
            walk_G,
            walk_index,
            source_hint,
            target_hint,
            gap_m,
        )
        if result is None:
            if error:
                errors.append(f"{a}<->{b}: {error}")
                print(f"    skipped: {error}")
            continue

        copied = _merge_connector_path(G, result, connector_count + 1)
        union.union(a, b)
        connector_count += 1
        connector_path_meters += float(copied)
        print(f"    added {copied:.0f} m connector path")

    for _, _, _, data in G.edges(keys=True, data=True):
        if str(data.get("route_class", "trail")) == "connector":
            data["ascent_m"] = float(data.get("ascent_m", 0) or 0)
            data["descent_m"] = float(data.get("descent_m", 0) or 0)
            data["elevation_sample_count"] = int(
                float(data.get("elevation_sample_count", 0) or 0)
            )
            data["routing_cost"] = float(edge_routing_cost(data))

    before = len(components)
    after = nx.number_connected_components(G.to_undirected(as_view=True))

    G.graph["dem_signature"] = get_dem_signature()
    G.graph["master_network_schema"] = MASTER_NETWORK_SCHEMA
    G.graph["master_elevation_precomputed"] = "1"
    G.graph["routing_network_schema"] = ROUTING_NETWORK_SCHEMA
    G.graph["routing_network_version"] = APP_VERSION
    G.graph["offline_connectors_prebuilt"] = "1"
    G.graph["offline_connector_count"] = int(connector_count)
    # Kept for API/UI compatibility: this is deliberately zero in V15.
    G.graph["offline_connector_queries"] = 0
    G.graph["offline_connector_checks"] = int(connector_checks)
    G.graph["offline_connector_path_meters"] = float(connector_path_meters)
    G.graph["offline_components_before"] = int(before)
    G.graph["offline_components_after"] = int(after)
    G.graph["offline_connector_errors_json"] = json.dumps(errors[:100])
    G.graph["local_osm_source"] = os.path.basename(LOCAL_OSM_PBF_PATH)
    G.graph["overpass_used"] = "0"

    if not save_master_routing_graph(G):
        raise RuntimeError(f"Could not save {MASTER_ROUTING_GRAPHML_PATH}")

    graphml_mb = os.path.getsize(MASTER_ROUTING_GRAPHML_PATH) / (1024 * 1024)
    print(f"Saved production routing graph: {MASTER_ROUTING_GRAPHML_PATH}")
    print(f"Routing GraphML size: {graphml_mb:.2f} MB")
    if os.path.exists(MASTER_ROUTING_PICKLE_PATH):
        pickle_mb = os.path.getsize(MASTER_ROUTING_PICKLE_PATH) / (1024 * 1024)
        print(f"Saved optional fast cache: {MASTER_ROUTING_PICKLE_PATH} ({pickle_mb:.2f} MB)")
    print(
        f"Local connectors added: {connector_count}; "
        f"connector checks: {connector_checks}; "
        f"selected connector mileage: {connector_path_meters / METERS_PER_MILE:.2f} mi"
    )
    print(f"Total local build time: {time.perf_counter() - build_started:.1f}s")
    print(
        "No Overpass calls were used. Commit master_routing.graphml beside "
        "main.py; normal Render requests remain fully offline."
    )
    return G



# ============================================================
# V37 RUNTIME ROUTING-TILE LOADER
# ============================================================

def load_routing_tile_manifest():
    """Load and validate routing_tiles/manifest.json once per server process."""
    global ROUTING_TILE_MANIFEST

    if ROUTING_TILE_MANIFEST is not None:
        return ROUTING_TILE_MANIFEST

    with ROUTING_TILE_MANIFEST_LOCK:
        if ROUTING_TILE_MANIFEST is not None:
            return ROUTING_TILE_MANIFEST

        if not os.path.exists(ROUTING_TILE_MANIFEST_PATH):
            raise HTTPException(
                status_code=503,
                detail=(
                    "Routing tile manifest is missing. Put the prebuilt "
                    "routing_tiles folder beside main.py. Expected: "
                    f"{ROUTING_TILE_MANIFEST_PATH}"
                ),
            )

        try:
            with open(ROUTING_TILE_MANIFEST_PATH, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Could not read routing tile manifest: {exc}",
            )

        if str(manifest.get("schema", "")) != ROUTING_TILE_MANIFEST_SCHEMA:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Routing tile manifest schema is incompatible: "
                    f"{manifest.get('schema')!r}"
                ),
            )

        rows = list(manifest.get("tiles") or [])
        if not rows:
            raise HTTPException(status_code=503, detail="Routing tile manifest contains no tiles.")

        # V47: production runtime is DEM-free. The routing tiles are the
        # authoritative elevation product and carry the build-time DEM signature.
        manifest_dem = str(manifest.get("dem_signature", "") or "")
        if not manifest_dem:
            raise HTTPException(
                status_code=503,
                detail="Routing tile manifest has no DEM signature.",
            )

        ROUTING_TILE_MANIFEST = manifest
        return ROUTING_TILE_MANIFEST


def _tile_row_bounds(row):
    bounds = row.get("nominal_bounds") or {}
    try:
        return (
            float(bounds["west"]),
            float(bounds["south"]),
            float(bounds["east"]),
            float(bounds["north"]),
        )
    except Exception:
        return None


def _bounds_intersect(a, b):
    aw, as_, ae, an = a
    bw, bs, be, bn = b
    return not (ae < bw or be < aw or an < bs or bn < as_)


def _workspace_bbox_wgs84(lat, lon, radius_m):
    bbox = ox.utils_geo.bbox_from_point(
        (float(lat), float(lon)),
        float(radius_m),
    )
    left, bottom, right, top = [float(v) for v in bbox]

    manifest = load_routing_tile_manifest()
    coverage = manifest.get("coverage_nominal") or {}
    try:
        dl = float(coverage["west"])
        db = float(coverage["south"])
        dr = float(coverage["east"])
        dt = float(coverage["north"])
        return (
            max(left, dl),
            max(bottom, db),
            min(right, dr),
            min(top, dt),
        )
    except Exception:
        return (left, bottom, right, top)


def routing_tiles_for_workspace(lat, lon, radius_m):
    """Return manifest rows whose nominal tile bounds intersect the workspace."""
    manifest = load_routing_tile_manifest()
    query_bbox = _workspace_bbox_wgs84(
        lat,
        lon,
        float(radius_m) + ROUTING_TILE_SELECTION_BUFFER_M,
    )

    selected = []
    for row in manifest.get("tiles", []):
        if int(row.get("edges", 0) or 0) <= 0:
            continue
        tile_bounds = _tile_row_bounds(row)
        if tile_bounds is None:
            continue
        if _bounds_intersect(query_bbox, tile_bounds):
            selected.append(row)

    selected.sort(key=lambda row: str(row.get("id", "")))
    return selected, query_bbox


def _load_routing_tile_uncached(row):
    filename = str(row.get("file", "") or "")
    if not filename:
        raise HTTPException(status_code=503, detail="Routing tile manifest row has no filename.")
    path = os.path.join(ROUTING_TILE_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(
            status_code=503,
            detail=f"Routing tile is missing: {filename}",
        )
    try:
        with open(path, "rb") as f:
            G = pickle.load(f)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Could not load routing tile {filename}: {exc}",
        )
    if not isinstance(G, nx.MultiDiGraph):
        # OSMnx graphs are MultiDiGraph. Converting here keeps the runtime API
        # consistent if an older builder emitted another NetworkX graph class.
        G = nx.MultiDiGraph(G)
    return G


def load_routing_tile(row):
    """Load one routing tile. V39 defaults to no decompressed-tile cache."""
    if MAX_ROUTING_TILE_CACHE <= 0:
        return _load_routing_tile_uncached(row)

    tile_id = str(row.get("id", row.get("file", "")))
    with ROUTING_TILE_CACHE_LOCK:
        cached = ROUTING_TILE_CACHE.pop(tile_id, None)
        if cached is not None:
            ROUTING_TILE_CACHE[tile_id] = cached
            return cached

    G = _load_routing_tile_uncached(row)
    with ROUTING_TILE_CACHE_LOCK:
        existing = ROUTING_TILE_CACHE.pop(tile_id, None)
        if existing is not None:
            ROUTING_TILE_CACHE[tile_id] = existing
            return existing
        ROUTING_TILE_CACHE[tile_id] = G
        while len(ROUTING_TILE_CACHE) > MAX_ROUTING_TILE_CACHE:
            ROUTING_TILE_CACHE.popitem(last=False)
    return G


def clear_routing_tile_cache():
    """Release decompressed tile graphs once a merged workspace owns the data."""
    with ROUTING_TILE_CACHE_LOCK:
        ROUTING_TILE_CACHE.clear()
    gc.collect()


def _edge_maybe_intersects_bbox(tile_G, u, v, data, bbox):
    """Cheap edge/bbox test used before copying a tile edge into the workspace.

    This prevents a request that barely touches a large 0.5-degree tile from
    materializing that entire NetworkX tile in the merged graph.
    """
    west, south, east, north = bbox

    geom = data.get("geometry")
    if geom is not None:
        try:
            gwest, gsouth, geast, gnorth = geom.bounds
            return not (
                float(geast) < west
                or float(gwest) > east
                or float(gnorth) < south
                or float(gsouth) > north
            )
        except Exception:
            pass

    try:
        ux = float(tile_G.nodes[u]["x"])
        uy = float(tile_G.nodes[u]["y"])
        vx = float(tile_G.nodes[v]["x"])
        vy = float(tile_G.nodes[v]["y"])
    except Exception:
        return False

    edge_west = min(ux, vx)
    edge_east = max(ux, vx)
    edge_south = min(uy, vy)
    edge_north = max(uy, vy)

    return not (
        edge_east < west
        or edge_west > east
        or edge_north < south
        or edge_south > north
    )


def _merge_tile_graph_into_bbox(target, tile_G, bbox):
    """Stream only bbox-relevant edges from one tile into the merged workspace."""
    if target.number_of_nodes() == 0:
        target.graph.update(tile_G.graph)

    kept_edges = 0
    kept_nodes = set()

    for u, v, key, data in tile_G.edges(keys=True, data=True):
        if not _edge_maybe_intersects_bbox(tile_G, u, v, data, bbox):
            continue

        if u not in kept_nodes:
            target.add_node(u, **tile_G.nodes[u])
            kept_nodes.add(u)
        if v not in kept_nodes:
            target.add_node(v, **tile_G.nodes[v])
            kept_nodes.add(v)

        if target.has_edge(u, v, key):
            target[u][v][key].update(data)
        else:
            target.add_edge(u, v, key=key, **data)
        kept_edges += 1

    return kept_edges


def _merge_tile_graph_into(target, tile_G):
    """Merge one tile in-place to avoid nx.compose_all's repeated full copies."""
    if target.number_of_nodes() == 0:
        target.graph.update(tile_G.graph)

    # Original OSM node IDs are retained by complete_ways extracts, so adjacent
    # tile graphs naturally stitch together on shared OSM nodes. Duplicate
    # reverse/overlap edges with the same (u,v,key) are updated, not duplicated.
    target.add_nodes_from(tile_G.nodes(data=True))
    for u, v, key, data in tile_G.edges(keys=True, data=True):
        if target.has_edge(u, v, key):
            target[u][v][key].update(data)
        else:
            target.add_edge(u, v, key=key, **data)


def tiled_routing_info(manifest, selected_rows, merged_G):
    all_rows = list(manifest.get("tiles") or [])
    return {
        "nodes": int(sum(int(r.get("nodes", 0) or 0) for r in all_rows)),
        "edges": int(sum(int(r.get("edges", 0) or 0) for r in all_rows)),
        "physical_segments": int(sum(int(r.get("trail_edges", 0) or 0) for r in all_rows) // 2),
        "trail_physical_segments": int(sum(int(r.get("trail_edges", 0) or 0) for r in all_rows) // 2),
        "connector_physical_segments": int(sum(int(r.get("connector_edges", 0) or 0) for r in all_rows) // 2),
        "filtered_edges_removed": 0,
        "loaded_from_disk": True,
        "elevation_precomputed": True,
        "saved_graph": os.path.join("routing_tiles", "manifest.json"),
        "offline_components_before": 0,
        "offline_components_after": 0,
        "offline_connector_queries": 0,
        "offline_connector_count": int(sum(int(r.get("connector_count", 0) or 0) for r in selected_rows)),
        "offline_connector_path_meters": float(sum(float(r.get("connector_path_meters", 0.0) or 0.0) for r in selected_rows)),
        "selected_tile_count": len(selected_rows),
        "selected_tile_ids": [str(r.get("id", "")) for r in selected_rows],
        "selected_graph_nodes": int(merged_G.number_of_nodes()),
        "selected_graph_edges": int(merged_G.number_of_edges()),
    }


def load_tiled_workspace_source_graph(lat, lon, radius_m):
    """
    Load/merge only routing tiles near one start, then truncate them to the
    requested workspace before exact-start insertion makes its graph copy.
    """
    manifest = load_routing_tile_manifest()
    rows, query_bbox = routing_tiles_for_workspace(lat, lon, radius_m)
    if not rows:
        raise HTTPException(
            status_code=400,
            detail="No prebuilt routing tiles cover this start/search area.",
        )

    merged = nx.MultiDiGraph()
    merged.graph["crs"] = "EPSG:4326"

    selected_tile_debug = []
    for row in rows:
        tile_G = load_routing_tile(row)
        source_edges = int(tile_G.number_of_edges())
        kept_edges = _merge_tile_graph_into_bbox(
            merged,
            tile_G,
            query_bbox,
        )
        selected_tile_debug.append(
            {
                "id": str(row.get("id", "")),
                "source_edges": source_edges,
                "kept_edges": int(kept_edges),
                "size_mb": round(
                    float(row.get("size_bytes", 0) or 0) / (1024.0 * 1024.0),
                    2,
                ),
            }
        )

        # MAX_ROUTING_TILE_CACHE is currently zero, but explicitly release the
        # local reference before loading the next tile to minimize peak memory.
        del tile_G
        gc.collect()

    print(
        "Routing tiles selected:",
        ", ".join(
            f"{row['id']}[{row['size_mb']:.2f}MB:{row['kept_edges']}/{row['source_edges']} edges]"
            for row in selected_tile_debug
        ),
    )

    if merged.number_of_edges() == 0:
        raise HTTPException(
            status_code=400,
            detail="Selected routing tiles contain no usable trails inside this search area.",
        )

    # Limit the merged tile envelope before insert_exact_routing_point projects
    # and copies the graph. This is the key Render-memory protection.
    local = extract_local_master_subgraph(
        merged,
        float(lat),
        float(lon),
        float(radius_m),
    )
    del merged
    # The local workspace now owns the merged/truncated data. Keeping the same
    # decompressed source tiles in the LRU would duplicate NetworkX objects in
    # memory on small Render instances.
    clear_routing_tile_cache()
    gc.collect()

    info = tiled_routing_info(manifest, rows, local)
    local.graph["routing_tile_ids"] = info["selected_tile_ids"]
    local.graph["routing_tile_count"] = len(rows)
    local.graph["routing_tile_query_bbox"] = tuple(query_bbox)
    local.graph["routing_tile_debug"] = selected_tile_debug
    return local, info

def get_master_routing_graph():
    global MASTER_ROUTING_GRAPH, MASTER_ROUTING_INFO

    if MASTER_ROUTING_GRAPH is not None:
        return MASTER_ROUTING_GRAPH, MASTER_ROUTING_INFO

    with MASTER_ROUTING_LOCK:
        if MASTER_ROUTING_GRAPH is not None:
            return MASTER_ROUTING_GRAPH, MASTER_ROUTING_INFO

        G = try_load_saved_routing_graph()
        if G is None:
            reason = (
                "Offline V15 routing graph is missing or incompatible. Put "
                "master_routing.graphml beside main.py. Build it locally from "
                "phoenix-tiff.osm.pbf with: python main.py --build-routing"
            )
            raise HTTPException(status_code=503, detail=reason)

        MASTER_ROUTING_GRAPH = G
        MASTER_ROUTING_INFO = master_graph_metadata(G, loaded_from_disk=True)
        MASTER_ROUTING_INFO["offline_components_before"] = int(
            float(G.graph.get("offline_components_before", 0) or 0)
        )
        MASTER_ROUTING_INFO["offline_components_after"] = int(
            float(G.graph.get("offline_components_after", 0) or 0)
        )
        MASTER_ROUTING_INFO["offline_connector_queries"] = int(
            float(G.graph.get("offline_connector_queries", 0) or 0)
        )

    return MASTER_ROUTING_GRAPH, MASTER_ROUTING_INFO


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
                "Start coordinate is outside routing coverage: "
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
            detail="No routing-covered search area exists around this start coordinate.",
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


def workspace_cache_key(lat, lon, radius_m):
    # Radius is bucketed so nearby target changes can reuse one workspace while
    # a later longer route cannot accidentally reuse a workspace that is too small.
    radius_bucket = int(math.ceil(float(radius_m) / 1000.0) * 1000)
    return (
        round(float(lat), 5),
        round(float(lon), 5),
        radius_bucket,
        str(load_routing_tile_manifest().get("dem_signature", "dem-free")),
        "start-workspace-v49",
    )


def dynamic_workspace_radius_m(requested_search_radius_m):
    requested = max(1000.0, float(requested_search_radius_m or 0.0))
    # Small safety margin for anchors/edge geometry without preparing a 25-mile
    # workspace for every short route.
    desired = max(3500.0, requested * 1.10 + 600.0)
    return min(float(WORKSPACE_RADIUS_CAP_M), desired)


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


def _vectorized_node_radial_distances(G, start_lat, start_lon):
    """Fast haversine distances from one start to every graph node."""
    nodes = list(G.nodes)
    if not nodes:
        return {}

    lats = np.fromiter(
        (float(G.nodes[n]["y"]) for n in nodes),
        dtype=float,
        count=len(nodes),
    )
    lons = np.fromiter(
        (float(G.nodes[n]["x"]) for n in nodes),
        dtype=float,
        count=len(nodes),
    )

    phi1 = math.radians(float(start_lat))
    lam1 = math.radians(float(start_lon))
    phi2 = np.radians(lats)
    lam2 = np.radians(lons)
    dphi = phi2 - phi1
    dlam = lam2 - lam1
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    distances = 6371000.0 * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return {node: float(distance) for node, distance in zip(nodes, distances)}


def finalize_workspace_graph(G, start_node, start_lat, start_lon):
    """
    Prepare only start-specific metadata.

    V15 normalizes the production routing graph once at load time, so we no
    longer walk every edge on every new start. Only the newly split exact-start
    edges need any defaults, and those inherit their source edge attributes.
    """
    radial = _vectorized_node_radial_distances(G, start_lat, start_lon)
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
        1 for u, v in G.edges()
        if u in reachable and v in reachable
    )

    G.graph["workspace_node_radial_m"] = radial
    G.graph["routeable_component_nodes"] = len(reachable)
    G.graph["routeable_component_edges"] = routeable_edges
    G.graph["offline_master_elevation_used"] = True
    return G


def get_start_workspace(lat, lon, requested_radius_m=None, force_rebuild=False):
    """
    V37 start workspace backed by compact geographic routing tiles.

    Production never opens the raw OSM PBF and never loads one statewide master
    graph. Only prebuilt .pkl tiles intersecting this start's bounded workspace
    are merged, then the exact start is inserted and existing route logic runs
    unchanged.
    """
    if not point_inside_dem(lat, lon):
        left, bottom, right, top = get_dem_bounds_wgs84()
        raise HTTPException(
            status_code=400,
            detail=(
                "Start coordinate is outside routing coverage: "
                f"west={left:.6f}, east={right:.6f}, "
                f"south={bottom:.6f}, north={top:.6f}."
            ),
        )

    requested_workspace_radius_m = dynamic_workspace_radius_m(requested_radius_m)
    key = workspace_cache_key(lat, lon, requested_workspace_radius_m)

    if not force_rebuild and key in WORKSPACE_CACHE:
        return WORKSPACE_CACHE[key], True

    with WORKSPACE_CACHE_LOCK:
        if not force_rebuild and key in WORKSPACE_CACHE:
            return WORKSPACE_CACHE[key], True

        started = time.perf_counter()
        full_footprint_radius_m = workspace_max_radius_meters(lat, lon)
        max_radius_m = min(full_footprint_radius_m, requested_workspace_radius_m)

        source_G, routing_info = load_tiled_workspace_source_graph(
            float(lat),
            float(lon),
            float(max_radius_m),
        )

        # One copy is still required because the exact-start helper splits the
        # nearest trail edge. Crucially, source_G is now only the local tiled
        # workspace rather than a statewide graph.
        G, start_node, start_info = insert_exact_routing_point(
            source_G,
            float(lat),
            float(lon),
        )
        del source_G
        gc.collect()

        G = finalize_workspace_graph(
            G,
            start_node,
            float(lat),
            float(lon),
        )

        build_seconds = time.perf_counter() - started
        connector_stats = {
            "attempted": False,
            "offline": True,
            "components_before": 0,
            "components_after": 0,
            "connectors_added": int(routing_info.get("offline_connector_count", 0) or 0),
            "connector_queries": 0,
            "connector_path_meters": float(routing_info.get("offline_connector_path_meters", 0.0) or 0.0),
            "errors": [],
        }

        G.graph["selective_connector_stats"] = connector_stats
        G.graph["workspace_start_node"] = start_node
        G.graph["workspace_start_info"] = dict(start_info)
        G.graph["workspace_start_lat"] = float(lat)
        G.graph["workspace_start_lon"] = float(lon)
        G.graph["workspace_max_radius_m"] = float(max_radius_m)
        G.graph["workspace_connector_radius_m"] = float(max_radius_m)
        G.graph["workspace_build_seconds"] = float(build_seconds)
        G.graph["workspace_master_file"] = routing_info.get(
            "saved_graph",
            os.path.join("routing_tiles", "manifest.json"),
        )
        G.graph["workspace_routing_tiles"] = list(routing_info.get("selected_tile_ids", []))

        workspace = {
            "graph": G,
            "start_node": start_node,
            "start_info": dict(start_info),
            "max_radius_m": float(max_radius_m),
            "connector_radius_m": float(max_radius_m),
            "build_seconds": float(build_seconds),
            "filtered_edges_removed": 0,
            "master_info": routing_info,
        }

        # A 512 MB instance keeps exactly one merged workspace. Release the old
        # NetworkX graph explicitly before retaining the new one.
        for old_key in list(WORKSPACE_CACHE.keys()):
            old_workspace = WORKSPACE_CACHE.pop(old_key, None)
            if old_workspace is not None:
                old_workspace.clear()
        gc.collect()

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
        requested_radius_m=float(radius_meters),
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


def load_overlay_tile_manifest():
    global OVERLAY_TILE_MANIFEST
    if OVERLAY_TILE_MANIFEST is not None:
        return OVERLAY_TILE_MANIFEST
    with OVERLAY_TILE_MANIFEST_LOCK:
        if OVERLAY_TILE_MANIFEST is not None:
            return OVERLAY_TILE_MANIFEST
        if not os.path.exists(OVERLAY_TILE_MANIFEST_PATH):
            raise HTTPException(
                status_code=503,
                detail="overlay_tiles/manifest.json is missing. Run python build_overlay_tiles.py in Codespaces.",
            )
        with open(OVERLAY_TILE_MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        if manifest.get("schema") != OVERLAY_TILE_MANIFEST_SCHEMA:
            raise HTTPException(status_code=503, detail="Overlay tile manifest schema is incompatible.")
        OVERLAY_TILE_MANIFEST = manifest
        return manifest


def _overlay_tile_bounds(row):
    b = row.get("bounds") or {}
    try:
        return (float(b["west"]), float(b["south"]), float(b["east"]), float(b["north"]))
    except Exception:
        return None


def _overlay_rows_for_bounds(west, south, east, north):
    manifest = load_overlay_tile_manifest()
    query = (
        float(west) - TRAIL_OVERLAY_BOUNDS_PAD_DEG,
        float(south) - TRAIL_OVERLAY_BOUNDS_PAD_DEG,
        float(east) + TRAIL_OVERLAY_BOUNDS_PAD_DEG,
        float(north) + TRAIL_OVERLAY_BOUNDS_PAD_DEG,
    )
    rows = []
    for row in manifest.get("tiles", []):
        if int(row.get("segments", 0) or 0) <= 0:
            continue
        bounds = _overlay_tile_bounds(row)
        if bounds is not None and _bounds_intersect(query, bounds):
            rows.append(row)
    too_wide = len(rows) > TRAIL_OVERLAY_MAX_TILES
    if too_wide:
        rows = []
    return manifest, rows, too_wide


def overlay_index_payload(west, south, east, north):
    manifest, rows, too_wide = _overlay_rows_for_bounds(west, south, east, north)
    return {
        "overlay_tiles": [
            {
                "id": str(row.get("id", "")),
                "url": "/overlay-tile/" + str(row.get("id", "")),
                "segments": int(row.get("segments", 0) or 0),
            }
            for row in rows
        ],
        "selected_tile_count": len(rows),
        "viewport_too_wide": bool(too_wide),
        "max_overlay_tiles": TRAIL_OVERLAY_MAX_TILES,
        "version": APP_VERSION,
    }


def overlay_tile_file(tile_id):
    manifest = load_overlay_tile_manifest()
    row = next((r for r in manifest.get("tiles", []) if str(r.get("id", "")) == str(tile_id)), None)
    if row is None:
        raise HTTPException(status_code=404, detail="Overlay tile not found.")
    filename = str(row.get("file", ""))
    path = os.path.join(OVERLAY_TILE_DIR, filename)
    if not filename or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Overlay tile file is missing.")
    return path


# ============================================================
# RUSTWORKX FAST RUNTIME ROUTING
# ============================================================

FAST_ROUTING_ENGINE = "rustworkx"


def _build_rustworkx_runtime(S):
    """Convert one request-local NetworkX DiGraph into a lightweight PyDiGraph.

    NetworkX remains the source of geometry and attributes. Rustworkx is used
    only for repeated pathfinding. The conversion is cached on S itself, so a
    waypoint search pays this cost once rather than once per leg.
    """
    cached = S.graph.get("_rustworkx_runtime")
    if cached is not None:
        return cached

    nx_nodes = list(S.nodes())
    rx_graph = rx.PyDiGraph(
        multigraph=False,
        node_count_hint=len(nx_nodes),
        edge_count_hint=S.number_of_edges(),
    )
    rx_indices = list(rx_graph.add_nodes_from(nx_nodes))
    nx_to_rx = {node: int(index) for node, index in zip(nx_nodes, rx_indices)}
    rx_to_nx = {int(index): node for node, index in zip(nx_nodes, rx_indices)}

    edge_rows = []
    for u, v, data in S.edges(data=True):
        if u not in nx_to_rx or v not in nx_to_rx:
            continue
        routing_cost = max(
            0.0,
            float(data.get("routing_cost", data.get("length", 1.0)) or 0.0),
        )
        length = max(0.0, float(data.get("length", 0.0) or 0.0))
        # Tuple payload is intentionally tiny:
        # (routing_cost, length, original_u, original_v)
        payload = (routing_cost, length, u, v)
        edge_rows.append((nx_to_rx[u], nx_to_rx[v], payload))

    if edge_rows:
        rx_graph.add_edges_from(edge_rows)

    runtime = {
        "graph": rx_graph,
        "nx_to_rx": nx_to_rx,
        "rx_to_nx": rx_to_nx,
    }
    S.graph["_rustworkx_runtime"] = runtime
    return runtime


def _rx_weight_fn_for(weight):
    if weight == "length":
        return lambda edge: float(edge[1])
    return lambda edge: float(edge[0])


def fast_shortest_path(S, source, target, weight="routing_cost", edge_use_counts=None, bridge_edges=None):
    """Return a NetworkX-node path while executing Dijkstra in rustworkx."""
    runtime = _build_rustworkx_runtime(S)
    nx_to_rx = runtime["nx_to_rx"]
    rx_to_nx = runtime["rx_to_nx"]
    graph = runtime["graph"]

    if source not in nx_to_rx or target not in nx_to_rx:
        raise nx.NodeNotFound(f"Source or target is not present in routing graph: {source}, {target}")

    source_rx = nx_to_rx[source]
    target_rx = nx_to_rx[target]

    if edge_use_counts:
        bridge_edges = bridge_edges if bridge_edges is not None else routing_bridge_edge_keys(S)

        def weight_fn(edge):
            base = float(edge[0])
            u = edge[2]
            v = edge[3]
            physical = undirected_edge_key(u, v)
            prior_count = int(edge_use_counts.get(physical, 0))
            if prior_count <= 0:
                return base
            if physical in bridge_edges:
                if prior_count == 1:
                    return base * NECESSARY_REUSED_EDGE_SECOND_COST_MULTIPLIER
                return base * NECESSARY_REUSED_EDGE_THIRD_PLUS_COST_MULTIPLIER
            if prior_count == 1:
                return base * REUSED_EDGE_SECOND_COST_MULTIPLIER
            return base * REUSED_EDGE_THIRD_PLUS_COST_MULTIPLIER
    else:
        weight_fn = _rx_weight_fn_for(weight)

    paths = rx.dijkstra_shortest_paths(
        graph,
        source_rx,
        target=target_rx,
        weight_fn=weight_fn,
    )
    # rustworkx returns a PathMapping, which supports indexed lookup but
    # intentionally does not implement dict.get().
    try:
        rx_path = paths[target_rx]
    except (KeyError, IndexError):
        raise nx.NetworkXNoPath(f"No path between {source} and {target}")

    if not rx_path:
        raise nx.NetworkXNoPath(f"No path between {source} and {target}")

    return [rx_to_nx[int(index)] for index in rx_path]


def fast_single_source_paths(S, source, weight="routing_cost"):
    """Return {networkx_node: [networkx path]} using one rustworkx Dijkstra run."""
    runtime = _build_rustworkx_runtime(S)
    nx_to_rx = runtime["nx_to_rx"]
    rx_to_nx = runtime["rx_to_nx"]
    graph = runtime["graph"]

    if source not in nx_to_rx:
        raise nx.NodeNotFound(f"Source is not present in routing graph: {source}")

    source_rx = nx_to_rx[source]
    paths = rx.dijkstra_shortest_paths(
        graph,
        source_rx,
        weight_fn=_rx_weight_fn_for(weight),
    )

    result = {}
    for destination_rx, path in paths.items():
        destination = rx_to_nx[int(destination_rx)]
        result[destination] = [rx_to_nx[int(index)] for index in path]
    result[source] = [source]
    return result


def fast_single_source_lengths(S, source, weight="routing_cost"):
    """Return {networkx_node: cost} using rustworkx Dijkstra path lengths."""
    runtime = _build_rustworkx_runtime(S)
    nx_to_rx = runtime["nx_to_rx"]
    rx_to_nx = runtime["rx_to_nx"]
    graph = runtime["graph"]

    if source not in nx_to_rx:
        raise nx.NodeNotFound(f"Source is not present in routing graph: {source}")

    source_rx = nx_to_rx[source]
    lengths = rx.dijkstra_shortest_path_lengths(
        graph,
        source_rx,
        _rx_weight_fn_for(weight),
    )
    result = {
        rx_to_nx[int(destination_rx)]: float(cost)
        for destination_rx, cost in lengths.items()
    }
    result[source] = 0.0
    return result


# ============================================================
# SIMPLE ROUTING GRAPH
# ============================================================

def connector_path_multiplier_for_target(target_distance_meters):
    """Long routes may use short connectors to unlock much larger trail systems."""
    miles = float(target_distance_meters) / METERS_PER_MILE
    if miles < 8.0:
        return CONNECTOR_PATH_COST_MULTIPLIER
    if miles < 15.0:
        return 1.60
    return 1.45


def connector_score_weight_for_target(target_distance_meters, cheap=False):
    miles = float(target_distance_meters) / METERS_PER_MILE
    if miles < 8.0:
        return CONNECTOR_CHEAP_SCORE_WEIGHT if cheap else CONNECTOR_FINAL_SCORE_WEIGHT
    if miles < 15.0:
        return 55.0 if cheap else 70.0
    return 45.0 if cheap else 60.0


def make_simple_routing_graph(G, connector_multiplier=None):
    S = nx.DiGraph()
    S.add_nodes_from(G.nodes(data=True))
    for u, v, data in G.edges(data=True):
        length = float(data.get("length", 0) or 0)
        if length <= 0:
            continue
        ascent = float(data.get("ascent_m", 0) or 0)
        route_class = str(data.get("route_class", "trail"))
        if route_class == "connector" and connector_multiplier is not None:
            routing_cost = length * float(connector_multiplier)
        else:
            routing_cost = float(data.get("routing_cost", edge_routing_cost(data)))
        if not S.has_edge(u, v) or routing_cost < float(S[u][v].get("routing_cost", float("inf"))):
            S.add_edge(
                u,
                v,
                length=length,
                ascent_m=ascent,
                route_class=route_class,
                routing_cost=routing_cost,
                preferred_segment=bool(data.get("preferred_segment", False)),
            )
    return S


# ============================================================
# REPETITION METRICS
# ============================================================

def routing_bridge_edge_keys(G):
    """
    Return undirected edge keys that are graph bridges.

    A bridge is the only graph connection between the two sides it joins. If a
    route enters a branch over a bridge, using that same corridor to come back is
    structurally unavoidable. V17 therefore treats repeated bridge mileage very
    differently from optional retracing.
    """
    cached = G.graph.get("_v17_bridge_edge_keys")
    if isinstance(cached, set):
        return cached

    U = nx.Graph()
    U.add_nodes_from(G.nodes)
    for u, v in G.edges():
        if u != v:
            U.add_edge(u, v)

    try:
        bridges = {undirected_edge_key(u, v) for u, v in nx.bridges(U)}
    except Exception:
        bridges = set()

    G.graph["_v17_bridge_edge_keys"] = bridges
    return bridges


def repeated_edge_breakdown(G, route_nodes, bridge_edges=None):
    counts = {}
    lengths = {}
    bridge_edges = bridge_edges if bridge_edges is not None else routing_bridge_edge_keys(G)

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
    necessary_repeated_distance = 0.0
    optional_repeated_distance = 0.0
    necessary_second_distance = 0.0
    optional_second_distance = 0.0
    necessary_third_plus_distance = 0.0
    optional_third_plus_distance = 0.0

    for edge_key, count in counts.items():
        if count <= 1:
            continue

        length = lengths.get(edge_key, 0.0)
        second_distance = length
        third_plus_distance = length * max(0, count - 2)
        extra_distance = second_distance + third_plus_distance

        repeated_edges += count - 1
        repeated_distance += extra_distance

        if edge_key in bridge_edges:
            necessary_repeated_distance += extra_distance
            necessary_second_distance += second_distance
            necessary_third_plus_distance += third_plus_distance
        else:
            optional_repeated_distance += extra_distance
            optional_second_distance += second_distance
            optional_third_plus_distance += third_plus_distance

    effective_repeated_distance = (
        optional_second_distance * OPTIONAL_SECOND_RETRACE_SCORE_FACTOR
        + necessary_second_distance * NECESSARY_SECOND_RETRACE_SCORE_FACTOR
        + optional_third_plus_distance * OPTIONAL_THIRD_PLUS_RETRACE_SCORE_FACTOR
        + necessary_third_plus_distance * NECESSARY_THIRD_PLUS_RETRACE_SCORE_FACTOR
    )

    return {
        "repeated_edges": repeated_edges,
        "repeated_distance_meters": repeated_distance,
        "necessary_repeated_distance_meters": necessary_repeated_distance,
        "optional_repeated_distance_meters": optional_repeated_distance,
        "necessary_second_retrace_meters": necessary_second_distance,
        "optional_second_retrace_meters": optional_second_distance,
        "necessary_third_plus_retrace_meters": necessary_third_plus_distance,
        "optional_third_plus_retrace_meters": optional_third_plus_distance,
        "effective_repeated_distance_meters": effective_repeated_distance,
    }


def repeated_edge_stats(G, route_nodes):
    breakdown = repeated_edge_breakdown(G, route_nodes)
    return breakdown["repeated_edges"], breakdown["repeated_distance_meters"]


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


def repeated_node_breakdown(route_nodes, bridge_edges):
    counts = {}
    bridge_nodes = set()
    for u, v in bridge_edges:
        bridge_nodes.add(int(u))
        bridge_nodes.add(int(v))

    for node in route_nodes[1:-1]:
        counts[node] = counts.get(node, 0) + 1

    necessary = 0
    optional = 0
    for node, count in counts.items():
        extra = max(0, count - 1)
        if not extra:
            continue
        if int(node) in bridge_nodes:
            necessary += extra
        else:
            optional += extra

    effective = optional + necessary * NECESSARY_RETRACE_SCORE_FACTOR
    return {
        "repeated_nodes": necessary + optional,
        "necessary_repeated_nodes": necessary,
        "optional_repeated_nodes": optional,
        "effective_repeated_nodes": effective,
    }


def count_immediate_reversals(route_nodes):
    count = 0

    for i in range(len(route_nodes) - 2):
        if route_nodes[i] == route_nodes[i + 2]:
            count += 1

    return count


def immediate_reversal_breakdown(route_nodes, bridge_edges):
    necessary = 0
    optional = 0
    for i in range(len(route_nodes) - 2):
        if route_nodes[i] != route_nodes[i + 2]:
            continue
        key = undirected_edge_key(route_nodes[i], route_nodes[i + 1])
        if key in bridge_edges:
            necessary += 1
        else:
            optional += 1

    effective = optional + necessary * NECESSARY_RETRACE_SCORE_FACTOR
    return {
        "immediate_reversals": necessary + optional,
        "necessary_immediate_reversals": necessary,
        "optional_immediate_reversals": optional,
        "effective_immediate_reversals": effective,
    }



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


def small_subloop_metrics(G, route_nodes, target_distance_meters):
    """Detect short mostly-unique closed detours, not ordinary out-and-backs."""
    if len(route_nodes) < 4:
        return {"count": 0, "distance_meters": 0.0, "threshold_meters": 0.0}

    threshold = min(
        SMALL_SUBLOOP_MAX_M,
        max(0.40 * METERS_PER_MILE, float(target_distance_meters) * 0.10),
    )

    edge_lengths = []
    edge_keys = []
    cumulative = [0.0]
    for i in range(len(route_nodes) - 1):
        u = route_nodes[i]
        v = route_nodes[i + 1]
        edge = get_shortest_edge(G, u, v)
        length = float(edge.get("length", 0) or 0) if edge else 0.0
        edge_lengths.append(length)
        edge_keys.append(undirected_edge_key(u, v))
        cumulative.append(cumulative[-1] + length)

    occurrences = {}
    for idx, node in enumerate(route_nodes):
        occurrences.setdefault(node, []).append(idx)

    candidate_intervals = []
    final_index = len(route_nodes) - 1
    start_node = route_nodes[0]

    for node, positions in occurrences.items():
        if len(positions) < 2:
            continue
        for a, b in zip(positions, positions[1:]):
            if node == start_node and a == 0 and b == final_index:
                continue
            span = cumulative[b] - cumulative[a]
            if span < SMALL_SUBLOOP_MIN_M or span > threshold:
                continue
            segment_keys = edge_keys[a:b]
            if len(segment_keys) < 2:
                continue
            # A true little loop mostly uses each edge once. An out-and-back
            # segment has many duplicated physical edges and is intentionally
            # excluded from this penalty.
            unique_fraction = len(set(segment_keys)) / max(len(segment_keys), 1)
            if unique_fraction < 0.80:
                continue
            candidate_intervals.append((a, b, span))

    # Keep a minimal non-duplicate set. Nested detections of the same tiny loop
    # can otherwise appear at multiple nodes along its boundary.
    candidate_intervals.sort(key=lambda row: (row[2], row[0]))
    kept = []
    for row in candidate_intervals:
        a, b, span = row
        overlaps_same = False
        for ka, kb, kspan in kept:
            overlap = max(0, min(b, kb) - max(a, ka))
            width = max(1, min(b - a, kb - ka))
            if overlap / width > 0.65:
                overlaps_same = True
                break
        if not overlaps_same:
            kept.append(row)

    return {
        "count": len(kept),
        "distance_meters": float(sum(row[2] for row in kept)),
        "threshold_meters": float(threshold),
    }


def small_subloop_penalty(G, route_nodes, target_distance_meters, cheap=False):
    metrics = small_subloop_metrics(G, route_nodes, target_distance_meters)
    target_miles = float(target_distance_meters) / METERS_PER_MILE
    if target_miles < 4.0:
        return 0.0, metrics

    if target_miles < 8.0:
        count_weight = 16.0 if cheap else 24.0
        distance_weight = 12.0 if cheap else 18.0
    elif target_miles < 15.0:
        count_weight = 32.0 if cheap else 46.0
        distance_weight = 20.0 if cheap else 30.0
    else:
        count_weight = 38.0 if cheap else 55.0
        distance_weight = 24.0 if cheap else 36.0

    distance_ratio = metrics["distance_meters"] / max(float(target_distance_meters), 1.0)
    penalty = metrics["count"] * count_weight + distance_ratio * distance_weight
    metrics["penalty"] = float(penalty)
    return float(penalty), metrics


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
        target_radial_ratio = 0.34
        cycle_weight = 42.0 if cheap else 62.0
        branch_weight = 6.0 if cheap else 9.0
        spread_weight = 46.0 if cheap else 68.0
        footprint_target = 0.024
        footprint_weight = 0.0 if cheap else 30.0
    else:
        target_radial_ratio = 0.36
        cycle_weight = 50.0 if cheap else 74.0
        branch_weight = 7.0 if cheap else 10.0
        spread_weight = 54.0 if cheap else 80.0
        footprint_target = 0.026
        footprint_weight = 0.0 if cheap else 36.0

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


def baked_edge_profile(G, u, v, edge):
    """Return oriented reconstructed sample coords + compact elevations."""
    packed = edge.get("dem_elevations_f32")
    if packed is None:
        # Compatibility with v47 tiles during a rolling/local transition.
        legacy_coords = list(edge.get("dem_sample_coords") or [])
        legacy_elev = list(edge.get("dem_raw_elevations_m") or [])
        if legacy_coords and legacy_elev and len(legacy_coords) == len(legacy_elev):
            coords = [(float(lon), float(lat)) for lon, lat in legacy_coords]
            elevations = [float(value) for value in legacy_elev]
        else:
            return [], []
    else:
        elevations = [float(value) for value in packed]
        if not elevations:
            return [], []

        edge_coords = oriented_edge_coords(G, u, v, edge)
        if len(edge_coords) < 2:
            return [], []

        spacing = float(
            edge.get("dem_sample_spacing_m", ELEVATION_SAMPLE_SPACING_M)
            or ELEVATION_SAMPLE_SPACING_M
        )
        coords = densify_polyline(edge_coords, spacing)
        if len(coords) < 2:
            coords = list(edge_coords)

        # Floating point/polyline reconstruction can occasionally differ by a
        # sample at a segment boundary. Resample positions to exactly match
        # the stored elevation count without touching the DEM.
        if len(coords) != len(elevations):
            dense = densify_polyline(edge_coords, max(spacing / 2.0, 1.0))
            if not dense:
                return [], []
            if len(elevations) == 1:
                coords = [dense[0]]
            else:
                coords = []
                last = len(dense) - 1
                for i in range(len(elevations)):
                    idx = int(round(i * last / (len(elevations) - 1)))
                    coords.append(dense[idx])

    u_lon = float(G.nodes[u]["x"])
    u_lat = float(G.nodes[u]["y"])
    first_lon, first_lat = coords[0]
    last_lon, last_lat = coords[-1]

    first_distance = abs(first_lon - u_lon) + abs(first_lat - u_lat)
    last_distance = abs(last_lon - u_lon) + abs(last_lat - u_lat)
    if last_distance < first_distance:
        coords.reverse()
        elevations.reverse()

    return coords, elevations

def route_baked_elevation_metrics(G, route_nodes):
    """Build continuous route elevation metrics without opening a DEM TIFF."""
    all_coords = []
    all_elevations = []

    for i in range(len(route_nodes) - 1):
        u = route_nodes[i]
        v = route_nodes[i + 1]
        edge = get_shortest_edge(G, u, v)
        if edge is None:
            continue

        coords, elevations = baked_edge_profile(G, u, v, edge)

        # Connector edges may not have detailed DEM samples. Fall back to
        # endpoint node elevation when present and otherwise carry the nearest
        # known elevation forward. This keeps the runtime DEM-free.
        if not coords:
            coords = oriented_edge_coords(G, u, v, edge)
            if len(coords) < 2:
                continue
            eu = G.nodes[u].get("elevation")
            ev = G.nodes[v].get("elevation")
            if eu is None and all_elevations:
                eu = all_elevations[-1]
            if ev is None:
                ev = eu if eu is not None else 0.0
            if eu is None:
                eu = ev
            elevations = [
                float(eu) + (float(ev) - float(eu)) * (j / max(len(coords) - 1, 1))
                for j in range(len(coords))
            ]

        if all_coords and coords:
            if (
                haversine_meters(
                    all_coords[-1][1], all_coords[-1][0],
                    coords[0][1], coords[0][0],
                ) < 0.25
            ):
                coords = coords[1:]
                elevations = elevations[1:]

        all_coords.extend(coords)
        all_elevations.extend(elevations)

    if len(all_coords) < 2 or len(all_elevations) < 2:
        return {
            "distance_meters": path_distance_meters(G, route_nodes),
            "gain_meters": path_gain_meters(G, route_nodes),
            "descent_meters": 0.0,
            "dem_sample_points": len(all_coords),
            "elevation_profile": [],
        }

    smoothed = smooth_elevations(
        all_elevations,
        radius=ELEVATION_SMOOTHING_RADIUS,
    )
    ascent_m, descent_m = calculate_ascent_descent(smoothed)

    distance_m = 0.0
    for i in range(len(all_coords) - 1):
        lon1, lat1 = all_coords[i]
        lon2, lat2 = all_coords[i + 1]
        distance_m += haversine_meters(lat1, lon1, lat2, lon2)

    return {
        "distance_meters": float(distance_m),
        "gain_meters": float(ascent_m),
        "descent_meters": float(descent_m),
        "dem_sample_points": len(all_coords),
        "elevation_profile": build_elevation_profile(all_coords, smoothed),
    }


def build_elevation_profile(dense_lonlat, smoothed_elevations, max_points=240):
    """Return a compact distance/elevation profile for browser rendering.

    The authoritative route scorer already densifies the route and samples the
    DEM, so this reuses those exact samples instead of performing any extra DEM
    reads. The profile is downsampled to keep multi-route API responses small.
    """
    if not dense_lonlat or not smoothed_elevations:
        return []

    count = min(len(dense_lonlat), len(smoothed_elevations))
    if count == 0:
        return []

    cumulative_m = [0.0] * count
    for i in range(1, count):
        lon1, lat1 = dense_lonlat[i - 1]
        lon2, lat2 = dense_lonlat[i]
        cumulative_m[i] = cumulative_m[i - 1] + haversine_meters(
            lat1, lon1, lat2, lon2
        )

    if count <= max_points:
        indices = list(range(count))
    else:
        # Preserve both endpoints and select evenly spaced source samples.
        indices = sorted(set(
            int(round(i * (count - 1) / (max_points - 1)))
            for i in range(max_points)
        ))

    profile = []
    for i in indices:
        elevation = smoothed_elevations[i]
        if elevation is None:
            continue
        lon, lat = dense_lonlat[i]
        profile.append({
            "distance_miles": round(cumulative_m[i] / METERS_PER_MILE, 3),
            "elevation_ft": round(float(elevation) * FEET_PER_METER, 1),
            "lat": round(float(lat), 7),
            "lon": round(float(lon), 7),
        })
    return profile


def route_geometry_metrics(coords):
    """Authoritative distance/gain from one continuous route geometry."""
    if len(coords) < 2:
        return {
            "distance_meters": 0.0,
            "gain_meters": 0.0,
            "descent_meters": 0.0,
            "dem_sample_points": 0,
            "elevation_profile": [],
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
        "elevation_profile": build_elevation_profile(dense, smoothed),
    }


def coordinate_distance_only_metrics(coords):
    """Compute exact coordinate distance without touching the DEM."""
    lonlat = [
        (float(point["lon"]), float(point["lat"]))
        for point in (coords or [])
    ]
    distance_m = 0.0
    for i in range(len(lonlat) - 1):
        lon1, lat1 = lonlat[i]
        lon2, lat2 = lonlat[i + 1]
        distance_m += haversine_meters(lat1, lon1, lat2, lon2)

    return {
        "distance_meters": float(distance_m),
        "gain_meters": 0.0,
        "descent_meters": 0.0,
        "dem_sample_points": 0,
        "elevation_profile": [],
    }


def score_route_coordinates(
    G,
    coords,
    route_nodes,
    target_distance_meters,
    target_gain_meters,
    partial_added_distance_m=0.0,
):
    if route_nodes:
        geometry = route_baked_elevation_metrics(G, route_nodes)
        # Preserve exact displayed route distance from the route geometry.
        if coords:
            lonlat = [
                (float(point["lon"]), float(point["lat"]))
                for point in coords
            ]
            exact_distance_m = 0.0
            for i in range(len(lonlat) - 1):
                lon1, lat1 = lonlat[i]
                lon2, lat2 = lonlat[i + 1]
                exact_distance_m += haversine_meters(lat1, lon1, lat2, lon2)
            geometry["distance_meters"] = float(exact_distance_m)
    else:
        # V48: production route generation must never reopen the TIFF.
        geometry = coordinate_distance_only_metrics(coords)
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

    bridge_edges = routing_bridge_edge_keys(G)
    repeat_breakdown = repeated_edge_breakdown(G, route_nodes, bridge_edges)
    repeated_edges = repeat_breakdown["repeated_edges"]
    repeated_distance = repeat_breakdown["repeated_distance_meters"]
    effective_repeated_distance = repeat_breakdown["effective_repeated_distance_meters"]
    # A partial out-and-back repeats the same physical trail by definition. Keep
    # it as an ordinary (not automatically necessary) retrace for scoring.
    partial_repeat = max(0.0, float(partial_added_distance_m) / 2.0)
    repeated_distance += partial_repeat
    # A single partial out-and-back is exactly the kind of natural second
    # traversal V18 allows. Do not score it like a third/lap traversal.
    effective_repeated_distance += partial_repeat * OPTIONAL_SECOND_RETRACE_SCORE_FACTOR
    repeat_ratio = effective_repeated_distance / total_distance

    node_breakdown = repeated_node_breakdown(route_nodes, bridge_edges)
    repeated_nodes = node_breakdown["repeated_nodes"]
    effective_repeated_nodes = node_breakdown["effective_repeated_nodes"]

    reversal_breakdown = immediate_reversal_breakdown(route_nodes, bridge_edges)
    immediate_reversals = reversal_breakdown["immediate_reversals"]
    effective_immediate_reversals = reversal_breakdown["effective_immediate_reversals"]
    connector_distance = connector_distance_meters(G, route_nodes)
    connector_ratio = connector_distance / max(total_distance, 1.0)
    preferred_distance, preferred_hit_count = preferred_segment_metrics(G, route_nodes)
    preferred_ratio = preferred_distance / max(total_distance, 1.0)
    preference_reward = (
        preferred_hit_count * PREFERRED_SEGMENT_FINAL_REWARD
        + min(0.20, preferred_ratio) * 60.0
    )

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
    subloop_penalty, subloop_metrics = small_subloop_penalty(
        G,
        route_nodes,
        target_distance_meters,
        cheap=False,
    )
    connector_score_weight = connector_score_weight_for_target(
        target_distance_meters,
        cheap=False,
    )

    if target_distance_meters < 4 * METERS_PER_MILE:
        repeat_weight = 35.0
        node_weight = 3.0
        reversal_weight = 3.0
    else:
        repeat_weight = LONG_REPEAT_SCORE_WEIGHT
        node_weight = LONG_REPEATED_NODE_WEIGHT
        reversal_weight = LONG_IMMEDIATE_REVERSAL_WEIGHT

    score = (
        distance_ratio * 190.0
        + gain_ratio * 240.0
        + repeat_ratio * repeat_weight
        + effective_repeated_nodes * node_weight
        + effective_immediate_reversals * reversal_weight
        + connector_ratio * connector_score_weight
        + shape_penalty
        + subloop_penalty
        - preference_reward
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
            "necessary_repeated_distance_meters": repeat_breakdown["necessary_repeated_distance_meters"],
            "optional_repeated_distance_meters": repeat_breakdown["optional_repeated_distance_meters"] + partial_repeat,
            "effective_repeated_distance_meters": effective_repeated_distance,
            "repeat_ratio": repeat_ratio,
            "repeated_nodes": repeated_nodes,
            "necessary_repeated_nodes": node_breakdown["necessary_repeated_nodes"],
            "optional_repeated_nodes": node_breakdown["optional_repeated_nodes"],
            "immediate_reversals": immediate_reversals,
            "necessary_immediate_reversals": reversal_breakdown["necessary_immediate_reversals"],
            "optional_immediate_reversals": reversal_breakdown["optional_immediate_reversals"],
            "connector_distance_meters": connector_distance,
            "connector_ratio": connector_ratio,
            "trail_fraction": max(0.0, 1.0 - connector_ratio),
            "preferred_distance_meters": preferred_distance,
            "preferred_hit_count": preferred_hit_count,
            "preference_reward": preference_reward,
            "cycle_rank": topology["cycle_rank"],
            "extra_cycles": topology["extra_cycles"],
            "branch_points": topology["branch_points"],
            "branch_excess": topology["branch_excess"],
            "max_radial_meters": max_radial_meters,
            "max_radial_ratio": shape_metrics["max_radial_ratio"],
            "footprint_area_m2": footprint_area_m2,
            "footprint_ratio": shape_metrics["footprint_ratio"],
            "shape_penalty": shape_metrics["shape_penalty"],
            "small_subloops": subloop_metrics["count"],
            "small_subloop_distance_meters": subloop_metrics["distance_meters"],
            "small_subloop_penalty": subloop_penalty,
            "score": score,
            "route_coordinates": coords,
            "route_elevation_sample_count": geometry["dem_sample_points"],
            "elevation_profile": geometry.get("elevation_profile", []),
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
    route_diversity=DEFAULT_ROUTE_DIVERSITY,
):
    """
    Select up to max_routes without throwing good solutions away.

    V27 turns diversity into a user-controlled ordering/selection preference.
    At 0, the best scores dominate. At 100, overlap with already-selected
    routes carries a strong penalty, so the visible alternatives spread across
    different trail corridors. Candidates are not hard-rejected by similarity.
    """
    ordered = sorted(scored_candidates, key=lambda item: item[0])
    if not ordered:
        return []

    diversity = normalized_route_diversity(route_diversity) / 100.0
    if diversity <= 0.001:
        return ordered[:max_routes]

    selected = [ordered.pop(0)]
    # Raw route scores commonly live in the tens/hundreds. This overlap penalty
    # is intentionally substantial at the high end but remains soft.
    overlap_weight = 180.0 * diversity

    while ordered and len(selected) < max_routes:
        best_index = 0
        best_effective = float("inf")
        for index, candidate in enumerate(ordered):
            score, route_nodes, _ = candidate
            max_overlap = max(
                route_shared_fraction(G, route_nodes, existing[1])
                for existing in selected
            )
            effective = float(score) + overlap_weight * max_overlap
            if effective < best_effective:
                best_effective = effective
                best_index = index
        selected.append(ordered.pop(best_index))

    return selected


def route_signature(route_nodes):
    """Stable short signature used by the browser to de-duplicate Find More batches."""
    forward = ",".join(str(node) for node in route_nodes)
    reverse = ",".join(str(node) for node in reversed(route_nodes))
    canonical = min(forward, reverse)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:16]


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
    connector_distance_miles = metrics.get("connector_distance_meters", 0.0) / METERS_PER_MILE
    retrace_percent = 100.0 * metrics.get("repeated_distance_meters", 0.0) / max(metrics.get("total_distance_meters", 0.0), 1.0)
    connector_percent = 100.0 * metrics.get("connector_distance_meters", 0.0) / max(metrics.get("total_distance_meters", 0.0), 1.0)
    preferred_distance_miles = metrics.get("preferred_distance_meters", 0.0) / METERS_PER_MILE

    return {
        "index": int(option_index),
        "name": f"Route {int(option_index) + 1}",
        "route_signature": route_signature(route_nodes),
        "actual_distance_miles": round(route_distance_miles, 2),
        "distance_error_miles": round(abs(route_distance_miles - request.target_distance_miles), 2),
        "actual_gain_ft": round(actual_gain_ft),
        "actual_descent_ft": round(actual_descent_ft),
        "elevation_error_ft": round(abs(actual_gain_ft - request.target_gain_ft)),
        "route": coords,
        "gpx_export_points": build_gpx_export_points(coords),
        "elevation_profile": metrics.get("elevation_profile", []),
        "route_nodes": len(route_nodes),
        "route_geometry_points": len(coords),
        "repeated_edges": metrics["repeated_edges"],
        "repeated_distance_miles": round(repeated_distance_miles, 2),
        "repeated_nodes": metrics["repeated_nodes"],
        "immediate_reversals": metrics["immediate_reversals"],
        "connector_distance_miles": round(connector_distance_miles, 2),
        "retrace_percent": round(retrace_percent, 1),
        "connector_percent": round(connector_percent, 1),
        "trail_percent": round(metrics.get("trail_fraction", 1.0) * 100.0, 1),
        "preferred_distance_miles": round(preferred_distance_miles, 2),
        "preferred_hit_count": int(metrics.get("preferred_hit_count", 0)),
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
        "required_pass_points": metrics.get("required_pass_points", []),
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
    route_diversity=DEFAULT_ROUTE_DIVERSITY,
):
    """Budgeted multi-objective closed-loop search with partial-edge tuning."""
    S = make_simple_routing_graph(G)
    bridge_edges = routing_bridge_edge_keys(S)
    reverse_S = S.reverse(copy=False)

    try:
        return_distance = fast_single_source_lengths(
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
        "preferred_distance": 0.0,
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
        preferred_ratio = state.get("preferred_distance", 0.0) / max(state["distance"], 1.0)
        reversal_penalty = state["reversals"] * 0.02
        connector_penalty = connector_ratio * 1.5
        preference_bonus = min(0.25, preferred_ratio) * 1.2
        # A tiny seeded jitter gives Find More a genuinely different short-loop
        # beam without overpowering the route-quality objectives.
        exploration_jitter = random.random() * 0.025

        return {
            "balanced": distance_error * 3.0 + density_error * 0.85 + repeat_ratio * 0.25 + reversal_penalty + connector_penalty - preference_bonus + exploration_jitter,
            "gain": density_error * 2.5 + distance_error * 1.0 + repeat_ratio * 0.20 + reversal_penalty + connector_penalty - preference_bonus + exploration_jitter,
            "distance": distance_error * 5.0 + density_error * 0.15 + repeat_ratio * 0.15 + reversal_penalty + connector_penalty - preference_bonus + exploration_jitter,
            "flat": gain_density * 8.0 + distance_error * 2.0 + repeat_ratio * 0.15 + reversal_penalty + connector_penalty - preference_bonus + exploration_jitter,
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
                    edge_length * (
                        NECESSARY_RETRACE_SCORE_FACTOR if edge_key in bridge_edges else 1.0
                    )
                    if already_used else 0.0
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
                new_preferred_distance = state.get("preferred_distance", 0.0) + (
                    edge_length if bool(edge.get("preferred_segment", False)) else 0.0
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
                            preferred_ratio = new_preferred_distance / max(new_distance, 1.0)

                            closed_candidates.append({
                                "route": list(new_route),
                                "distance": new_distance,
                                "gain": new_gain,
                                "distance_ratio": distance_ratio,
                                "gain_ratio": gain_ratio,
                                "repeat_ratio": repeat_ratio,
                                "gain_density": gain_density,
                                "connector_ratio": connector_ratio,
                                "preferred_ratio": preferred_ratio,
                                "cheap_balanced": distance_ratio * 3.0 + gain_ratio * 1.2 + repeat_ratio * 0.25 + connector_ratio * 1.5 - min(0.20, preferred_ratio) * 0.8,
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
                        "preferred_distance": new_preferred_distance,
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
                    "connector_distance": new_connector_distance,
                    "preferred_distance": new_preferred_distance,
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
                int(state.get("preferred_distance", 0.0) / 50.0),
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

    # V20: once a closed loop has been accurately scored, keep it. Target
    # distance/gain and route quality are ranking signals rather than hard gates.
    valid_candidates = [
        (score, route_nodes, metrics)
        for score, route_nodes, metrics in accurately_scored
        if metrics
    ]

    if valid_candidates:
        for _, _, metrics in valid_candidates:
            metrics["relaxed_target_filtering"] = True
        diverse = select_diverse_accurate_candidates(
            G,
            valid_candidates,
            max_routes=MAX_ROUTE_OPTIONS,
            route_diversity=route_diversity,
        )
        if not diverse:
            diverse = sorted(valid_candidates, key=lambda item: item[0])[:MAX_ROUTE_OPTIONS]
        _, route_nodes, metrics = diverse[0]
        metrics = dict(metrics)
        metrics["_route_options_candidates"] = copy_internal_route_options(diverse)
        return route_nodes, metrics, last_depth, states_expanded

    budget_text = " Search budget was reached." if budget_reached else ""
    raise HTTPException(
        status_code=400,
        detail=(
            "Closed loops were generated, but none could be accurately scored."
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

def _route_edge_key_set(path):
    return {
        undirected_edge_key(path[i], path[i + 1])
        for i in range(len(path) - 1)
    }


def waypoint_path(S, source, target, edge_use_counts, leg_cache=None, cache_stats=None, bridge_edges=None):
    """
    A* leg with a safe baseline-path cache.

    V18 allows a normal second traversal of the same corridor, especially when
    that corridor is a bridge/stem. A third-and-later traversal is deliberately
    expensive so the search does not manufacture repeated laps or spaghetti.
    """
    key = (source, target)
    used_edge_keys = {
        edge_key for edge_key, count in (edge_use_counts or {}).items() if count > 0
    }

    if leg_cache is not None:
        cached = leg_cache.get(key)
        if cached is not None:
            cached_path, cached_edges = cached
            if not used_edge_keys.intersection(cached_edges):
                if cache_stats is not None:
                    cache_stats["hits"] = cache_stats.get("hits", 0) + 1
                return list(cached_path)
            if cache_stats is not None:
                cache_stats["blocked"] = cache_stats.get("blocked", 0) + 1

    if not used_edge_keys:
        path = fast_shortest_path(
            S,
            source,
            target,
            weight="routing_cost",
        )
        if leg_cache is not None and len(leg_cache) < WAYPOINT_LEG_CACHE_MAX:
            leg_cache[key] = (tuple(path), _route_edge_key_set(path))
        if cache_stats is not None:
            cache_stats["misses"] = cache_stats.get("misses", 0) + 1
        return path

    bridge_edges = bridge_edges if bridge_edges is not None else routing_bridge_edge_keys(S)

    path = fast_shortest_path(
        S,
        source,
        target,
        weight="routing_cost",
        edge_use_counts=edge_use_counts,
        bridge_edges=bridge_edges,
    )
    if cache_stats is not None:
        cache_stats["penalized_searches"] = cache_stats.get("penalized_searches", 0) + 1
    return path



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

    bridge_edges = routing_bridge_edge_keys(G)
    repeat_breakdown = repeated_edge_breakdown(G, route_nodes, bridge_edges)
    repeated_edges = repeat_breakdown["repeated_edges"]
    repeated_distance = repeat_breakdown["repeated_distance_meters"]
    effective_repeated_distance = repeat_breakdown["effective_repeated_distance_meters"]
    repeat_ratio = effective_repeated_distance / max(total_distance, 1.0)

    node_breakdown = repeated_node_breakdown(route_nodes, bridge_edges)
    repeated_nodes = node_breakdown["repeated_nodes"]
    effective_repeated_nodes = node_breakdown["effective_repeated_nodes"]

    reversal_breakdown = immediate_reversal_breakdown(route_nodes, bridge_edges)
    immediate_reversals = reversal_breakdown["immediate_reversals"]
    effective_immediate_reversals = reversal_breakdown["effective_immediate_reversals"]
    connector_distance = connector_distance_meters(G, route_nodes)
    connector_ratio = connector_distance / max(total_distance, 1.0)
    preferred_distance, preferred_hit_count = preferred_segment_metrics(G, route_nodes)
    preferred_ratio = preferred_distance / max(total_distance, 1.0)
    preference_reward = (
        preferred_hit_count * PREFERRED_SEGMENT_CHEAP_REWARD
        + min(0.20, preferred_ratio) * 35.0
    )

    topology = route_topology_metrics(route_nodes)
    max_radial_meters = route_max_radial_meters_from_nodes(G, route_nodes)
    shape_penalty, shape_metrics = big_loop_shape_penalty(
        target_distance_meters,
        topology,
        max_radial_meters,
        footprint_area_m2=None,
        cheap=True,
    )
    subloop_penalty, subloop_metrics = small_subloop_penalty(
        G,
        route_nodes,
        target_distance_meters,
        cheap=True,
    )
    connector_score_weight = connector_score_weight_for_target(
        target_distance_meters,
        cheap=True,
    )

    # Distance and approximate elevation are the primary exploratory goals.
    # V17 keeps the strong mini-loop shape penalty, but makes ordinary retracing
    # secondary and treats bridge/stem retracing as nearly unavoidable.
    score = (
        distance_ratio * 190.0
        + gain_ratio * 150.0
        + repeat_ratio * CHEAP_REPEAT_SCORE_WEIGHT
        + effective_repeated_nodes * CHEAP_REPEATED_NODE_WEIGHT
        + effective_immediate_reversals * CHEAP_IMMEDIATE_REVERSAL_WEIGHT
        + connector_ratio * connector_score_weight
        + shape_penalty
        + subloop_penalty
        - preference_reward
    )

    return score, {
        "total_distance_meters": total_distance,
        "approximate_gain_meters": approximate_gain,
        "repeated_edges": repeated_edges,
        "repeated_distance_meters": repeated_distance,
        "necessary_repeated_distance_meters": repeat_breakdown["necessary_repeated_distance_meters"],
        "optional_repeated_distance_meters": repeat_breakdown["optional_repeated_distance_meters"],
        "effective_repeated_distance_meters": effective_repeated_distance,
        "repeated_nodes": repeated_nodes,
        "necessary_repeated_nodes": node_breakdown["necessary_repeated_nodes"],
        "optional_repeated_nodes": node_breakdown["optional_repeated_nodes"],
        "immediate_reversals": immediate_reversals,
        "necessary_immediate_reversals": reversal_breakdown["necessary_immediate_reversals"],
        "optional_immediate_reversals": reversal_breakdown["optional_immediate_reversals"],
        "connector_distance_meters": connector_distance,
        "connector_ratio": connector_ratio,
        "preferred_distance_meters": preferred_distance,
        "preferred_hit_count": preferred_hit_count,
        "preference_reward": preference_reward,
        "cycle_rank": topology["cycle_rank"],
        "extra_cycles": topology["extra_cycles"],
        "branch_points": topology["branch_points"],
        "branch_excess": topology["branch_excess"],
        "max_radial_meters": max_radial_meters,
        "max_radial_ratio": shape_metrics["max_radial_ratio"],
        "shape_penalty": shape_metrics["shape_penalty"],
        "small_subloops": subloop_metrics["count"],
        "small_subloop_distance_meters": subloop_metrics["distance_meters"],
        "small_subloop_penalty": subloop_penalty,
    }


def generate_waypoint_loop(
    G,
    start_node,
    target_distance_meters,
    target_gain_meters,
    profile,
    limits,
    required_pass_points=None,
    route_diversity=DEFAULT_ROUTE_DIVERSITY,
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
    connector_multiplier = connector_path_multiplier_for_target(target_distance_meters)
    S = make_simple_routing_graph(G, connector_multiplier=connector_multiplier)
    bridge_edges = routing_bridge_edge_keys(S)

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

    required_pass_points = list(required_pass_points or [])
    required_groups = []
    for info in required_pass_points:
        group = [node for node in info.get("candidate_nodes", []) if node in reachable_nodes and node in S]
        if not group:
            raise HTTPException(
                status_code=400,
                detail=f"Required pass-through point {int(info.get('index', 0)) + 1} is near a trail, but that trail is not reachable from the selected start.",
            )
        required_groups.append(group)

    trail_nodes = set()
    for u, v, data in G.edges(data=True):
        if str(data.get("route_class", "trail")) == "trail":
            if u in reachable_nodes:
                trail_nodes.add(u)
            if v in reachable_nodes:
                trail_nodes.add(v)

    radial_by_node = {}
    for node in trail_nodes:
        if node == start_node or node not in S:
            continue

        radial = haversine_meters(
            start_lat,
            start_lon,
            float(G.nodes[node]["y"]),
            float(G.nodes[node]["x"]),
        )
        radial_by_node[node] = radial

        if (
            radial >= profile["min_anchor_distance_m"]
            and radial <= max_radial_distance
        ):
            candidates.append(node)

    far_anchor_min_m = float(profile.get("far_anchor_min_ratio", 0.0)) * target_distance_meters
    far_candidates = [
        node for node in candidates
        if radial_by_node.get(node, 0.0) >= far_anchor_min_m
    ]

    max_total_anchors = max(profile["anchor_counts"])
    max_optional_anchors = max(0, max_total_anchors - len(required_groups))
    if len(candidates) < max_optional_anchors:
        raise HTTPException(
            status_code=400,
            detail="Not enough trail junctions for this route and its required pass-through points.",
        )

    base_accurate_finalists = int(profile.get("accurate_finalists", 14))
    diversity = normalized_route_diversity(route_diversity)
    accurate_finalists = max(
        base_accurate_finalists,
        min(24, MAX_ROUTE_OPTIONS + int(round(10.0 * diversity / 100.0))),
    )
    pool_multiplier = int(profile.get("candidate_pool_multiplier", 3))
    pool_limit = max(accurate_finalists, accurate_finalists * pool_multiplier)

    # V15: one static shortest-path tree from the start replaces hundreds of
    # repeated start->first-anchor A* calls. Additional baseline legs are cached
    # lazily and reused only when they do not overlap already-used loop edges.
    leg_cache = {}
    leg_cache_stats = {"hits": 0, "misses": 0, "blocked": 0, "penalized_searches": 0}
    try:
        start_paths = fast_single_source_paths(
            S,
            start_node,
            weight="routing_cost",
        )
        reverse_S = S.reverse(copy=False)
        reverse_start_paths = fast_single_source_paths(
            reverse_S,
            start_node,
            weight="routing_cost",
        )
        for node in candidates:
            if len(leg_cache) >= WAYPOINT_LEG_CACHE_MAX:
                break

            path = start_paths.get(node)
            if path and len(path) >= 2:
                path_tuple = tuple(path)
                leg_cache[(start_node, node)] = (
                    path_tuple,
                    _route_edge_key_set(path_tuple),
                )

            reverse_path = reverse_start_paths.get(node)
            if reverse_path and len(reverse_path) >= 2:
                # reverse graph path start->node becomes original node->start
                to_start = tuple(reversed(reverse_path))
                leg_cache[(node, start_node)] = (
                    to_start,
                    _route_edge_key_set(to_start),
                )
    except Exception:
        pass

    exploratory = []
    seen_routes = set()

    for _ in range(profile["attempts"]):
        desired_anchor_count = random.choice(profile["anchor_counts"])

        # Pick one allowed trail node from every required pass-through zone.
        required_anchors = []
        for group in required_groups:
            # Prefer candidates closest to the user's marker while occasionally
            # exploring the rest of the accepted zone for route diversity.
            close_pool = group[: min(6, len(group))]
            pool = close_pool if random.random() < 0.8 else group
            required_anchors.append(random.choice(pool))

        # Remove duplicate required nodes (e.g. overlapping pass-through zones).
        required_anchors = list(dict.fromkeys(required_anchors))
        optional_count = max(0, desired_anchor_count - len(required_anchors))

        available = [node for node in candidates if node not in required_anchors]
        random.shuffle(available)
        optional_anchors = []

        # V18 outward bias: when the network actually has a reachable trail far
        # enough away, make at least one anchor use it. This prevents a 12-15 mi
        # request from solving everything with tightly clustered local anchors.
        required_already_far = any(
            radial_by_node.get(node, 0.0) >= far_anchor_min_m
            for node in required_anchors
        )
        far_anchor_probability = float(profile.get("far_anchor_attempt_probability", 0.0))
        if (
            optional_count > 0
            and far_candidates
            and not required_already_far
            and random.random() < far_anchor_probability
        ):
            far_pool = [node for node in far_candidates if node not in required_anchors]
            random.shuffle(far_pool)
            for candidate in far_pool:
                spacing_bad = False
                for existing in required_anchors:
                    separation = haversine_meters(
                        float(G.nodes[candidate]["y"]),
                        float(G.nodes[candidate]["x"]),
                        float(G.nodes[existing]["y"]),
                        float(G.nodes[existing]["x"]),
                    )
                    if separation < profile["min_anchor_separation_m"]:
                        spacing_bad = True
                        break
                if not spacing_bad:
                    optional_anchors.append(candidate)
                    break

        for candidate in available:
            if len(optional_anchors) >= optional_count:
                break
            if candidate in optional_anchors:
                continue
            # Keep optional anchors geographically separated. Required points are
            # hard constraints and are never rejected for being close together.
            spacing_bad = False
            for existing in optional_anchors + required_anchors:
                separation = haversine_meters(
                    float(G.nodes[candidate]["y"]),
                    float(G.nodes[candidate]["x"]),
                    float(G.nodes[existing]["y"]),
                    float(G.nodes[existing]["x"]),
                )
                if separation < profile["min_anchor_separation_m"]:
                    spacing_bad = True
                    break
            if not spacing_bad:
                optional_anchors.append(candidate)

        # V20: anchor spacing is a preference, not a hard rejection. If the
        # trail network cannot satisfy the preferred spacing, fill remaining
        # anchor slots from any reachable candidates and let scoring rank the
        # resulting route instead of discarding the attempt.
        if len(optional_anchors) < optional_count:
            fallback = [
                node for node in available
                if node not in optional_anchors and node not in required_anchors
            ]
            for candidate in fallback:
                optional_anchors.append(candidate)
                if len(optional_anchors) >= optional_count:
                    break

        if len(optional_anchors) < optional_count:
            continue

        anchors = required_anchors + optional_anchors

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
        edge_use_counts = {}
        current = start_node
        failed = False

        for destination in anchors + [start_node]:
            try:
                leg = waypoint_path(
                    S,
                    current,
                    destination,
                    edge_use_counts,
                    leg_cache=leg_cache,
                    cache_stats=leg_cache_stats,
                    bridge_edges=bridge_edges,
                )
            except nx.NetworkXNoPath:
                failed = True
                break

            for i in range(len(leg) - 1):
                edge_key = undirected_edge_key(leg[i], leg[i + 1])
                edge_use_counts[edge_key] = edge_use_counts.get(edge_key, 0) + 1

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

        # V20: do not reject a valid loop just because it misses the requested
        # distance during cheap exploration. Distance, gain, footprint, repeat,
        # connector use, and loop shape are ranking signals only.

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

    valid_candidates = []
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

        pass_metrics = required_pass_metrics_for_coords(
            metrics.get("route_coordinates", []),
            required_pass_points,
        )
        metrics["required_pass_points"] = pass_metrics
        if pass_metrics and not all(item.get("satisfied") for item in pass_metrics):
            # Explicit user pass-through points remain a hard requirement.
            continue

        metrics["waypoint_attempts"] = profile["attempts"]
        metrics["waypoint_unique_candidates"] = len(exploratory)
        metrics["waypoint_accurate_finalists"] = accurately_scored
        metrics["waypoint_cheap_score"] = cheap_score
        metrics["waypoint_leg_cache_hits"] = int(leg_cache_stats.get("hits", 0))
        metrics["waypoint_penalized_searches"] = int(leg_cache_stats.get("penalized_searches", 0))
        metrics["relaxed_target_filtering"] = True

        # V20: every accurately scored, routable candidate that satisfies any
        # explicit pass-through zones survives. Numerical target mismatch and
        # route-shape preferences only affect ranking.
        valid_candidates.append((score, route, metrics))

    if valid_candidates:
        diverse = select_diverse_accurate_candidates(
            G,
            valid_candidates,
            max_routes=MAX_ROUTE_OPTIONS,
            route_diversity=route_diversity,
        )
        if not diverse:
            diverse = sorted(valid_candidates, key=lambda item: item[0])[:MAX_ROUTE_OPTIONS]

        _, best_route, best_metrics = diverse[0]
        best_metrics = dict(best_metrics)
        best_metrics["waypoint_accurate_finalists"] = accurately_scored
        best_metrics["_route_options_candidates"] = copy_internal_route_options(diverse)
        return (
            best_route,
            best_metrics,
            profile["attempts"],
        )

    raise HTTPException(
        status_code=400,
        detail=(
            "No routable waypoint finalist survived. "
            "If required pass-through points are set, try increasing their tolerance."
        ),
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
    return_distance = fast_single_source_lengths(
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
def trail_overlay(
    west: float = Query(...),
    south: float = Query(...),
    east: float = Query(...),
    north: float = Query(...),
):
    """Return lightweight overlay tile URLs for the current viewport."""
    if not all(math.isfinite(v) for v in (west, south, east, north)):
        raise HTTPException(status_code=400, detail="Invalid map bounds.")
    if east <= west or north <= south:
        raise HTTPException(status_code=400, detail="Invalid map bounds order.")
    return overlay_index_payload(west, south, east, north)


@app.get("/overlay-tile/{tile_id}")
def overlay_tile(tile_id: str):
    """Serve a prebuilt gzip JSON overlay tile without loading NetworkX."""
    path = overlay_tile_file(tile_id)
    with open(path, "rb") as f:
        data = f.read()
    return Response(
        content=data,
        media_type="application/json",
        headers={
            "Content-Encoding": "gzip",
            "Cache-Control": "public, max-age=86400",
        },
    )


@app.post("/trail-network")
def trail_network(request: TrailNetworkRequest):
    try:
        profile = get_route_profile(request.target_distance_miles)

        workspace, workspace_from_cache = get_start_workspace(
            request.start_lat,
            request.start_lon,
            requested_radius_m=float(profile["search_radius_m"]),
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
            "master_graph_file": master_info.get("saved_graph", os.path.basename(MASTER_ROUTING_GRAPHML_PATH)),
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
    _ensure_data_file_downloaded(DEM_PATH, "DEM_TIF_URL")

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
# V29 EXPLICIT ROUTE-SECTION REPLACEMENT
# ============================================================

def _coordinate_route_signature(coords):
    payload = "|".join(
        f"{float(point['lat']):.6f},{float(point['lon']):.6f}"
        for point in coords
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _sample_section_coords(coords, max_samples=28):
    coords = list(coords or [])
    if len(coords) <= 2:
        return coords
    # Skip the exact cut endpoints so the replacement can still leave/join the
    # original route cleanly even when a cut lies in the middle of one OSM edge.
    interior = coords[1:-1]
    if len(interior) <= max_samples:
        return interior
    return [
        interior[int(round(i * (len(interior) - 1) / max(1, max_samples - 1)))]
        for i in range(max_samples)
    ]


def penalize_replaced_section_edges(G, section_coords, multiplier=100000.0, exempt_physical_keys=None):
    """Make the old orange corridor effectively unavailable during replacement.

    We retain the edges instead of deleting them so a cut made in the middle of
    an OSM edge can still attach cleanly, but the huge cost means a replacement
    will not silently fall back to the old section when another connected path
    exists.
    """
    exempt_physical_keys = set(exempt_physical_keys or [])
    matches = set()
    for point in _sample_section_coords(section_coords):
        match = _nearest_natural_edge_to_selected_point(
            G,
            float(point["lat"]),
            float(point["lon"]),
            max_distance_m=max(TRAIL_SEGMENT_SELECTION_MAX_DISTANCE_M, 55.0),
        )
        if match is not None:
            matches.add(match["physical_key"])

    if not matches:
        return G, 0

    H = G.copy()
    touched = 0
    for u, v, key, data in H.edges(keys=True, data=True):
        physical_key = undirected_edge_key(u, v)
        if physical_key not in matches or physical_key in exempt_physical_keys:
            continue
        base = float(data.get("routing_cost", edge_routing_cost(data)) or 0.0)
        length = float(data.get("length", 0.0) or 0.0)
        data["routing_cost"] = max(base, length, 0.01) * float(multiplier)
        data["replaced_section_penalty"] = True
        touched += 1
    return H, touched


def _append_route_coords(target, incoming):
    """Append route coordinates without introducing duplicate join points."""
    for point in incoming or []:
        item = {"lat": float(point["lat"]), "lon": float(point["lon"])}
        if target:
            last = target[-1]
            if haversine_meters(
                float(last["lat"]), float(last["lon"]),
                float(item["lat"]), float(item["lon"]),
            ) <= 0.08:
                target[-1] = item
                continue
        target.append(item)
    return target



def _resolve_exact_overlay_edge(G, selected, selected_index=None):
    """Resolve a green selection by tile/u/v/key instead of nearest-edge guessing."""
    u = getattr(selected, "edge_u", None)
    v = getattr(selected, "edge_v", None)
    wanted_key = getattr(selected, "edge_key", None)
    label = (
        f"Replacement trail segment {selected_index + 1}"
        if selected_index is not None
        else "Replacement trail segment"
    )

    if u is None or v is None or wanted_key is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{label} does not contain an exact edge ID. "
                "Rebuild overlay_tiles using build_overlay_tiles_v41.py, refresh the browser, "
                "then start the replacement selection again."
            ),
        )

    u = int(u)
    v = int(v)
    wanted_key = str(wanted_key)

    candidates = []
    for a, b in ((u, v), (v, u)):
        bundle = G.get_edge_data(a, b) or {}
        for key, data in bundle.items():
            if str(key) != wanted_key:
                continue
            if str(data.get("route_class", "trail")) != "trail":
                continue
            coords = oriented_edge_coords(G, a, b, data)
            if len(coords) >= 2:
                candidates.append((a, b, key, data, coords))

    # Exact cut-point insertion can split the first/last chosen edge. In that
    # case, find the split child with the same original edge key and geometry.
    if not candidates:
        raw_geometry = [
            (float(point[1]), float(point[0]))
            for point in (selected.geometry or [])
            if isinstance(point, (list, tuple)) and len(point) >= 2
        ]
        if len(raw_geometry) >= 2:
            mid_lon, mid_lat = raw_geometry[len(raw_geometry) // 2]
            split_candidates = []
            for a, b, key, data in G.edges(keys=True, data=True):
                if str(key) != wanted_key:
                    continue
                if str(data.get("route_class", "trail")) != "trail":
                    continue
                if (
                    a not in (u, v)
                    and b not in (u, v)
                    and not bool(data.get("virtual_split_edge", False))
                ):
                    continue
                coords = oriented_edge_coords(G, a, b, data)
                nearest = nearest_position_on_polyline(coords, mid_lon, mid_lat)
                if nearest is None:
                    continue
                split_candidates.append(
                    (float(nearest["distance_m"]), a, b, key, data, coords)
                )
            split_candidates.sort(key=lambda item: item[0])
            if split_candidates and split_candidates[0][0] <= 8.0:
                _, a, b, key, data, coords = split_candidates[0]
                candidates.append((a, b, key, data, coords))

    if not candidates:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{label} ({getattr(selected, 'tile_id', '')}:{u}->{v}:{wanted_key}) "
                "is not present in this routing workspace. Refresh the gray trails "
                "and select the replacement corridor again."
            ),
        )

    # If both directions exist, pick the copy nearest the exact browser click.
    ranked = []
    for a, b, key, data, coords in candidates:
        nearest = nearest_position_on_polyline(
            coords, float(selected.lon), float(selected.lat)
        )
        distance = float(nearest["distance_m"]) if nearest else 1e12
        ranked.append((distance, a, b, key, data, coords))
    ranked.sort(key=lambda item: item[0])
    distance, a, b, key, data, coords = ranked[0]

    return {
        "u": a,
        "v": b,
        "key": key,
        "physical_key": undirected_edge_key(a, b),
        "distance_m": distance,
        "routing_lat": float(selected.lat),
        "routing_lon": float(selected.lon),
        "tile_id": getattr(selected, "tile_id", None),
        "selected_geometry": [
            [float(point[0]), float(point[1])]
            for point in (selected.geometry or [])
            if isinstance(point, (list, tuple)) and len(point) >= 2
        ],
    }


def _exact_directed_edge_choice(G, required, enter, leave):
    """Return the exact graph edge that corresponds to the selected green trail.

    V33 forced only the selected edge's node pair through a simplified graph.
    When parallel/simplified OSM edges shared the same nodes, geometry
    reconstruction could silently draw a different trail.  V34 keeps the exact
    MultiDiGraph edge key and geometry for every green selection.
    """
    preferred_key = required.get("key")
    candidates = []
    edge_bundle = G.get_edge_data(enter, leave) or {}
    for key, data in edge_bundle.items():
        if str(data.get("route_class", "trail")) != "trail":
            continue
        key_bonus = 0 if preferred_key is not None and str(key) == str(preferred_key) else 1
        coords = oriented_edge_coords(G, enter, leave, data)
        if len(coords) < 2:
            continue
        # Favor the exact key first, then the edge whose geometry passes closest
        # to the browser-selected snap point.
        snap_lon = float(required.get("routing_lon", G.nodes[enter].get("x", 0.0)))
        snap_lat = float(required.get("routing_lat", G.nodes[enter].get("y", 0.0)))
        nearest = nearest_position_on_polyline(coords, snap_lon, snap_lat)
        distance = float(nearest["distance_m"]) if nearest else 1e12
        candidates.append((key_bonus, distance, key, data, coords))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    _, _, key, data, coords = candidates[0]
    return key, data, coords


def _selected_overlay_geometry_for_orientation(G, required, enter, leave):
    """Return the browser-highlighted gray/green geometry oriented enter -> leave.

    The gray overlay is generated directly from the master trail graph, so the
    geometry sent back by the browser is the exact visual trail piece the user
    selected.  Using it here prevents the replacement from silently drawing a
    parallel/simplified edge that merely shares the same graph endpoints.
    """
    raw = required.get("selected_geometry") or []
    coords = []
    for point in raw:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        coords.append({"lat": float(point[0]), "lon": float(point[1])})

    if len(coords) < 2:
        exact = _exact_directed_edge_choice(G, required, enter, leave)
        if exact is None:
            return []
        _, _, edge_coords_lonlat = exact
        return [
            {"lat": float(lat), "lon": float(lon)}
            for lon, lat in edge_coords_lonlat
        ]

    enter_lat = float(G.nodes[enter]["y"])
    enter_lon = float(G.nodes[enter]["x"])
    first_d = haversine_meters(enter_lat, enter_lon, coords[0]["lat"], coords[0]["lon"])
    last_d = haversine_meters(enter_lat, enter_lon, coords[-1]["lat"], coords[-1]["lon"])
    if last_d < first_d:
        coords.reverse()
    return coords


def shortest_path_forcing_selected_edges_exact(G, S, start_node, end_node, required_edges):
    """Build a deterministic replacement through the selected green corridor.

    V34 used a global orientation/cost search.  Although it technically forced
    selected edges, it could choose surprising approaches/orientations and make
    the visible result look almost identical to the original route.  V35 treats
    the clicked green pieces as an ordered corridor:

      cut start -> green 1 -> green 2 -> ... -> green N -> cut end

    Every green geometry is appended verbatim.  Shortest-path routing is used
    only for the small gaps between those exact pieces and for the two endpoint
    connections.  This matches the visual editing model in the browser.
    """
    if not required_edges:
        nodes = fast_shortest_path(S, start_node, end_node, weight="routing_cost")
        return nodes, route_coordinates(G, nodes)

    current = start_node
    node_path = [start_node]
    coord_path = [
        {"lat": float(G.nodes[start_node]["y"]), "lon": float(G.nodes[start_node]["x"])}
    ]

    for required in required_edges:
        u = required["u"]
        v = required["v"]
        candidates = []

        for enter, leave in ((u, v), (v, u)):
            if not S.has_edge(enter, leave):
                continue
            exact = _exact_directed_edge_choice(G, required, enter, leave)
            if exact is None:
                continue
            try:
                leg_nodes = fast_shortest_path(S, current, enter, weight="routing_cost")
                leg_cost = (
                    nx.path_weight(S, leg_nodes, weight="routing_cost")
                    if len(leg_nodes) > 1 else 0.0
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

            # Strongly prefer actual endpoint continuity.  If two consecutively
            # selected gray trail pieces share a graph node, this makes that
            # zero-gap connection win instead of taking an unrelated detour.
            continuity_bonus = 0.0 if current == enter else 0.001
            candidates.append((float(leg_cost) + continuity_bonus, enter, leave, leg_nodes))

        if not candidates:
            raise nx.NetworkXNoPath("Selected replacement trail edge is not traversable.")

        candidates.sort(key=lambda item: item[0])
        _, enter, leave, leg_nodes = candidates[0]

        if len(leg_nodes) > 1:
            node_path.extend(leg_nodes[1:])
            _append_route_coords(coord_path, route_coordinates(G, leg_nodes)[1:])

        selected_coords = _selected_overlay_geometry_for_orientation(G, required, enter, leave)
        if len(selected_coords) < 2:
            raise nx.NetworkXNoPath("Could not preserve a selected green trail geometry.")
        _append_route_coords(coord_path, selected_coords)

        if not node_path or node_path[-1] != enter:
            node_path.append(enter)
        if node_path[-1] != leave:
            node_path.append(leave)
        current = leave

    try:
        leg_nodes = fast_shortest_path(S, current, end_node, weight="routing_cost")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        raise nx.NetworkXNoPath("Replacement corridor cannot reconnect to the route.")

    if len(leg_nodes) > 1:
        node_path.extend(leg_nodes[1:])
        _append_route_coords(coord_path, route_coordinates(G, leg_nodes)[1:])

    return node_path, coord_path

def classify_route_geometry_against_graph(G, coords):
    """Estimate trail/connector/retrace stats for a stitched edited geometry.

    Each displayed geometry segment is associated with its nearest routing edge.
    Consecutive pieces on the same physical OSM edge are merged into one run so
    a single traversal is not mistakenly counted as retracing.
    """
    if len(coords) < 2:
        return {
            "connector_distance_meters": 0.0,
            "repeated_distance_meters": 0.0,
            "repeated_edges": 0,
            "preferred_hit_count": 0,
            "preferred_distance_meters": 0.0,
        }

    mids_lon = []
    mids_lat = []
    lengths = []
    for a, b in zip(coords[:-1], coords[1:]):
        lat1, lon1 = float(a["lat"]), float(a["lon"])
        lat2, lon2 = float(b["lat"]), float(b["lon"])
        length = haversine_meters(lat1, lon1, lat2, lon2)
        if length <= 0.02:
            continue
        mids_lat.append((lat1 + lat2) / 2.0)
        mids_lon.append((lon1 + lon2) / 2.0)
        lengths.append(length)

    if not lengths:
        return {
            "connector_distance_meters": 0.0,
            "repeated_distance_meters": 0.0,
            "repeated_edges": 0,
            "preferred_hit_count": 0,
            "preferred_distance_meters": 0.0,
        }

    try:
        projected = ox.projection.project_graph(G)
        transformer = Transformer.from_crs(
            "EPSG:4326", projected.graph["crs"], always_xy=True
        )
        xs, ys = transformer.transform(mids_lon, mids_lat)
        raw_edges = ox.distance.nearest_edges(
            projected,
            X=np.asarray(xs, dtype=float),
            Y=np.asarray(ys, dtype=float),
        )

        if hasattr(raw_edges, "tolist"):
            raw_edges = raw_edges.tolist()
        if isinstance(raw_edges, tuple) and len(raw_edges) >= 3 and all(
            np.isscalar(value) for value in raw_edges[:3]
        ):
            edge_ids = [tuple(raw_edges[:3])]
        else:
            edge_ids = []
            for item in list(raw_edges):
                if isinstance(item, np.ndarray):
                    item = item.tolist()
                if isinstance(item, (list, tuple)) and len(item) >= 3:
                    edge_ids.append((item[0], item[1], item[2]))

        if len(edge_ids) != len(lengths):
            raise ValueError("nearest-edge result length mismatch")
    except Exception:
        # Editing should still succeed if diagnostic classification fails.
        return {
            "connector_distance_meters": 0.0,
            "repeated_distance_meters": 0.0,
            "repeated_edges": 0,
            "preferred_hit_count": 0,
            "preferred_distance_meters": 0.0,
        }

    runs = []
    connector_distance = 0.0
    preferred_distance = 0.0
    preferred_keys = set()

    for edge_id, segment_length in zip(edge_ids, lengths):
        u, v, key = edge_id
        data = projected.get_edge_data(u, v, key) or {}
        physical = undirected_edge_key(u, v)
        route_class = str(data.get("route_class", "trail"))
        preferred = bool(data.get("preferred_segment", False))

        if route_class == "connector":
            connector_distance += segment_length
        if preferred:
            preferred_distance += segment_length
            preferred_keys.add(physical)

        if runs and runs[-1]["physical"] == physical:
            runs[-1]["length"] += segment_length
        else:
            runs.append({"physical": physical, "length": float(segment_length)})

    use_counts = {}
    repeated_distance = 0.0
    repeated_edges = 0
    for run in runs:
        physical = run["physical"]
        count = use_counts.get(physical, 0)
        if count >= 1:
            repeated_distance += run["length"]
            repeated_edges += 1
        use_counts[physical] = count + 1

    return {
        "connector_distance_meters": float(connector_distance),
        "repeated_distance_meters": float(repeated_distance),
        "repeated_edges": int(repeated_edges),
        "preferred_hit_count": len(preferred_keys),
        "preferred_distance_meters": float(preferred_distance),
    }


def build_stitched_route_option(G, coords, request, replacement_route_nodes):
    if replacement_route_nodes:
        geometry = route_baked_elevation_metrics(G, replacement_route_nodes)
    else:
        # Direct-splice edits may not have exact node IDs for every coordinate.
        # Classify against the graph and use nearest edge baked elevations only
        # when available; distance remains exact from coords.
        geometry = coordinate_distance_only_metrics(coords)
    total_m = float(geometry["distance_meters"])
    gain_m = float(geometry["gain_meters"])
    classification = classify_route_geometry_against_graph(G, coords)

    repeated_m = float(classification["repeated_distance_meters"])
    connector_m = float(classification["connector_distance_meters"])
    retrace_percent = 100.0 * repeated_m / max(total_m, 1.0)
    connector_percent = 100.0 * connector_m / max(total_m, 1.0)
    trail_percent = max(0.0, 100.0 - connector_percent)
    distance_miles = total_m / METERS_PER_MILE
    gain_ft = gain_m * FEET_PER_METER
    descent_ft = float(geometry.get("descent_meters", 0.0)) * FEET_PER_METER
    max_reach_m = route_max_radial_meters_from_coords(coords)
    footprint_m2 = route_convex_hull_area_m2(coords)

    distance_ratio = abs(total_m - request.target_distance_miles * METERS_PER_MILE) / max(
        request.target_distance_miles * METERS_PER_MILE, 1.0
    )
    target_gain_m = request.target_gain_ft / FEET_PER_METER
    if target_gain_m > 0:
        gain_ratio = abs(gain_m - target_gain_m) / target_gain_m
    else:
        gain_ratio = gain_m / 30.48
    route_score_value = (
        distance_ratio * 190.0
        + gain_ratio * 240.0
        + (repeated_m / max(total_m, 1.0)) * LONG_REPEAT_SCORE_WEIGHT
        + (connector_m / max(total_m, 1.0)) * connector_score_weight_for_target(
            request.target_distance_miles * METERS_PER_MILE,
            cheap=False,
        )
    )

    return {
        "index": 0,
        "name": "Edited route",
        "route_signature": _coordinate_route_signature(coords),
        "actual_distance_miles": round(distance_miles, 2),
        "distance_error_miles": round(abs(distance_miles - request.target_distance_miles), 2),
        "actual_gain_ft": round(gain_ft),
        "actual_descent_ft": round(descent_ft),
        "elevation_error_ft": round(abs(gain_ft - request.target_gain_ft)),
        "route": coords,
        "gpx_export_points": build_gpx_export_points(coords),
        "elevation_profile": geometry.get("elevation_profile", []),
        "route_nodes": len(replacement_route_nodes),
        "route_geometry_points": len(coords),
        "repeated_edges": int(classification["repeated_edges"]),
        "repeated_distance_miles": round(repeated_m / METERS_PER_MILE, 2),
        "repeated_nodes": 0,
        "immediate_reversals": 0,
        "connector_distance_miles": round(connector_m / METERS_PER_MILE, 2),
        "retrace_percent": round(retrace_percent, 1),
        "connector_percent": round(connector_percent, 1),
        "trail_percent": round(trail_percent, 1),
        "preferred_distance_miles": round(
            classification["preferred_distance_meters"] / METERS_PER_MILE, 2
        ),
        "preferred_hit_count": int(classification["preferred_hit_count"]),
        "independent_loops": 0,
        "extra_subloops": 0,
        "branch_points": 0,
        "max_reach_miles": round(max_reach_m / METERS_PER_MILE, 2),
        "footprint_sq_miles": round(footprint_m2 / (METERS_PER_MILE ** 2), 2),
        "shape_penalty": 0.0,
        "route_score": round(route_score_value, 2),
        "partial_edge_used": False,
        "partial_added_distance_miles": 0.0,
        "partial_outward_distance_meters": 0.0,
        "required_pass_points": [],
        "is_edited": True,
        "edit_type": "section-replacement",
    }



@app.post("/recalculate-edited-route")
def recalculate_edited_route(request: RouteRecalculateRequest):
    """Recalculate distance/elevation/route stats after the browser performs an exact splice.

    V42 deliberately separates editing from routing. The browser changes the red
    line immediately using the exact highlighted green geometries. This endpoint
    only refreshes metrics/elevation afterward; it never changes the edited line.
    """
    try:
        coords = [
            {"lat": float(point.lat), "lon": float(point.lon)}
            for point in request.current_route
        ]
        if len(coords) < 2:
            raise HTTPException(status_code=400, detail="Edited route is empty.")

        farthest_m = 0.0
        for point in coords:
            farthest_m = max(
                farthest_m,
                haversine_meters(
                    float(request.start_lat),
                    float(request.start_lon),
                    float(point["lat"]),
                    float(point["lon"]),
                ),
            )

        requested_radius_m = max(
            2.0 * METERS_PER_MILE,
            farthest_m + 1.0 * METERS_PER_MILE,
        )

        workspace, _ = get_start_workspace(
            float(request.start_lat),
            float(request.start_lon),
            requested_radius_m=requested_radius_m,
            force_rebuild=False,
        )
        G = workspace["graph"].copy()
        workspace_start_node = workspace["start_node"]

        G, _, _ = apply_trail_segment_controls(
            G,
            request.avoid_segments,
            request.prefer_segments,
            start_node=workspace_start_node,
        )
        G, _ = apply_avoid_areas_to_graph(
            G,
            request.avoid_areas,
            start_node=workspace_start_node,
            start_lat=request.start_lat,
            start_lon=request.start_lon,
        )

        option = build_stitched_route_option(G, coords, request, [])
        option["is_edited"] = True
        option["edit_type"] = "section-replacement-direct-splice"
        return {"option": option, "version": APP_VERSION}

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Edited route metric refresh failed: {exc}",
        )


@app.post("/replace-route-section")
def replace_route_section(request: RouteSectionReplacementRequest):
    try:
        route = [
            {"lat": float(point.lat), "lon": float(point.lon)}
            for point in request.current_route
        ]
        if len(route) < 4:
            raise HTTPException(status_code=400, detail="The selected route is too short to edit.")
        if not request.replacement_segments:
            raise HTTPException(
                status_code=400,
                detail="Select at least one green replacement trail segment before applying the edit.",
            )

        selected_start_index = int(request.cut_start_index)
        selected_end_index = int(request.cut_end_index)
        cut_a = min(selected_start_index, selected_end_index)
        cut_b = max(selected_start_index, selected_end_index)
        selection_reversed = selected_start_index > selected_end_index
        if cut_a < 0 or cut_b >= len(route) or cut_b - cut_a < 2:
            raise HTTPException(
                status_code=400,
                detail="Choose two separated points on the red route for the section to replace.",
            )

        # Use the full cached TIFF workspace for deliberate edits. A user may
        # choose a replacement corridor outside the normal target-radius slice.
        edit_profile = get_route_profile(float(request.target_distance_miles))
        replacement_reach_m = 0.0
        for selected in request.replacement_segments:
            replacement_reach_m = max(
                replacement_reach_m,
                haversine_meters(
                    float(request.start_lat),
                    float(request.start_lon),
                    float(selected.lat),
                    float(selected.lon),
                ),
            )
        edit_radius_m = max(
            float(edit_profile["search_radius_m"]),
            replacement_reach_m + 1.25 * METERS_PER_MILE,
        )
        workspace, _ = get_start_workspace(
            float(request.start_lat),
            float(request.start_lon),
            requested_radius_m=edit_radius_m,
            force_rebuild=False,
        )
        G = workspace["graph"].copy()
        workspace_start_node = workspace["start_node"]

        G, _, _ = apply_trail_segment_controls(
            G,
            request.avoid_segments,
            request.prefer_segments,
            start_node=workspace_start_node,
        )
        G, _ = apply_avoid_areas_to_graph(
            G,
            request.avoid_areas,
            start_node=workspace_start_node,
            start_lat=request.start_lat,
            start_lon=request.start_lon,
        )

        selected_start = route[selected_start_index]
        selected_end = route[selected_end_index]
        G, selected_start_node, cut_start_info = insert_exact_routing_point(
            G, selected_start["lat"], selected_start["lon"]
        )
        G, selected_end_node, cut_end_info = insert_exact_routing_point(
            G, selected_end["lat"], selected_end["lon"]
        )

        if float(cut_start_info.get("routing_offset_m", 9999)) > 20 or float(
            cut_end_info.get("routing_offset_m", 9999)
        ) > 20:
            raise HTTPException(
                status_code=400,
                detail="Could not anchor the selected orange section cleanly to the trail graph. Zoom in and choose the cut points again.",
            )

        # Resolve each green selection to the actual natural-trail edge.  V31
        # treated these as loose waypoint points, which meant the shortest path
        # could touch a green segment and then continue somewhere else.  V32
        # forces traversal of every selected edge, in click order.
        required_edges = []
        resolved_segments = []
        for index, selected in enumerate(request.replacement_segments):
            match = _resolve_exact_overlay_edge(G, selected, selected_index=index)
            required_edges.append(match)
            resolved_segments.append({
                "index": index,
                "tile_id": match.get("tile_id"),
                "lat": float(match["routing_lat"]),
                "lon": float(match["routing_lon"]),
                "snap_distance_m": round(float(match["distance_m"]), 2),
                "edge_u": int(match["u"]),
                "edge_v": int(match["v"]),
                "edge_key": str(match["key"]),
            })

        # Make the old orange corridor effectively unavailable, but NEVER
        # penalize one of the green trail pieces the user explicitly selected.
        # The older code could accidentally penalize a nearby green segment when
        # the orange and green corridors ran close together.
        selected_physical_keys = {
            item["physical_key"] for item in required_edges if item.get("physical_key") is not None
        }
        G, penalized_edge_count = penalize_replaced_section_edges(
            G,
            route[cut_a:cut_b + 1],
            exempt_physical_keys=selected_physical_keys,
        )

        # Re-resolve against the penalized graph, but carry the browser's exact
        # highlighted geometry forward. This is what gets spliced into the route.
        forced_edges = []
        for index, (selected, original_match) in enumerate(
            zip(request.replacement_segments, required_edges)
        ):
            match = _resolve_exact_overlay_edge(G, selected, selected_index=index)
            match["selected_geometry"] = list(original_match.get("selected_geometry") or [])
            forced_edges.append(match)

        connector_multiplier = connector_path_multiplier_for_target(
            request.target_distance_miles * METERS_PER_MILE
        )
        S = make_simple_routing_graph(G, connector_multiplier=connector_multiplier)
        try:
            new_path_nodes, replacement_coords = shortest_path_forcing_selected_edges_exact(
                G, S, selected_start_node, selected_end_node, forced_edges
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            raise HTTPException(
                status_code=400,
                detail="The green trail pieces cannot form one connected replacement between the two red-route cut points. Start the selection again and choose a connected corridor.",
            )

        if len(replacement_coords) < 2:
            raise HTTPException(status_code=400, detail="The replacement corridor produced an empty path.")

        # Preserve everything outside the orange section exactly. The new graph
        # path only replaces route[cut_a:cut_b].
        replacement_coords[0] = dict(selected_start)
        replacement_coords[-1] = dict(selected_end)

        # The graph path follows the user's click order (start -> guide clicks ->
        # rejoin). The stored route geometry is stitched in ascending route-index
        # order, so reverse only the replacement geometry when the user happened
        # to start from the higher-index side of the orange section.
        if selection_reversed:
            replacement_coords = list(reversed(replacement_coords))
        replacement_coords[0] = dict(route[cut_a])
        replacement_coords[-1] = dict(route[cut_b])

        stitched = [dict(point) for point in route[:cut_a + 1]]
        for point in replacement_coords[1:]:
            if stitched:
                last = stitched[-1]
                if haversine_meters(
                    float(last["lat"]), float(last["lon"]),
                    float(point["lat"]), float(point["lon"]),
                ) <= 0.08:
                    stitched[-1] = dict(point)
                    continue
            stitched.append(dict(point))
        for point in route[cut_b + 1:]:
            if stitched:
                last = stitched[-1]
                if haversine_meters(
                    float(last["lat"]), float(last["lon"]),
                    float(point["lat"]), float(point["lon"]),
                ) <= 0.08:
                    stitched[-1] = dict(point)
                    continue
            stitched.append(dict(point))

        # A replacement operation must visibly change the selected route. If the
        # stitched geometry is still effectively identical, fail loudly instead
        # of telling the browser that the edit succeeded.
        old_signature = _coordinate_route_signature(route)
        new_signature = _coordinate_route_signature(stitched)
        if old_signature == new_signature:
            raise HTTPException(
                status_code=409,
                detail="The selected green corridor produced the same route. Select trail pieces that leave the orange section before reconnecting.",
            )

        option = build_stitched_route_option(G, stitched, request, new_path_nodes)
        option["replacement_distance_miles"] = round(
            coordinate_distance_only_metrics(replacement_coords)["distance_meters"]
            / METERS_PER_MILE,
            2,
        )
        option["replaced_route_points"] = int(cut_b - cut_a + 1)

        return {
            "option": option,
            "cut_start_index": cut_a,
            "cut_end_index": cut_b,
            "resolved_replacement_segments": resolved_segments,
            "penalized_old_section_edges": int(penalized_edge_count),
            "version": APP_VERSION,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Section replacement failed: {exc}")


# ============================================================
# GENERATE ROUTE
# ============================================================

@app.post("/generate-route")
def generate_route(request: RouteRequest):
    _v46_total_started = time.perf_counter()
    _v46_timings = {}

    def _v46_mark(name, started):
        _v46_timings[name] = _v46_timings.get(name, 0.0) + (
            time.perf_counter() - started
        )

    def _v46_report():
        total = time.perf_counter() - _v46_total_started
        measured = sum(_v46_timings.values())
        print("")
        print("=" * 72)
        print(
            f"V46 TIMING — {float(request.target_distance_miles):.1f} mi / "
            f"{float(request.target_gain_ft):.0f} ft"
        )
        print("=" * 72)
        for name, seconds in sorted(
            _v46_timings.items(),
            key=lambda item: item[1],
            reverse=True,
        ):
            pct = 100.0 * seconds / total if total else 0.0
            print(f"{name:38s} {seconds:8.2f}s  {pct:6.1f}%")
        other = max(0.0, total - measured)
        print(
            f"{'Other / orchestration':38s} {other:8.2f}s  "
            f"{(100.0 * other / total if total else 0.0):6.1f}%"
        )
        print("-" * 72)
        print(f"{'TOTAL':38s} {total:8.2f}s")
        print("=" * 72)
        print("")
        return {
            "total_seconds": round(total, 3),
            "sections": {
                key: round(value, 3)
                for key, value in _v46_timings.items()
            },
            "other_seconds": round(other, 3),
        }

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

        # Find More sends a new seed while keeping every other request setting
        # unchanged. Waypoint sampling and the short-route beam jitter then
        # explore a different batch of solutions.
        if request.search_seed is not None:
            random.seed(int(request.search_seed))

        route_diversity = normalized_route_diversity(request.route_diversity)

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

        _v46_t = time.perf_counter()
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
        _v46_mark("Workspace / tile loading", _v46_t)

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

        snapped_start_lat = float(G.nodes[start_node]["y"])
        snapped_start_lon = float(G.nodes[start_node]["x"])
        snap_distance_m = float(start_info["routing_offset_m"])

        # V27: clicked trail-segment controls are resolved against this exact
        # request graph. Avoided segments are removed; preferred segments get a
        # soft path/scoring bonus. This happens before pass-through edge splits
        # so preference attributes propagate into any temporary split pieces.
        _v46_t = time.perf_counter()
        G, resolved_avoid_segments, resolved_prefer_segments = apply_trail_segment_controls(
            G,
            request.avoid_segments,
            request.prefer_segments,
            start_node=start_node,
        )
        _v46_mark("Trail preference / avoid controls", _v46_t)

        # V25: avoid areas are hard exclusions. Remove intersecting routing
        # edges before selecting the end node or inserting required pass-through
        # points, so every returned candidate automatically respects them.
        _v46_t = time.perf_counter()
        G, resolved_avoid_areas = apply_avoid_areas_to_graph(
            G,
            request.avoid_areas,
            start_node=start_node,
            start_lat=request.start_lat,
            start_lon=request.start_lon,
            end_lat=None if same_point else request.end_lat,
            end_lon=None if same_point else request.end_lon,
        )
        _v46_mark("Avoid-area graph filtering", _v46_t)

        if same_point:
            end_node = start_node
        else:
            end_node = ox.distance.nearest_nodes(
                G,
                X=request.end_lon,
                Y=request.end_lat,
            )

        # Pass-through markers and V28 direct-edit handles are both snapped to
        # nearby natural trails and inserted as temporary routing nodes. Direct
        # edit handles stay visually separate in the browser, but routing treats
        # them as additional hard corridor requirements.
        combined_pass_points = list(request.pass_points or []) + list(request.route_edit_points or [])
        _v46_t = time.perf_counter()
        G, required_pass_points = resolve_required_pass_points(G, combined_pass_points)
        _v46_mark("Pass-point resolution", _v46_t)
        if start_node not in G:
            raise HTTPException(status_code=500, detail="Start node was lost while inserting required pass-through points.")

        _v46_search_started = time.perf_counter()
        if same_point:
            if required_pass_points:
                constrained_profile = required_waypoint_profile(
                    profile, request.target_distance_miles
                )
                (
                    route_nodes,
                    metrics,
                    search_steps,
                ) = generate_waypoint_loop(
                    G,
                    start_node,
                    target_distance_meters,
                    target_gain_meters,
                    constrained_profile,
                    limits,
                    required_pass_points=required_pass_points,
                    route_diversity=route_diversity,
                )
                profile = constrained_profile
                route_type = "required pass-through trail loop"
                search_method = "required-waypoint"
                states_expanded = None

            elif request.target_distance_miles < 4.0:
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
                    route_diversity=route_diversity,
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
                    route_diversity=route_diversity,
                )

                route_type = "adaptive waypoint trail loop"
                search_method = "waypoint"
                states_expanded = None

        else:
            S = make_simple_routing_graph(G)

            destinations = []
            remaining = list(required_pass_points)
            current_node = start_node
            # Automatic order: greedily visit the geographically nearest
            # required zone next, then finish at the requested end point.
            while remaining:
                current_lat = float(G.nodes[current_node]["y"])
                current_lon = float(G.nodes[current_node]["x"])
                remaining.sort(
                    key=lambda info: haversine_meters(
                        current_lat, current_lon,
                        info["requested_lat"], info["requested_lon"],
                    )
                )
                info = remaining.pop(0)
                node = info["candidate_nodes"][0]
                destinations.append(node)
                current_node = node
            destinations.append(end_node)

            route_nodes = [start_node]
            current = start_node
            try:
                for destination in destinations:
                    leg = fast_shortest_path(S, current, destination, weight="routing_cost")
                    route_nodes.extend(leg[1:])
                    current = destination
            except nx.NetworkXNoPath:
                raise HTTPException(
                    status_code=400,
                    detail="No connected trail route found through all required pass-through points.",
                )

            _, metrics = route_score(
                G,
                route_nodes,
                target_distance_meters,
                target_gain_meters,
            )
            metrics["required_pass_points"] = required_pass_metrics_for_coords(
                metrics.get("route_coordinates", []), required_pass_points
            )

            search_steps = 1
            states_expanded = None
            search_method = "required-point-to-point" if required_pass_points else "point-to-point"
            route_type = "trail point-to-point with required pass-through" if required_pass_points else "trail point-to-point"

        _v46_mark("Core route search", _v46_search_started)

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

        _v46_t = time.perf_counter()
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
        _v46_mark("Final option payload / elevation", _v46_t)

        # V50: route choices are presented from lowest mileage to highest.
        route_options.sort(
            key=lambda option: (
                float(option.get("actual_distance_miles", 0.0)),
                float(option.get("actual_gain_ft", 0.0)),
                float(option.get("route_score", 0.0)),
            )
        )
        for option_index, option in enumerate(route_options):
            option["index"] = option_index
            option["name"] = f"Route {option_index + 1}"

        # Build the no-elevation COROS GPX track only after the winning route
        # has been selected. This is geometry-only and performs no extra DEM
        # raster sampling.
        _v46_t = time.perf_counter()
        gpx_export_points = build_gpx_export_points(coords)
        _v46_mark("GPX geometry preparation", _v46_t)

        _v46_profile = _v46_report()

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
            "elevation_profile": metrics.get("elevation_profile", []),
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
            "required_pass_points": metrics.get("required_pass_points", []),
            "required_pass_points_count": len(required_pass_points),
            "route_edit_points_count": len(request.route_edit_points or []),
            "avoid_areas_count": len(resolved_avoid_areas),
            "avoid_segments_count": len(resolved_avoid_segments),
            "prefer_segments_count": len(resolved_prefer_segments),
            "route_diversity": route_diversity,
            "search_seed": request.search_seed,
            "version": APP_VERSION,
            "snapped_start_lat": snapped_start_lat,
            "snapped_start_lon": snapped_start_lon,
            "snap_distance_m": round(snap_distance_m, 1),
            "exact_start_inserted": bool(start_info["exact_inserted"]),
            "start_trail_offset_m": float(start_info["trail_offset_m"]),
            "start_source_edge": start_info.get("source_edge"),
            "status": "Route generated",
            "timing_profile": _v46_profile,
        }

    except HTTPException:
        _v46_report()
        raise
    except Exception as exc:
        _v46_report()
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
<link rel="stylesheet" href="https://unpkg.com/maplibre-gl@5.24.0/dist/maplibre-gl.css">

<style>
* {
    box-sizing: border-box;
}

html, body {
    margin: 0;
    width: 100%;
    height: 100%;
}

body {
    font-family: Arial, sans-serif;
    background: #f5f5f5;
    display: flex;
    height: 100vh;
    overflow: hidden;
}

#controls {
    flex: 0 0 20%;
    width: 20%;
    max-width: 20%;
    height: 100vh;
    overflow-y: auto;
    overflow-x: hidden;
    padding: 14px;
    background: white;
    border-right: 1px solid #ccc;
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
    flex: 1 1 135px;
    min-width: 0;
}

label {
    font-size: 13px;
    margin-bottom: 4px;
    font-weight: bold;
}

input[type="number"] {
    width: 100%;
    min-width: 0;
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


#results {
    width: 100%;
}

#visual-panel {
    flex: 1 1 80%;
    width: 80%;
    height: 100vh;
    min-width: 0;
    min-height: 0;
    display: flex;
    flex-direction: column;
    background: white;
}

#map-wrap {
    position: relative;
    flex: 0 0 80%;
    width: 100%;
    height: 80%;
    min-height: 0;
    overflow: hidden;
}

#map,
#map3d {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    min-height: 0;
}

#map3d {
    display: none;
}

#map-mode-control {
    position: absolute;
    top: 10px;
    right: 10px;
    z-index: 1000;
    display: flex;
    gap: 4px;
    padding: 4px;
    border-radius: 8px;
    background: rgba(255,255,255,0.94);
    box-shadow: 0 1px 5px rgba(0,0,0,0.28);
}

#map-mode-control button {
    margin: 0;
    padding: 7px 11px;
    min-width: 46px;
    font-size: 13px;
    border-radius: 6px;
    background: #444;
}

#map-mode-control button.active {
    background: #111;
}

#terrain-status {
    position: absolute;
    left: 10px;
    bottom: 10px;
    z-index: 1000;
    display: none;
    padding: 5px 8px;
    border-radius: 6px;
    background: rgba(255,255,255,0.92);
    color: #333;
    font-size: 11px;
}

#elevation-profile-panel {
    flex: 0 0 20%;
    width: 100%;
    height: 20%;
    min-height: 0;
    padding: 8px 12px 10px 12px;
    border-top: 1px solid #cfcfcf;
    background: #fff;
    display: flex;
    flex-direction: column;
}

#elevation-profile-title {
    font-size: 13px;
    font-weight: bold;
    margin-bottom: 3px;
}

#elevationProfileSvg {
    display: block;
    width: 100%;
    flex: 1 1 auto;
    min-height: 0;
}

.elevation-axis-label {
    fill: #555;
    font-size: 11px;
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

#pass-point-panel,
#avoid-area-panel,
#segment-control-panel,
#diversity-panel {
    border: 1px solid #ddd;
    border-radius: 6px;
    padding: 10px;
    margin: 10px 0;
    background: #fafafa;
}

.pass-point-row,
.avoid-area-row,
.segment-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: end;
    margin-top: 8px;
    padding-top: 8px;
    border-top: 1px solid #e5e5e5;
}

.pass-point-row input[type="number"],
.avoid-area-row input[type="number"] {
    width: 150px;
}

.pass-point-remove,
.avoid-area-remove,
.segment-remove {
    background: #666;
}

.segment-button-row {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 8px;
}

.segment-button-row button {
    flex: 1 1 130px;
    margin-right: 0;
}

#avoidTrailSegmentButton {
    background: #9a3412;
}

#preferTrailSegmentButton {
    background: #166534;
}

.segment-row {
    justify-content: space-between;
    font-size: 12px;
}

.segment-row .segment-kind {
    font-weight: bold;
}

.segment-row.avoid .segment-kind { color: #9a3412; }
.segment-row.prefer .segment-kind { color: #166534; }

#diversityValue {
    font-weight: bold;
}

input[type="range"] {
    width: 100%;
}

#passPointStatus,
#avoidAreaStatus {
    margin-top: 6px;
}

.route-choice-grid {
    display: flex;
    flex-direction: column;
    gap: 7px;
    margin: 10px 0 12px 0;
}

.route-choice {
    width: 100%;
    text-align: left;
    background: #f8fafc;
    color: #111;
    border: 1px solid #cbd5e1;
    margin: 0;
    padding: 9px 10px;
    font-size: 12px;
    line-height: 1.35;
}

.route-choice:hover {
    background: #eef2ff;
}

.route-choice.selected {
    background: #b91c1c;
    color: white;
    border-color: #991b1b;
}

.route-card-title {
    display: block;
    font-size: 13px;
    font-weight: 700;
    margin-bottom: 2px;
}

.route-card-primary,
.route-card-secondary {
    display: block;
}

.route-card-secondary {
    opacity: 0.82;
    margin-top: 2px;
}

#findMoreButton {
    background: #334155;
}

.trail-selection-line-avoid {
    stroke-dasharray: 7 5;
}

.elevation-hover-tooltip {
    background: rgba(17, 24, 39, 0.9);
    color: white;
    border: none;
    box-shadow: none;
    font-size: 11px;
}

#selectedRouteDetails {
    margin-top: 8px;
}


/* V28 compact accordion sidebar + modification controls */
.sidebar-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    margin-bottom: 10px;
}

.sidebar-header h2 {
    margin: 0;
    font-size: 20px;
}

.history-buttons {
    display: flex;
    gap: 5px;
    flex: 0 0 auto;
}

.icon-button {
    width: 34px;
    height: 32px;
    padding: 0;
    margin: 0;
    border-radius: 7px;
    background: #e2e8f0;
    color: #0f172a;
    font-size: 20px;
    line-height: 1;
}

.icon-button:disabled {
    opacity: 0.35;
    cursor: default;
}

.control-section {
    border: 1px solid #dbe2ea;
    border-radius: 9px;
    margin: 8px 0;
    background: #fff;
    overflow: hidden;
}

.control-section > summary {
    cursor: pointer;
    list-style: none;
    padding: 10px 12px;
    font-size: 13px;
    font-weight: 700;
    color: #0f172a;
    background: #f8fafc;
    user-select: none;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.control-section > summary::-webkit-details-marker { display: none; }
.control-section > summary::after {
    content: "▾";
    color: #64748b;
    font-size: 12px;
    transition: transform 0.15s ease;
}
.control-section:not([open]) > summary::after { transform: rotate(-90deg); }
.control-section[open] > summary { border-bottom: 1px solid #e5e7eb; }

.section-content {
    padding: 10px;
    background: #fff;
}

.tool-block,
#pass-point-panel,
#avoid-area-panel,
#segment-control-panel,
#diversity-panel {
    border: 0;
    border-radius: 8px;
    padding: 9px;
    margin: 0 0 9px 0;
    background: #f8fafc;
}

.tool-block:last-child { margin-bottom: 0; }
.tool-heading { font-size: 12px; font-weight: 700; margin-bottom: 4px; color: #0f172a; }
.compact-tool-block { padding-bottom: 7px; }
.tool-action { margin-top: 7px; }

.secondary-button {
    background: #475569;
}

.primary-actions {
    position: sticky;
    bottom: 0;
    z-index: 600;
    display: grid;
    grid-template-columns: 1fr;
    gap: 6px;
    padding: 9px 0 8px 0;
    margin-top: 8px;
    background: linear-gradient(to bottom, rgba(255,255,255,0.84), #fff 24%);
}

.primary-actions button {
    width: 100%;
    margin: 0;
}

#generateButton { background: #0f172a; }
#downloadGpxButton { background: #166534; }

.toggle-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
}
.simple-toggle-row { margin-bottom: 9px; font-size: 12px; font-weight: 600; }

.switch {
    position: relative;
    display: inline-block;
    width: 40px;
    height: 22px;
    flex: 0 0 40px;
    margin: 0;
}
.switch input { opacity: 0; width: 0; height: 0; }
.switch .slider {
    position: absolute;
    inset: 0;
    cursor: pointer;
    background: #cbd5e1;
    border-radius: 999px;
    transition: .18s;
}
.switch .slider::before {
    content: "";
    position: absolute;
    height: 16px;
    width: 16px;
    left: 3px;
    top: 3px;
    background: #fff;
    border-radius: 50%;
    box-shadow: 0 1px 2px rgba(0,0,0,.2);
    transition: .18s;
}
.switch input:checked + .slider { background: #0f766e; }
.switch input:checked + .slider::before { transform: translateX(18px); }

.quality-filter-fields {
    margin-top: 8px;
    transition: opacity .15s ease;
}
.disabled-block {
    opacity: .42;
    pointer-events: none;
}

.range-labels {
    display: flex;
    justify-content: space-between;
    font-size: 10px;
    color: #64748b;
    margin-top: -2px;
}

.route-edit-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    padding: 7px 0 0 0;
    margin-top: 7px;
    border-top: 1px solid #e5e7eb;
    font-size: 11px;
}
.route-edit-remove { background: #64748b; margin: 0; padding: 6px 8px; font-size: 11px; }

.route-edit-handle.leaflet-marker-icon {
    filter: hue-rotate(320deg) saturate(1.4);
}

.filter-badge {
    display: inline-block;
    margin-left: 4px;
    padding: 2px 6px;
    border-radius: 999px;
    background: #e2e8f0;
    color: #334155;
    font-size: 10px;
    font-weight: 700;
}

.route-card-title-line {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 6px;
}
.edited-badge {
    display: inline-block;
    padding: 1px 5px;
    border-radius: 999px;
    background: rgba(255,255,255,.18);
    font-size: 9px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .03em;
}
.route-choice:not(.selected) .edited-badge {
    background: #fee2e2;
    color: #991b1b;
}

.section-replacement-panel { border-color: #dbe5df; background: #fbfdfb; }
.replacement-legend {
    display: flex;
    flex-wrap: wrap;
    gap: 8px 12px;
    margin: 9px 0;
    font-size: 10px;
    color: #475569;
}
.replacement-legend span { display: inline-flex; align-items: center; gap: 5px; }
.legend-line { display: inline-block; width: 20px; height: 0; border-top: 4px solid; border-radius: 9px; }
.legend-line.keep { border-color: #d60000; }
.legend-line.remove { border-color: #f97316; }
.legend-line.replacement { border-color: #16a34a; }
.replacement-actions {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 6px;
    margin-top: 8px;
}
.replacement-actions #applyReplacementButton { background: #166534; }
.replacement-status { margin-top: 7px; line-height: 1.35; }
.replacement-summary {
    margin-top: 7px;
    padding: 7px 8px;
    border-radius: 7px;
    background: #f1f5f9;
    font-size: 10px;
    color: #334155;
    min-height: 0;
}
.replacement-summary:empty { display: none; }

#qualityFilterStatus { margin-top: 3px; }
#routeEditStatus { margin-top: 5px; }

@media (max-width: 900px) {
    body {
        display: block;
        height: auto;
        overflow: auto;
    }

    #controls {
        width: 100%;
        max-width: none;
        height: auto;
        overflow: visible;
        border-right: none;
        border-bottom: 1px solid #ccc;
    }

    #visual-panel {
        width: 100%;
        height: auto;
        display: block;
    }

    #map {
        width: 100%;
        height: 65vh;
        min-height: 450px;
    }

    #elevation-profile-panel {
        width: 100%;
        height: 22vh;
        min-height: 150px;
    }
}
</style>
</head>

<body>

<div id="controls">

<div class="sidebar-header">
    <h2>Trail Running Creator</h2>
    <div class="history-buttons" aria-label="Undo and redo">
        <button id="undoButton" type="button" class="icon-button" disabled title="Undo last change">↶</button>
        <button id="redoButton" type="button" class="icon-button" disabled title="Redo last change">↷</button>
    </div>
</div>

<details class="control-section" open>
    <summary>Route request</summary>
    <div class="section-content">
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

        <button id="chooseStartButton" type="button" class="secondary-button">Choose start on map</button>
        <div id="startPointStatus" class="small">Click anywhere on the map to choose the starting point.</div>

        <div class="input-row request-target-row">
            <div class="input-group">
                <label for="distance">Target distance (miles)</label>
                <input id="distance" type="number" step="0.1" min="0.1" value="2.5">
            </div>
            <div class="input-group">
                <label for="gain">Target elevation gain (ft)</label>
                <input id="gain" type="number" step="25" min="0" value="200">
            </div>
        </div>
    </div>
</details>

<details class="control-section">
    <summary>Route settings</summary>
    <div class="section-content">
        <div id="diversity-panel" class="tool-block compact-tool-block">
            <div class="tool-heading">Route diversity</div>
            <input id="routeDiversity" type="range" min="0" max="100" step="5" value="50">
            <div class="range-labels"><span>Similar</span><span>Very different</span></div>
            <div class="small">Current: <span id="diversityValue">50</span>/100</div>
        </div>

        <div id="quality-filter-panel" class="tool-block">
            <div class="toggle-row">
                <div>
                    <div class="tool-heading">Route quality filters</div>
                    <div class="small">Optional limits applied to the route choices you see.</div>
                </div>
                <label class="switch" title="Enable route quality filters">
                    <input id="enableQualityFilters" type="checkbox">
                    <span class="slider"></span>
                </label>
            </div>
            <div id="qualityFilterFields" class="quality-filter-fields disabled-block">
                <div class="input-row">
                    <div class="input-group">
                        <label for="maxRetrace">Max retrace (%)</label>
                        <input id="maxRetrace" type="number" min="0" max="100" step="1" value="15">
                    </div>
                    <div class="input-group">
                        <label for="maxConnector">Max connector (%)</label>
                        <input id="maxConnector" type="number" min="0" max="100" step="1" value="5">
                    </div>
                    <div class="input-group">
                        <label for="minTrail">Minimum trail (%)</label>
                        <input id="minTrail" type="number" min="0" max="100" step="1" value="95">
                    </div>
                    <div class="input-group">
                        <label for="distanceTolerance">Distance tolerance (±%)</label>
                        <input id="distanceTolerance" type="number" min="0" max="100" step="1" value="10">
                    </div>
                </div>
                <div id="qualityFilterStatus" class="small"></div>
            </div>
        </div>
    </div>
</details>

<details class="control-section">
    <summary>Modify route</summary>
    <div class="section-content">
        <div id="section-replacement-panel" class="tool-block section-replacement-panel">
            <div class="tool-heading">Replace a route section</div>
            <div class="small">Click Start selection, click the red route once where the old section begins, then click the gray trail pieces you want to use in order. When the green corridor reaches the red route again, click Replace section. The orange section is removed and the highlighted green trail geometry is spliced into the route exactly.</div>
            <div class="replacement-legend">
                <span><i class="legend-line keep"></i> kept route</span>
                <span><i class="legend-line remove"></i> remove</span>
                <span><i class="legend-line replacement"></i> replacement</span>
            </div>
            <div class="replacement-actions">
                <button id="replaceSectionButton" type="button" disabled>Start selection</button>
                <button id="applyReplacementButton" type="button" disabled>Replace section</button>
            </div>
            <div id="routeReplacementStatus" class="small replacement-status">Select a route first.</div>
            <div id="replacementSelectionSummary" class="replacement-summary"></div>
            <!-- Kept only as hidden compatibility hooks for older helper code. -->
            <button id="undoReplacementSegmentButton" type="button" style="display:none" disabled></button>
            <button id="cancelReplacementButton" type="button" style="display:none" disabled></button>
        </div>

        <!-- Legacy v28 edit-handle elements stay hidden so older browser code
             paths remain harmless during the transition to section replacement. -->
        <div style="display:none">
            <button id="editRouteButton" type="button" disabled>Edit selected route</button>
            <button id="clearRouteEditsButton" type="button" disabled>Clear edit points</button>
            <div id="routeEditStatus"></div>
            <div id="routeEditRows"></div>
        </div>

        <div id="pass-point-panel" class="tool-block">
            <div class="tool-heading">Required pass-through points</div>
            <div class="small">Force the route through a nearby trail corridor.</div>
            <button id="addPassPointButton" type="button" class="secondary-button tool-action">Add pass-through point</button>
            <div id="passPointStatus" class="small"></div>
            <div id="passPointRows"></div>
        </div>

        <div id="avoid-area-panel" class="tool-block">
            <div class="tool-heading">Avoid areas</div>
            <div class="small">Block routing through a circular area.</div>
            <button id="addAvoidAreaButton" type="button" class="secondary-button tool-action">Add avoid area</button>
            <div id="avoidAreaStatus" class="small"></div>
            <div id="avoidAreaRows"></div>
        </div>

        <div id="segment-control-panel" class="tool-block">
            <div class="tool-heading">Trail segment controls</div>
            <div class="small">Click a gray trail to avoid it completely or softly prefer it.</div>
            <div class="segment-button-row">
                <button id="avoidTrailSegmentButton" type="button">Avoid trail segment</button>
                <button id="preferTrailSegmentButton" type="button">Prefer trail segment</button>
            </div>
            <div id="segmentStatus" class="small"></div>
            <div id="segmentRows"></div>
        </div>

        <div id="persistent-blocklist-panel" class="tool-block">
            <div class="tool-heading">Permanent personal blocks</div>
            <div class="small">
                Avoided areas and avoided trail segments are saved automatically in this browser.
                Export a backup to move them to another browser or device.
            </div>
            <div class="segment-button-row">
                <button id="exportBlocklistButton" type="button" class="secondary-button">Export blocks</button>
                <button id="importBlocklistButton" type="button" class="secondary-button">Import blocks</button>
                <button id="clearBlocklistButton" type="button" class="secondary-button">Clear blocks</button>
            </div>
            <input id="importBlocklistInput" type="file" accept=".json,application/json" style="display:none">
            <div id="persistentBlocklistStatus" class="small"></div>
        </div>
    </div>
</details>

<!-- V50: network overlay/workspace preparation is automatic. -->
<div style="display:none">
    <input id="showNetwork" type="checkbox" checked>
    <button id="networkButton" type="button"></button>
</div>

<div class="primary-actions">
    <button id="generateButton">Generate Trail Route</button>
    <button id="findMoreButton" disabled>Find More Routes</button>
    <button id="downloadGpxButton" disabled>Download GPX for COROS</button>
</div>

<div id="results">Ready.</div>
</div>

<div id="visual-panel">
    <div id="map-wrap">
        <div id="map"></div>
        <div id="map3d"></div>
        <div id="map-mode-control" aria-label="Map view">
            <button id="map2dButton" type="button" class="active">2D</button>
            <button id="map3dButton" type="button" title="3D terrain preview">3D</button>
        </div>
        <div id="terrain-status">3D terrain preview · use 2D for route editing</div>
    </div>
    <div id="elevation-profile-panel">
        <div id="elevation-profile-title">Elevation profile · select a route</div>
        <svg id="elevationProfileSvg" role="img" aria-label="Elevation profile of selected route"></svg>
    </div>
</div>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://unpkg.com/maplibre-gl@5.24.0/dist/maplibre-gl.js"></script>

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

// ============================================================
// V40 OPTIONAL 3D MAP PREVIEW
// ============================================================
//
// Routing/editing remains on the proven Leaflet map. The 3D view is a fast
// MapLibre preview using a hosted raster DEM. This avoids changing any backend
// routing behavior and can be removed later if we switch to self-hosted terrain.
//
// Terrain source follows MapLibre's current 3D terrain example.
let map3d = null;
let map3dReady = false;
let map3dActive = false;

const map2dButton = document.getElementById("map2dButton");
const map3dButton = document.getElementById("map3dButton");
const map2dContainer = document.getElementById("map");
const map3dContainer = document.getElementById("map3d");
const terrainStatus = document.getElementById("terrain-status");

function emptyFeatureCollection() {
    return {type: "FeatureCollection", features: []};
}

function lineFeatureCollectionFromLatLonSegments(segments) {
    const features = [];
    for (const segment of (segments || [])) {
        if (!segment || segment.length < 2) continue;
        const coords = segment
            .map(p => [Number(p[1]), Number(p[0])])
            .filter(p => Number.isFinite(p[0]) && Number.isFinite(p[1]));
        if (coords.length >= 2) {
            features.push({
                type: "Feature",
                properties: {},
                geometry: {type: "LineString", coordinates: coords}
            });
        }
    }
    return {type: "FeatureCollection", features};
}

function selectedRoute3DGeoJSON() {
    if (!lastGeneratedRoute || !lastGeneratedRoute.route_options) {
        return emptyFeatureCollection();
    }
    const option = lastGeneratedRoute.route_options[selectedRouteOptionIndex];
    if (!option || !Array.isArray(option.route) || option.route.length < 2) {
        return emptyFeatureCollection();
    }
    const coords = option.route
        .map(p => [Number(p.lon), Number(p.lat)])
        .filter(p => Number.isFinite(p[0]) && Number.isFinite(p[1]));
    if (coords.length < 2) return emptyFeatureCollection();
    return {
        type: "FeatureCollection",
        features: [{
            type: "Feature",
            properties: {},
            geometry: {type: "LineString", coordinates: coords}
        }]
    };
}

function plannerPoints3DGeoJSON() {
    const features = [];

    const slat = Number(document.getElementById("start_lat").value);
    const slon = Number(document.getElementById("start_lon").value);
    if (Number.isFinite(slat) && Number.isFinite(slon)) {
        features.push({
            type: "Feature",
            properties: {kind: "start"},
            geometry: {type: "Point", coordinates: [slon, slat]}
        });
    }

    for (const point of (passPoints || [])) {
        if (Number.isFinite(Number(point.lat)) && Number.isFinite(Number(point.lon))) {
            features.push({
                type: "Feature",
                properties: {kind: "pass"},
                geometry: {type: "Point", coordinates: [Number(point.lon), Number(point.lat)]}
            });
        }
    }

    for (const area of (avoidAreas || [])) {
        if (Number.isFinite(Number(area.lat)) && Number.isFinite(Number(area.lon))) {
            features.push({
                type: "Feature",
                properties: {kind: "avoid"},
                geometry: {type: "Point", coordinates: [Number(area.lon), Number(area.lat)]}
            });
        }
    }

    return {type: "FeatureCollection", features};
}

function set3DSourceData(id, data) {
    if (!map3d || !map3dReady) return;
    const source = map3d.getSource(id);
    if (source && typeof source.setData === "function") {
        source.setData(data);
    }
}

function refresh3DMapData() {
    if (!map3d || !map3dReady) return;
    set3DSourceData("trail-overlay", lineFeatureCollectionFromLatLonSegments(masterTrailSegments));
    set3DSourceData("selected-route", selectedRoute3DGeoJSON());
    set3DSourceData("planner-points", plannerPoints3DGeoJSON());
}

function initialize3DMap() {
    if (map3d) return;

    const center = map.getCenter();
    map3d = new maplibregl.Map({
        container: "map3d",
        center: [center.lng, center.lat],
        zoom: map.getZoom(),
        pitch: 67,
        bearing: -18,
        maxPitch: 85,
        maxZoom: 18,
        style: {
            version: 8,
            sources: {
                osm: {
                    type: "raster",
                    tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
                    tileSize: 256,
                    maxzoom: 19,
                    attribution: "&copy; OpenStreetMap contributors"
                },
                terrainSource: {
                    type: "raster-dem",
                    url: "https://tiles.mapterhorn.com/tilejson.json"
                },
                hillshadeSource: {
                    type: "raster-dem",
                    url: "https://tiles.mapterhorn.com/tilejson.json"
                }
            },
            layers: [
                {
                    id: "osm-base",
                    type: "raster",
                    source: "osm"
                },
                {
                    id: "terrain-hillshade",
                    type: "hillshade",
                    source: "hillshadeSource",
                    paint: {
                        "hillshade-exaggeration": 0.35
                    }
                }
            ],
            terrain: {
                source: "terrainSource",
                exaggeration: 1.15
            }
        }
    });

    map3d.addControl(
        new maplibregl.NavigationControl({
            visualizePitch: true,
            showZoom: true,
            showCompass: true
        }),
        "top-left"
    );

    map3d.on("load", () => {
        map3d.addSource("trail-overlay", {
            type: "geojson",
            data: emptyFeatureCollection()
        });
        map3d.addLayer({
            id: "trail-overlay-line",
            type: "line",
            source: "trail-overlay",
            paint: {
                "line-color": "#666666",
                "line-width": 2.4,
                "line-opacity": 0.62
            }
        });

        map3d.addSource("selected-route", {
            type: "geojson",
            data: emptyFeatureCollection()
        });
        map3d.addLayer({
            id: "selected-route-line",
            type: "line",
            source: "selected-route",
            paint: {
                "line-color": "#d60000",
                "line-width": 5.5,
                "line-opacity": 0.98
            }
        });

        map3d.addSource("planner-points", {
            type: "geojson",
            data: emptyFeatureCollection()
        });
        map3d.addLayer({
            id: "planner-point-layer",
            type: "circle",
            source: "planner-points",
            paint: {
                "circle-radius": [
                    "match", ["get", "kind"],
                    "start", 7,
                    "pass", 6,
                    "avoid", 6,
                    5
                ],
                "circle-color": [
                    "match", ["get", "kind"],
                    "start", "#1565c0",
                    "pass", "#7b1fa2",
                    "avoid", "#ef6c00",
                    "#222222"
                ],
                "circle-stroke-color": "#ffffff",
                "circle-stroke-width": 1.5
            }
        });

        map3dReady = true;
        refresh3DMapData();
    });
}

function show3DMap() {
    initialize3DMap();

    const center = map.getCenter();
    map3dContainer.style.display = "block";
    map2dContainer.style.visibility = "hidden";
    map3dActive = true;
    map3dButton.classList.add("active");
    map2dButton.classList.remove("active");
    terrainStatus.style.display = "block";

    // MapLibre must resize after its previously hidden container becomes visible.
    requestAnimationFrame(() => {
        map3d.resize();
        map3d.jumpTo({
            center: [center.lng, center.lat],
            zoom: Math.max(0, map.getZoom() - 0.2),
            pitch: 67
        });
        refresh3DMapData();
    });
}

function show2DMap() {
    if (map3d && map3dActive) {
        const center = map3d.getCenter();
        const zoom = map3d.getZoom();
        map.setView([center.lat, center.lng], zoom, {animate: false});
    }

    map3dContainer.style.display = "none";
    map2dContainer.style.visibility = "visible";
    map3dActive = false;
    map2dButton.classList.add("active");
    map3dButton.classList.remove("active");
    terrainStatus.style.display = "none";
    requestAnimationFrame(() => map.invalidateSize());
}

map2dButton.addEventListener("click", show2DMap);
map3dButton.addEventListener("click", show3DMap);

let routeLine = null;
let routeOptionLines = [];
let selectedRouteOptionIndex = 0;
let networkLayer = L.layerGroup();
let lastGeneratedRoute = null;
let requestedStartMarker = null;
let snappedStartMarker = null;
let snapLine = null;
let loadedWorkspaceStartKey = null;
let lastWorkspaceResult = null;
let masterTrailOverlayPromise = null;
let masterTrailOverlayRequestId = 0;
let masterTrailOverlayLastKey = "";
let masterTrailOverlayTimer = null;
let masterTrailSegments = [];
let masterTrailRecords = [];
const MIN_TRAIL_OVERLAY_ZOOM = 11;
let startPointPlacementMode = true;
let passPoints = [];
let passPointLayers = [];
let passPointPlacementMode = false;
let nextPassPointId = 1;
let avoidAreas = [];
let avoidAreaLayers = [];
let avoidAreaPlacementMode = false;
let nextAvoidAreaId = 1;
let avoidSegments = [];
let preferSegments = [];
let trailSegmentLayers = [];
let trailSegmentPlacementMode = null; // "avoid" | "prefer" | null
let nextTrailSegmentId = 1;
let routeEditMode = false;
let routeEditPoints = [];
let routeEditLayers = [];
let nextRouteEditId = 1;

// V29 explicit section replacement state. This is intentionally separate from
// planner settings/history because it is a temporary edit selection until Apply.
let routeReplacementMode = false;
let routeReplacementStage = "idle"; // idle | cut-start | guide | ready
let replacementBaseRouteIndex = null;
let replacementCutStartIndex = null;
let replacementCutEndIndex = null;
let replacementTrailSegments = [];
let replacementLayers = [];

let elevationHoverMarker = null;
let elevationRenderState = null;
let currentSearchSeed = Math.floor(Date.now() % 2147483647);
let findMoreBatch = 0;
let lastSearchConfigKey = null;

// V28 undo/redo tracks planner settings and map modification controls. Generated
// route batches are intentionally not copied into history, keeping browser memory
// bounded even after many Find More searches.
let undoStack = [];
let redoStack = [];
let currentPlannerState = null;
let restoringPlannerState = false;

const PERSISTENT_BLOCKLIST_STORAGE_KEY = "trail-running-creator.personal-blocklist.v1";
const PERSISTENT_BLOCKLIST_SCHEMA = "trail-running-creator-personal-blocklist-v1";

const generateButton = document.getElementById("generateButton");
const findMoreButton = document.getElementById("findMoreButton");
const downloadGpxButton = document.getElementById("downloadGpxButton");
const networkButton = document.getElementById("networkButton");
const showNetworkCheckbox = document.getElementById("showNetwork");
const chooseStartButton = document.getElementById("chooseStartButton");
const addPassPointButton = document.getElementById("addPassPointButton");
const addAvoidAreaButton = document.getElementById("addAvoidAreaButton");
const avoidTrailSegmentButton = document.getElementById("avoidTrailSegmentButton");
const preferTrailSegmentButton = document.getElementById("preferTrailSegmentButton");
const exportBlocklistButton = document.getElementById("exportBlocklistButton");
const importBlocklistButton = document.getElementById("importBlocklistButton");
const clearBlocklistButton = document.getElementById("clearBlocklistButton");
const importBlocklistInput = document.getElementById("importBlocklistInput");
const persistentBlocklistStatus = document.getElementById("persistentBlocklistStatus");
const routeDiversityInput = document.getElementById("routeDiversity");
const diversityValue = document.getElementById("diversityValue");
const editRouteButton = document.getElementById("editRouteButton");
const clearRouteEditsButton = document.getElementById("clearRouteEditsButton");
const replaceSectionButton = document.getElementById("replaceSectionButton");
const applyReplacementButton = document.getElementById("applyReplacementButton");
const undoReplacementSegmentButton = document.getElementById("undoReplacementSegmentButton");
const cancelReplacementButton = document.getElementById("cancelReplacementButton");
const undoButton = document.getElementById("undoButton");
const redoButton = document.getElementById("redoButton");
const enableQualityFilters = document.getElementById("enableQualityFilters");
const maxRetraceInput = document.getElementById("maxRetrace");
const maxConnectorInput = document.getElementById("maxConnector");
const minTrailInput = document.getElementById("minTrail");
const distanceToleranceInput = document.getElementById("distanceTolerance");


generateButton.addEventListener("click", generateRoute);
findMoreButton.addEventListener("click", findMoreRoutes);
downloadGpxButton.addEventListener("click", downloadGeneratedGpx);
networkButton.addEventListener("click", reloadNetwork);
showNetworkCheckbox.addEventListener("change", updateNetworkVisibility);
chooseStartButton.addEventListener("click", beginStartPointPlacement);
addPassPointButton.addEventListener("click", beginPassPointPlacement);
addAvoidAreaButton.addEventListener("click", beginAvoidAreaPlacement);
avoidTrailSegmentButton.addEventListener("click", () => beginTrailSegmentPlacement("avoid"));
preferTrailSegmentButton.addEventListener("click", () => beginTrailSegmentPlacement("prefer"));
exportBlocklistButton.addEventListener("click", exportPersistentBlocklist);
importBlocklistButton.addEventListener("click", () => importBlocklistInput.click());
importBlocklistInput.addEventListener("change", importPersistentBlocklist);
clearBlocklistButton.addEventListener("click", clearPersistentBlocklist);
editRouteButton.addEventListener("click", beginRouteEditMode);
clearRouteEditsButton.addEventListener("click", clearRouteEditPoints);
replaceSectionButton.addEventListener("click", beginRouteSectionReplacement);
applyReplacementButton.addEventListener("click", applyRouteSectionReplacement);
undoReplacementSegmentButton.addEventListener("click", undoReplacementTrailSegment);
cancelReplacementButton.addEventListener("click", () => cancelRouteSectionReplacement(true));
undoButton.addEventListener("click", undoPlannerChange);
redoButton.addEventListener("click", redoPlannerChange);
routeDiversityInput.addEventListener("input", () => {
    diversityValue.textContent = routeDiversityInput.value;
});
routeDiversityInput.addEventListener("change", commitPlannerState);
enableQualityFilters.addEventListener("change", () => {
    updateQualityFilterUi();
    commitPlannerState();
    refreshQualityFilterView();
});
[maxRetraceInput, maxConnectorInput, minTrailInput, distanceToleranceInput].forEach(input => {
    input.addEventListener("change", () => {
        commitPlannerState();
        refreshQualityFilterView();
    });
});
["start_lat", "start_lon", "end_lat", "end_lon", "distance", "gain"].forEach(id => {
    document.getElementById(id).addEventListener("change", () => {
        if (id === "start_lat" || id === "start_lon") {
            const lat = Number(document.getElementById("start_lat").value);
            const lon = Number(document.getElementById("start_lon").value);
            if (Number.isFinite(lat) && Number.isFinite(lon)) {
                loadedWorkspaceStartKey = null;
                lastWorkspaceResult = null;
                createRequestedStartMarker(lat, lon, "Selected start · drag to adjust", false);
            }
        }
        commitPlannerState();
    });
});
map.on("click", handleMapPlacementClick);
document.addEventListener("keydown", event => {
    const tag = String(document.activeElement?.tagName || "").toLowerCase();
    if (tag === "input" || tag === "textarea" || tag === "select") return;
    const modifier = event.ctrlKey || event.metaKey;
    if (!modifier) return;
    if (event.key.toLowerCase() === "z" && !event.shiftKey) {
        event.preventDefault();
        undoPlannerChange();
    } else if ((event.key.toLowerCase() === "z" && event.shiftKey) || event.key.toLowerCase() === "y") {
        event.preventDefault();
        redoPlannerChange();
    }
});
window.addEventListener("resize", () => {
    map.invalidateSize();
    const selected = getSelectedRouteOption();
    if (selected) renderElevationProfile(selected);
});



function deepClonePlannerValue(value) {
    return JSON.parse(JSON.stringify(value));
}


function persistentBlocklistPayload() {
    return {
        schema: PERSISTENT_BLOCKLIST_SCHEMA,
        version: 1,
        saved_at: new Date().toISOString(),
        avoid_areas: avoidAreas.map(area => ({
            lat: Number(area.lat),
            lon: Number(area.lon),
            radius_miles: Number(area.radius_miles)
        })),
        avoid_segments: avoidSegments.map(item => ({
            lat: Number(item.lat),
            lon: Number(item.lon),
            geometry: deepClonePlannerValue(item.geometry || []),
            tile_id: item.tile_id ?? null,
            edge_u: item.edge_u ?? null,
            edge_v: item.edge_v ?? null,
            edge_key: item.edge_key ?? null
        }))
    };
}

function setPersistentBlocklistStatus(message, isError = false) {
    persistentBlocklistStatus.textContent = message;
    persistentBlocklistStatus.className = isError ? "small error" : "small";
}

function persistentBlockCountText() {
    const areaCount = avoidAreas.length;
    const segmentCount = avoidSegments.length;
    return `${areaCount} area${areaCount === 1 ? "" : "s"} and ${segmentCount} trail segment${segmentCount === 1 ? "" : "s"}`;
}

function savePersistentBlocklist(announce = false) {
    try {
        localStorage.setItem(PERSISTENT_BLOCKLIST_STORAGE_KEY, JSON.stringify(persistentBlocklistPayload()));
        setPersistentBlocklistStatus(
            avoidAreas.length || avoidSegments.length
                ? `Saved ${persistentBlockCountText()} permanently in this browser.`
                : "No permanent personal blocks saved yet."
        );
        return true;
    } catch (error) {
        setPersistentBlocklistStatus("Could not save permanent blocks in this browser: " + error.message, true);
        return false;
    }
}

function normalizePersistentBlocklist(value) {
    if (!value || typeof value !== "object" || Array.isArray(value)) {
        throw new Error("The blocklist file must contain a JSON object.");
    }
    if (value.schema !== PERSISTENT_BLOCKLIST_SCHEMA || Number(value.version) !== 1) {
        throw new Error("This is not a supported Trail Running Creator blocklist file.");
    }
    if (!Array.isArray(value.avoid_areas) || !Array.isArray(value.avoid_segments)) {
        throw new Error("The blocklist is missing its avoided areas or trail segments.");
    }
    if (value.avoid_areas.length > 500 || value.avoid_segments.length > 500) {
        throw new Error("The blocklist is too large to import safely.");
    }

    const areas = value.avoid_areas.map((area, index) => {
        const lat = Number(area?.lat);
        const lon = Number(area?.lon);
        const radius = Number(area?.radius_miles);
        if (!Number.isFinite(lat) || lat < -90 || lat > 90 ||
            !Number.isFinite(lon) || lon < -180 || lon > 180 ||
            !Number.isFinite(radius) || radius < 0.01 || radius > 10) {
            throw new Error(`Avoid area ${index + 1} has invalid coordinates or radius.`);
        }
        return {
            id: index + 1,
            lat: Number(lat.toFixed(7)),
            lon: Number(lon.toFixed(7)),
            radius_miles: Number(radius.toFixed(3))
        };
    });

    const segments = value.avoid_segments.map((segment, index) => {
        const lat = Number(segment?.lat);
        const lon = Number(segment?.lon);
        const rawGeometry = segment?.geometry;
        if (!Number.isFinite(lat) || lat < -90 || lat > 90 ||
            !Number.isFinite(lon) || lon < -180 || lon > 180 ||
            !Array.isArray(rawGeometry) || rawGeometry.length < 2 || rawGeometry.length > 10000) {
            throw new Error(`Avoided trail segment ${index + 1} is invalid.`);
        }
        const geometry = rawGeometry.map((point, pointIndex) => {
            const pointLat = Number(point?.[0]);
            const pointLon = Number(point?.[1]);
            if (!Number.isFinite(pointLat) || pointLat < -90 || pointLat > 90 ||
                !Number.isFinite(pointLon) || pointLon < -180 || pointLon > 180) {
                throw new Error(`Avoided trail segment ${index + 1}, point ${pointIndex + 1} is invalid.`);
            }
            return [Number(pointLat.toFixed(7)), Number(pointLon.toFixed(7))];
        });
        return {
            id: areas.length + index + 1,
            lat: Number(lat.toFixed(7)),
            lon: Number(lon.toFixed(7)),
            geometry,
            tile_id: typeof segment.tile_id === "string" ? segment.tile_id : null,
            edge_u: segment.edge_u === null || segment.edge_u === undefined
                ? null
                : (Number.isFinite(Number(segment.edge_u)) ? Number(segment.edge_u) : null),
            edge_v: segment.edge_v === null || segment.edge_v === undefined
                ? null
                : (Number.isFinite(Number(segment.edge_v)) ? Number(segment.edge_v) : null),
            edge_key: segment.edge_key === null || segment.edge_key === undefined ? null : String(segment.edge_key)
        };
    });
    return {areas, segments};
}

function applyPersistentBlocklist(value, announce = true) {
    const normalized = normalizePersistentBlocklist(value);
    avoidAreas = normalized.areas;
    avoidSegments = normalized.segments;
    nextAvoidAreaId = avoidAreas.length + 1;
    nextTrailSegmentId = Math.max(0, ...avoidSegments.concat(preferSegments).map(item => Number(item.id) || 0)) + 1;
    renderAvoidAreaRows();
    drawAvoidAreaLayers();
    renderTrailSegmentRows();
    drawTrailSegmentLayers();
    savePersistentBlocklist(false);
    if (announce) {
        setPersistentBlocklistStatus(`Loaded ${persistentBlockCountText()}. These blocks now apply to future routes.`);
    }
}

function loadPersistentBlocklist() {
    let raw;
    try {
        raw = localStorage.getItem(PERSISTENT_BLOCKLIST_STORAGE_KEY);
    } catch (error) {
        setPersistentBlocklistStatus("Permanent browser storage is unavailable: " + error.message, true);
        return;
    }
    if (!raw) {
        setPersistentBlocklistStatus("No permanent personal blocks saved yet.");
        return;
    }
    try {
        applyPersistentBlocklist(JSON.parse(raw), true);
    } catch (error) {
        setPersistentBlocklistStatus("Saved permanent blocks could not be loaded: " + error.message, true);
    }
}

function exportPersistentBlocklist() {
    const payload = persistentBlocklistPayload();
    payload.exported_at = new Date().toISOString();
    triggerTextDownload(
        "trail-running-permanent-blocks.json",
        JSON.stringify(payload, null, 2) + "\n",
        "application/json;charset=utf-8"
    );
    setPersistentBlocklistStatus(`Exported ${persistentBlockCountText()}.`);
}

async function importPersistentBlocklist() {
    const file = importBlocklistInput.files?.[0];
    if (!file) return;
    try {
        const value = JSON.parse(await file.text());
        applyPersistentBlocklist(value, false);
        commitPlannerState();
        setPersistentBlocklistStatus(`Imported and saved ${persistentBlockCountText()} permanently in this browser.`);
    } catch (error) {
        setPersistentBlocklistStatus("Could not import blocklist: " + error.message, true);
    } finally {
        importBlocklistInput.value = "";
    }
}

function clearPersistentBlocklist() {
    if (!avoidAreas.length && !avoidSegments.length) {
        setPersistentBlocklistStatus("There are no permanent personal blocks to clear.");
        return;
    }
    if (!window.confirm("Clear every permanently avoided area and trail segment from this browser?")) return;
    avoidAreas = [];
    avoidSegments = [];
    nextAvoidAreaId = 1;
    nextTrailSegmentId = Math.max(0, ...preferSegments.map(item => Number(item.id) || 0)) + 1;
    renderAvoidAreaRows();
    drawAvoidAreaLayers();
    renderTrailSegmentRows();
    drawTrailSegmentLayers();
    commitPlannerState();
    setPersistentBlocklistStatus("Permanent personal blocks cleared.");
}

function capturePlannerState() {
    return {
        start_lat: document.getElementById("start_lat").value,
        start_lon: document.getElementById("start_lon").value,
        end_lat: document.getElementById("end_lat").value,
        end_lon: document.getElementById("end_lon").value,
        distance: document.getElementById("distance").value,
        gain: document.getElementById("gain").value,
        route_diversity: routeDiversityInput.value,
        pass_points: deepClonePlannerValue(passPoints),
        avoid_areas: deepClonePlannerValue(avoidAreas),
        avoid_segments: deepClonePlannerValue(avoidSegments),
        prefer_segments: deepClonePlannerValue(preferSegments),
        route_edit_points: deepClonePlannerValue(routeEditPoints),
        quality_enabled: enableQualityFilters.checked,
        max_retrace: maxRetraceInput.value,
        max_connector: maxConnectorInput.value,
        min_trail: minTrailInput.value,
        distance_tolerance: distanceToleranceInput.value,
        show_network: showNetworkCheckbox.checked
    };
}


function plannerStateKey(state) {
    return JSON.stringify(state);
}


function updateUndoRedoButtons() {
    undoButton.disabled = undoStack.length === 0;
    redoButton.disabled = redoStack.length === 0;
}


function resetPlannerHistory() {
    undoStack = [];
    redoStack = [];
    currentPlannerState = capturePlannerState();
    updateUndoRedoButtons();
}


function commitPlannerState() {
    if (restoringPlannerState) return;
    savePersistentBlocklist(false);
    const nextState = capturePlannerState();
    if (currentPlannerState === null) {
        currentPlannerState = nextState;
        updateUndoRedoButtons();
        return;
    }
    if (plannerStateKey(nextState) === plannerStateKey(currentPlannerState)) return;
    undoStack.push(deepClonePlannerValue(currentPlannerState));
    if (undoStack.length > 60) undoStack.shift();
    currentPlannerState = nextState;
    redoStack = [];
    updateUndoRedoButtons();
}


function applyPlannerState(state) {
    if (!state) return;
    restoringPlannerState = true;
    try {
        const oldStartLat = Number(document.getElementById("start_lat").value);
        const oldStartLon = Number(document.getElementById("start_lon").value);

        document.getElementById("start_lat").value = state.start_lat;
        document.getElementById("start_lon").value = state.start_lon;
        document.getElementById("end_lat").value = state.end_lat;
        document.getElementById("end_lon").value = state.end_lon;
        document.getElementById("distance").value = state.distance;
        document.getElementById("gain").value = state.gain;
        routeDiversityInput.value = state.route_diversity ?? 50;
        diversityValue.textContent = routeDiversityInput.value;

        passPoints = deepClonePlannerValue(state.pass_points || []);
        avoidAreas = deepClonePlannerValue(state.avoid_areas || []);
        avoidSegments = deepClonePlannerValue(state.avoid_segments || []);
        preferSegments = deepClonePlannerValue(state.prefer_segments || []);
        routeEditPoints = deepClonePlannerValue(state.route_edit_points || []);

        nextPassPointId = Math.max(0, ...passPoints.map(p => Number(p.id) || 0)) + 1;
        nextAvoidAreaId = Math.max(0, ...avoidAreas.map(p => Number(p.id) || 0)) + 1;
        nextTrailSegmentId = Math.max(0, ...avoidSegments.concat(preferSegments).map(p => Number(p.id) || 0)) + 1;
        nextRouteEditId = Math.max(0, ...routeEditPoints.map(p => Number(p.id) || 0)) + 1;

        enableQualityFilters.checked = Boolean(state.quality_enabled);
        maxRetraceInput.value = state.max_retrace ?? 15;
        maxConnectorInput.value = state.max_connector ?? 5;
        minTrailInput.value = state.min_trail ?? 95;
        distanceToleranceInput.value = state.distance_tolerance ?? 10;
        showNetworkCheckbox.checked = state.show_network !== false;

        clearPlacementModes();
        renderPassPointRows();
        drawPassPointLayers();
        renderAvoidAreaRows();
        drawAvoidAreaLayers();
        renderTrailSegmentRows();
        drawTrailSegmentLayers();
        renderRouteEditRows();
        drawRouteEditLayers();
        updateQualityFilterUi();
        updateNetworkVisibility();

        const newStartLat = Number(state.start_lat);
        const newStartLon = Number(state.start_lon);
        if (Number.isFinite(newStartLat) && Number.isFinite(newStartLon)) {
            createRequestedStartMarker(newStartLat, newStartLon, "Selected start · drag to adjust", false);
        }
        if (Math.abs(newStartLat - oldStartLat) > 1e-8 || Math.abs(newStartLon - oldStartLon) > 1e-8) {
            loadedWorkspaceStartKey = null;
            lastWorkspaceResult = null;
        }

        refreshQualityFilterView(true);
        document.getElementById("routeEditStatus").textContent =
            "Planner change restored. Generate again if you want the route search recalculated.";
    } finally {
        restoringPlannerState = false;
    }
    savePersistentBlocklist(false);
}


function undoPlannerChange() {
    if (!undoStack.length) return;
    const previous = undoStack.pop();
    redoStack.push(deepClonePlannerValue(currentPlannerState));
    currentPlannerState = deepClonePlannerValue(previous);
    applyPlannerState(previous);
    updateUndoRedoButtons();
}


function redoPlannerChange() {
    if (!redoStack.length) return;
    const next = redoStack.pop();
    undoStack.push(deepClonePlannerValue(currentPlannerState));
    currentPlannerState = deepClonePlannerValue(next);
    applyPlannerState(next);
    updateUndoRedoButtons();
}


function updateQualityFilterUi() {
    const fields = document.getElementById("qualityFilterFields");
    fields.classList.toggle("disabled-block", !enableQualityFilters.checked);
}


function getQualityFilterConfig() {
    return {
        enabled: enableQualityFilters.checked,
        maxRetrace: Math.max(0, Number(maxRetraceInput.value) || 0),
        maxConnector: Math.max(0, Number(maxConnectorInput.value) || 0),
        minTrail: Math.max(0, Number(minTrailInput.value) || 0),
        distanceTolerance: Math.max(0, Number(distanceToleranceInput.value) || 0)
    };
}


function routePassesQualityFilters(option) {
    // A route the user explicitly edited must remain visible even when optional
    // quality filters would normally hide generated alternatives.
    if (option && option.is_edited) return true;
    const filter = getQualityFilterConfig();
    if (!filter.enabled) return true;
    const targetDistance = Number(lastGeneratedRoute?.requested_distance_miles ?? document.getElementById("distance").value);
    const actualDistance = Number(option.actual_distance_miles || 0);
    const distanceToleranceMiles = targetDistance * filter.distanceTolerance / 100.0;
    return Number(option.retrace_percent ?? 0) <= filter.maxRetrace + 1e-9 &&
        Number(option.connector_percent ?? 0) <= filter.maxConnector + 1e-9 &&
        Number(option.trail_percent ?? 0) >= filter.minTrail - 1e-9 &&
        Math.abs(actualDistance - targetDistance) <= distanceToleranceMiles + 1e-9;
}


function getVisibleRouteIndices() {
    if (!lastGeneratedRoute) return [];
    return (lastGeneratedRoute.route_options || [])
        .map((option, index) => routePassesQualityFilters(option) ? index : -1)
        .filter(index => index >= 0);
}


function refreshQualityFilterView(redrawMap = true) {
    updateQualityFilterUi();
    if (!lastGeneratedRoute) return;
    const visible = getVisibleRouteIndices();
    const status = document.getElementById("qualityFilterStatus");
    if (enableQualityFilters.checked) {
        status.textContent = `${visible.length} of ${(lastGeneratedRoute.route_options || []).length} route choices meet these limits.`;
    } else {
        status.textContent = "";
    }
    let nextIndex = selectedRouteOptionIndex;
    if (!visible.includes(nextIndex)) nextIndex = visible.length ? visible[0] : -1;
    renderRouteResults("Route choices updated");
    if (redrawMap) {
        drawRouteOptions(lastGeneratedRoute, nextIndex >= 0 ? nextIndex : 0, false);
    }
}


function nearestRouteCoordinateIndex(option, latlng) {
    const route = (option && option.route) || [];
    if (route.length < 3) return null;
    const click = map.latLngToLayerPoint(latlng);
    let bestIndex = null;
    let bestDistance = Infinity;
    // Do not use the duplicated first/last loop point as a cut anchor. Keeping
    // edits away from the route seam makes the kept red geometry unambiguous.
    const first = route.length > 4 ? 1 : 0;
    const last = route.length > 4 ? route.length - 2 : route.length - 1;
    for (let i = first; i <= last; i++) {
        const point = route[i];
        const layerPoint = map.latLngToLayerPoint(
            L.latLng(Number(point.lat), Number(point.lon))
        );
        const distance = Math.hypot(click.x - layerPoint.x, click.y - layerPoint.y);
        if (distance < bestDistance) {
            bestDistance = distance;
            bestIndex = i;
        }
    }
    return bestIndex;
}


function clearRouteReplacementLayers() {
    for (const layer of replacementLayers) {
        if (layer && map.hasLayer(layer)) map.removeLayer(layer);
    }
    replacementLayers = [];
}


function updateRouteReplacementControls() {
    const status = document.getElementById("routeReplacementStatus");
    const summary = document.getElementById("replacementSelectionSummary");
    const selected = getSelectedRouteOption();

    replaceSectionButton.disabled = !selected;
    replaceSectionButton.textContent = "Start selection";
    cancelReplacementButton.disabled = true;
    undoReplacementSegmentButton.disabled = true;
    // V33: there are only two user actions. Once a start and at least one
    // replacement trail piece exist, Replace section is available. It finds
    // the rejoin point automatically from the final green trail selection.
    applyReplacementButton.disabled = !(
        routeReplacementStage === "guide" &&
        replacementCutStartIndex !== null &&
        replacementTrailSegments.length > 0
    );

    if (routeReplacementStage === "idle") {
        if (status) status.textContent = selected
            ? "Click Start selection to replace part of the selected red route."
            : "Select a route first.";
        if (summary) summary.innerHTML = "";
        return;
    }

    if (routeReplacementStage === "cut-start") {
        status.textContent = "Click the red route once where the section you want to replace begins.";
        summary.innerHTML = "";
        return;
    }

    if (routeReplacementStage === "guide") {
        status.textContent = replacementTrailSegments.length
            ? "Keep clicking gray trail pieces in order. When the last green piece reaches the red route again, click Replace section."
            : "Now click the gray trail pieces you want the replacement to follow, in order.";
        summary.innerHTML =
            `<b>Start:</b> selected · <b>green trail pieces:</b> ${replacementTrailSegments.length}`;
        return;
    }

    status.textContent = "Replacing section...";
}


function drawRouteReplacementLayers() {
    clearRouteReplacementLayers();
    if (routeReplacementStage === "idle" || replacementBaseRouteIndex === null) return;
    if (!lastGeneratedRoute || !lastGeneratedRoute.route_options) return;
    const option = lastGeneratedRoute.route_options[replacementBaseRouteIndex];
    if (!option || !(option.route || []).length) return;

    if (replacementCutStartIndex !== null) {
        const p = option.route[replacementCutStartIndex];
        const marker = L.circleMarker([Number(p.lat), Number(p.lon)], {
            radius: 7,
            weight: 3,
            color: "#f97316",
            fillColor: "#fff7ed",
            fillOpacity: 1,
            interactive: false
        }).addTo(map).bindTooltip("Replacement start");
        replacementLayers.push(marker);
    }

    if (replacementCutStartIndex !== null && replacementCutEndIndex !== null) {
        const cutLow = Math.min(replacementCutStartIndex, replacementCutEndIndex);
        const cutHigh = Math.max(replacementCutStartIndex, replacementCutEndIndex);
        const section = option.route
            .slice(cutLow, cutHigh + 1)
            .map(point => [Number(point.lat), Number(point.lon)]);
        if (section.length >= 2) {
            const orange = L.polyline(section, {
                color: "#f97316",
                weight: 10,
                opacity: 0.95,
                lineCap: "round",
                interactive: false
            }).addTo(map).bindTooltip("Section to replace");
            replacementLayers.push(orange);
        }
        const p = option.route[replacementCutEndIndex];
        const marker = L.circleMarker([Number(p.lat), Number(p.lon)], {
            radius: 7,
            weight: 3,
            color: "#f97316",
            fillColor: "#fff7ed",
            fillOpacity: 1,
            interactive: false
        }).addTo(map).bindTooltip("Replacement end");
        replacementLayers.push(marker);
    }

    replacementTrailSegments.forEach((item, index) => {
        const green = L.polyline(item.geometry, {
            color: "#16a34a",
            weight: 9,
            opacity: 0.95,
            lineCap: "round",
            interactive: false
        }).addTo(map).bindTooltip(`Guide trail ${index + 1}`);
        replacementLayers.push(green);

        const guideMarker = L.circleMarker([Number(item.lat), Number(item.lon)], {
            radius: 6,
            weight: 2,
            color: "#166534",
            fillColor: "#dcfce7",
            fillOpacity: 1,
            interactive: false
        }).addTo(map).bindTooltip(`Guide point ${index + 1}`, {permanent: false});
        replacementLayers.push(guideMarker);
    });
}


function cancelRouteSectionReplacement(showMessage = false) {
    routeReplacementMode = false;
    routeReplacementStage = "idle";
    replacementBaseRouteIndex = null;
    replacementCutStartIndex = null;
    replacementCutEndIndex = null;
    replacementTrailSegments = [];
    clearRouteReplacementLayers();
    updateRouteReplacementControls();
    if (showMessage) {
        document.getElementById("routeReplacementStatus").textContent =
            "Replacement selection cleared. The current route was not changed.";
    }
}


function beginRouteSectionReplacement() {
    const selected = getSelectedRouteOption();
    if (!selected) return;
    clearPlacementModes();
    cancelRouteSectionReplacement(false);
    routeReplacementMode = true;
    routeReplacementStage = "cut-start";
    replacementBaseRouteIndex = selectedRouteOptionIndex;
    updateRouteReplacementControls();
    drawRouteReplacementLayers();
}


function handleRouteReplacementRouteClick(latlng, routeIndex) {
    if (!routeReplacementMode || routeIndex !== replacementBaseRouteIndex) return;
    const option = lastGeneratedRoute?.route_options?.[routeIndex];
    if (!option) return;
    const index = nearestRouteCoordinateIndex(option, latlng);
    if (index === null) return;

    if (routeReplacementStage === "cut-start") {
        replacementCutStartIndex = index;
        replacementCutEndIndex = null;
        replacementTrailSegments = [];
        routeReplacementStage = "guide";
        updateRouteReplacementControls();
        drawRouteReplacementLayers();
        return;
    }

    // V33: after the first red-route click, route clicks no longer choose the
    // end. The end is inferred from the final selected green trail piece when
    // the user presses Replace section. This keeps the editor to two buttons.
    if (routeReplacementStage === "guide") return;
}


function addReplacementTrailSegment(latlng) {
    if (!routeReplacementMode || routeReplacementStage !== "guide") return;
    const nearest = findNearestOverlayTrailSegment(latlng);
    if (!nearest || nearest.distancePx > 22) {
        document.getElementById("routeReplacementStatus").textContent =
            "No gray trail was close enough. Zoom in and click directly on the trail you want to use.";
        return;
    }

    const signature = trailGeometrySignature(nearest.geometry);
    const duplicate = replacementTrailSegments.some(item =>
        trailGeometrySignature(item.geometry) === signature
    );
    if (duplicate) {
        document.getElementById("routeReplacementStatus").textContent =
            "That trail piece is already one of your green guide clicks.";
        return;
    }

    replacementTrailSegments.push({
        lat: Number(nearest.lat.toFixed(7)),
        lon: Number(nearest.lon.toFixed(7)),
        geometry: nearest.geometry,
        tile_id: nearest.tile_id,
        edge_u: nearest.edge_u,
        edge_v: nearest.edge_v,
        edge_key: nearest.edge_key
    });
    updateRouteReplacementControls();
    drawRouteReplacementLayers();
}


function undoReplacementTrailSegment() {
    if (!replacementTrailSegments.length) return;
    replacementTrailSegments.pop();
    if (routeReplacementStage === "ready") {
        replacementCutEndIndex = null;
        routeReplacementStage = "guide";
    }
    updateRouteReplacementControls();
    drawRouteReplacementLayers();
}


function closestPointsBetweenEditSegments(a0, a1, b0, b1) {
    // Work in Leaflet layer pixels so both polylines use the same local planar
    // coordinate system. We convert the winning points back to lat/lon and use
    // map.distance() for the final meter distance.
    const A0 = map.latLngToLayerPoint(L.latLng(Number(a0.lat), Number(a0.lon)));
    const A1 = map.latLngToLayerPoint(L.latLng(Number(a1.lat), Number(a1.lon)));
    const B0 = map.latLngToLayerPoint(L.latLng(Number(b0.lat), Number(b0.lon)));
    const B1 = map.latLngToLayerPoint(L.latLng(Number(b1.lat), Number(b1.lon)));

    const ux = A1.x - A0.x;
    const uy = A1.y - A0.y;
    const vx = B1.x - B0.x;
    const vy = B1.y - B0.y;
    const wx = A0.x - B0.x;
    const wy = A0.y - B0.y;

    const a = ux * ux + uy * uy;
    const b = ux * vx + uy * vy;
    const c = vx * vx + vy * vy;
    const d = ux * wx + uy * wy;
    const e = vx * wx + vy * wy;
    const EPS = 1e-12;

    let sN;
    let sD = a * c - b * b;
    let tN;
    let tD = sD;

    if (a <= EPS && c <= EPS) {
        sN = 0;
        sD = 1;
        tN = 0;
        tD = 1;
    } else if (a <= EPS) {
        sN = 0;
        sD = 1;
        tN = e;
        tD = c;
    } else if (c <= EPS) {
        tN = 0;
        tD = 1;
        sN = -d;
        sD = a;
    } else {
        if (sD < EPS) {
            sN = 0;
            sD = 1;
            tN = e;
            tD = c;
        } else {
            sN = b * e - c * d;
            tN = a * e - b * d;

            if (sN < 0) {
                sN = 0;
                tN = e;
                tD = c;
            } else if (sN > sD) {
                sN = sD;
                tN = e + b;
                tD = c;
            }
        }

        if (tN < 0) {
            tN = 0;
            if (-d < 0) {
                sN = 0;
                sD = 1;
            } else if (-d > a) {
                sN = sD;
            } else {
                sN = -d;
                sD = a;
            }
        } else if (tN > tD) {
            tN = tD;
            if ((-d + b) < 0) {
                sN = 0;
                sD = 1;
            } else if ((-d + b) > a) {
                sN = sD;
            } else {
                sN = -d + b;
                sD = a;
            }
        }
    }

    const s = Math.max(0, Math.min(1, Math.abs(sN) < EPS ? 0 : sN / sD));
    const t = Math.max(0, Math.min(1, Math.abs(tN) < EPS ? 0 : tN / tD));

    const ap = L.point(A0.x + s * ux, A0.y + s * uy);
    const bp = L.point(B0.x + t * vx, B0.y + t * vy);
    const aLL = map.layerPointToLatLng(ap);
    const bLL = map.layerPointToLatLng(bp);

    return {
        routeT: s,
        greenT: t,
        routePoint: {lat: Number(aLL.lat), lon: Number(aLL.lng)},
        greenPoint: {lat: Number(bLL.lat), lon: Number(bLL.lng)},
        distanceMeters: map.distance(aLL, bLL)
    };
}


function inferReplacementRejoinIndex() {
    if (
        replacementBaseRouteIndex === null ||
        replacementCutStartIndex === null ||
        !replacementTrailSegments.length ||
        !lastGeneratedRoute
    ) return null;

    const option = lastGeneratedRoute.route_options?.[replacementBaseRouteIndex];
    if (!option || !option.route || option.route.length < 2) return null;

    const lastSelected =
        replacementTrailSegments[replacementTrailSegments.length - 1];

    const geometry = (lastSelected.geometry || [])
        .map(p => ({lat: Number(p[0]), lon: Number(p[1])}))
        .filter(p => Number.isFinite(p.lat) && Number.isFinite(p.lon));

    if (geometry.length < 2) return null;

    const route = option.route.map(p => ({
        lat: Number(p.lat),
        lon: Number(p.lon)
    }));

    const n = route.length;
    const minIndexGap = Math.max(4, Math.round(n * 0.006));
    const MAX_REJOIN_DISTANCE_M = 18;

    let best = null;

    // Compare EVERY segment of the final green trail edge with EVERY eligible
    // segment of the red route. This fixes the old bug where only stored
    // vertices were compared and a mid-edge intersection could be missed.
    for (let ri = 0; ri < route.length - 1; ri++) {
        const routeMidIndex = ri + 0.5;
        const directGap = Math.abs(routeMidIndex - replacementCutStartIndex);
        const cyclicGap = Math.min(
            directGap,
            Math.max(0, n - 1 - directGap)
        );
        if (cyclicGap < minIndexGap) continue;

        for (let gi = 0; gi < geometry.length - 1; gi++) {
            const closest = closestPointsBetweenEditSegments(
                route[ri],
                route[ri + 1],
                geometry[gi],
                geometry[gi + 1]
            );

            if (closest.distanceMeters > MAX_REJOIN_DISTANCE_M) continue;

            const greenPosition = gi + closest.greenT;
            const routePosition = ri + closest.routeT;

            // Prefer the FIRST valid reconnection encountered along the final
            // selected green edge. Distance breaks near-ties.
            const candidate = {
                index: closest.routeT < 0.5 ? ri : ri + 1,
                routeSegmentIndex: ri,
                routeT: closest.routeT,
                routePosition,
                greenSegmentIndex: gi,
                greenT: closest.greenT,
                greenPosition,
                routePoint: closest.routePoint,
                greenPoint: closest.greenPoint,
                distanceMeters: closest.distanceMeters
            };

            if (
                !best ||
                candidate.greenPosition < best.greenPosition - 1e-6 ||
                (
                    Math.abs(candidate.greenPosition - best.greenPosition) <= 1e-6 &&
                    candidate.distanceMeters < best.distanceMeters
                )
            ) {
                best = candidate;
            }
        }
    }

    return best;
}

function routeEditDistanceMeters(a, b) {
    return map.distance(
        L.latLng(Number(a.lat), Number(a.lon)),
        L.latLng(Number(b.lat), Number(b.lon))
    );
}


function nearestPointOnEditPolyline(points, target) {
    if (!points || points.length < 2) return null;
    const targetLL = L.latLng(Number(target.lat), Number(target.lon));
    const targetPx = map.latLngToLayerPoint(targetLL);
    let best = null;

    for (let i = 0; i < points.length - 1; i++) {
        const aLL = L.latLng(Number(points[i].lat), Number(points[i].lon));
        const bLL = L.latLng(Number(points[i + 1].lat), Number(points[i + 1].lon));
        const a = map.latLngToLayerPoint(aLL);
        const b = map.latLngToLayerPoint(bLL);
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const denom = dx * dx + dy * dy;
        let t = denom > 0
            ? ((targetPx.x - a.x) * dx + (targetPx.y - a.y) * dy) / denom
            : 0;
        t = Math.max(0, Math.min(1, t));

        const px = a.x + t * dx;
        const py = a.y + t * dy;
        const projectedLL = map.layerPointToLatLng(L.point(px, py));
        const distanceMeters = map.distance(targetLL, projectedLL);

        if (!best || distanceMeters < best.distanceMeters) {
            best = {
                segmentIndex: i,
                t,
                position: i + t,
                distanceMeters,
                point: {
                    lat: Number(projectedLL.lat),
                    lon: Number(projectedLL.lng)
                }
            };
        }
    }
    return best;
}


function orientGreenGeometries(startPoint, endPoint) {
    const raw = replacementTrailSegments.map(item =>
        (item.geometry || []).map(p => ({
            lat: Number(p[0]),
            lon: Number(p[1])
        }))
    );

    if (!raw.length || raw.some(g => g.length < 2)) {
        throw new Error("One of the selected green trail pieces has no usable geometry.");
    }

    // Dynamic programming: each selected edge has two possible orientations.
    // Choose the orientation sequence with the smallest endpoint connection gaps.
    const n = raw.length;
    const dp = Array.from({length: n}, () => [null, null]);

    for (let orientation = 0; orientation < 2; orientation++) {
        const g = orientation === 0 ? raw[0] : [...raw[0]].reverse();
        dp[0][orientation] = {
            cost: routeEditDistanceMeters(startPoint, g[0]),
            prev: null
        };
    }

    for (let i = 1; i < n; i++) {
        for (let orientation = 0; orientation < 2; orientation++) {
            const current = orientation === 0 ? raw[i] : [...raw[i]].reverse();
            let best = null;

            for (let prevOrientation = 0; prevOrientation < 2; prevOrientation++) {
                const prevGeom = prevOrientation === 0
                    ? raw[i - 1]
                    : [...raw[i - 1]].reverse();
                const gap = routeEditDistanceMeters(
                    prevGeom[prevGeom.length - 1],
                    current[0]
                );
                const cost = dp[i - 1][prevOrientation].cost + gap;
                if (!best || cost < best.cost) {
                    best = {cost, prev: prevOrientation};
                }
            }
            dp[i][orientation] = best;
        }
    }

    let lastOrientation = 0;
    let bestFinal = Infinity;
    for (let orientation = 0; orientation < 2; orientation++) {
        const g = orientation === 0 ? raw[n - 1] : [...raw[n - 1]].reverse();
        const cost = dp[n - 1][orientation].cost
            + routeEditDistanceMeters(g[g.length - 1], endPoint);
        if (cost < bestFinal) {
            bestFinal = cost;
            lastOrientation = orientation;
        }
    }

    const orientations = Array(n);
    orientations[n - 1] = lastOrientation;
    for (let i = n - 1; i > 0; i--) {
        orientations[i - 1] = dp[i][orientations[i]].prev;
    }

    return raw.map((g, i) =>
        orientations[i] === 0 ? g : [...g].reverse()
    );
}


function buildDirectReplacementGeometry(startPoint, endPoint) {
    const oriented = orientGreenGeometries(startPoint, endPoint);
    const MAX_ATTACH_GAP_M = 45;
    const MAX_BETWEEN_GAP_M = 35;

    // Verify consecutive selected trail pieces really connect.
    for (let i = 1; i < oriented.length; i++) {
        const gap = routeEditDistanceMeters(
            oriented[i - 1][oriented[i - 1].length - 1],
            oriented[i][0]
        );
        if (gap > MAX_BETWEEN_GAP_M) {
            throw new Error(
                `Green trail piece ${i} does not connect to green trail piece ${i + 1} (${Math.round(gap)} m gap). Select the missing trail piece.`
            );
        }
    }

    const firstProjection = nearestPointOnEditPolyline(oriented[0], startPoint);
    const lastProjection = nearestPointOnEditPolyline(
        oriented[oriented.length - 1],
        endPoint
    );

    if (!firstProjection || firstProjection.distanceMeters > MAX_ATTACH_GAP_M) {
        throw new Error(
            `The first green trail piece is ${Math.round(firstProjection?.distanceMeters ?? 999)} m from the replacement start. Start with the trail piece that actually leaves the red route.`
        );
    }
    if (!lastProjection || lastProjection.distanceMeters > MAX_ATTACH_GAP_M) {
        throw new Error(
            `The last green trail piece is ${Math.round(lastProjection?.distanceMeters ?? 999)} m from the red route. Keep selecting until the green corridor actually reconnects.`
        );
    }

    let replacement = [];

    if (oriented.length === 1) {
        let g = oriented[0];
        let a = firstProjection;
        let b = lastProjection;

        // If the chosen orientation places the end before the start, flip it.
        if (b.position < a.position) {
            g = [...g].reverse();
            a = nearestPointOnEditPolyline(g, startPoint);
            b = nearestPointOnEditPolyline(g, endPoint);
        }
        if (!a || !b || b.position < a.position) {
            throw new Error("The selected green trail piece cannot connect the chosen start and rejoin points in one direction.");
        }

        replacement = [a.point];
        for (let i = a.segmentIndex + 1; i <= b.segmentIndex; i++) {
            replacement.push({...g[i]});
        }
        replacement.push(b.point);
    } else {
        // Trim the first edge so we start exactly where it leaves the red route.
        replacement.push(firstProjection.point);
        for (let i = firstProjection.segmentIndex + 1; i < oriented[0].length; i++) {
            replacement.push({...oriented[0][i]});
        }

        // Middle selected edges are inserted verbatim.
        for (let gi = 1; gi < oriented.length - 1; gi++) {
            const g = oriented[gi];
            if (
                replacement.length &&
                routeEditDistanceMeters(replacement[replacement.length - 1], g[0]) <= MAX_BETWEEN_GAP_M
            ) {
                replacement.push(...g.map(p => ({...p})));
            }
        }

        // Trim the final edge at the red-route rejoin.
        const last = oriented[oriented.length - 1];
        if (
            replacement.length &&
            routeEditDistanceMeters(replacement[replacement.length - 1], last[0]) <= MAX_BETWEEN_GAP_M
        ) {
            for (let i = 0; i <= lastProjection.segmentIndex; i++) {
                replacement.push({...last[i]});
            }
            replacement.push(lastProjection.point);
        }
    }

    // Remove adjacent duplicate coordinates.
    const clean = [];
    for (const point of replacement) {
        if (
            clean.length &&
            routeEditDistanceMeters(clean[clean.length - 1], point) < 0.15
        ) {
            clean[clean.length - 1] = {...point};
        } else {
            clean.push({...point});
        }
    }

    clean[0] = {...startPoint};
    clean[clean.length - 1] = {...endPoint};
    return clean;
}


function directRouteDistanceMiles(route) {
    let meters = 0;
    for (let i = 1; i < route.length; i++) {
        meters += routeEditDistanceMeters(route[i - 1], route[i]);
    }
    return meters / 1609.344;
}


async function applyRouteSectionReplacement() {
    if (
        replacementBaseRouteIndex === null ||
        replacementCutStartIndex === null ||
        routeReplacementStage !== "guide" ||
        !replacementTrailSegments.length ||
        !lastGeneratedRoute
    ) return;

    const inferredRejoin = inferReplacementRejoinIndex();
    const status = document.getElementById("routeReplacementStatus");

    if (!inferredRejoin) {
        status.textContent =
            "The last green trail piece does not reach the red route yet. Keep selecting trail pieces until the green path reconnects, then click Replace section.";
        return;
    }

    replacementCutEndIndex = inferredRejoin.index;
    if (Math.abs(replacementCutEndIndex - replacementCutStartIndex) < 2) {
        status.textContent =
            "The green corridor reconnects too close to where it started. Keep selecting farther along the trail.";
        replacementCutEndIndex = null;
        return;
    }

    routeReplacementStage = "ready";
    status.textContent =
        `Rejoin detected ${inferredRejoin.distanceMeters.toFixed(1)} m from the red route. Replacing section...`;
    drawRouteReplacementLayers();

    const baseOption = lastGeneratedRoute.route_options[replacementBaseRouteIndex];
    if (!baseOption || !(baseOption.route || []).length) return;

    const selectedStartIndex = replacementCutStartIndex;
    const selectedEndIndex = replacementCutEndIndex;
    const cutA = Math.min(selectedStartIndex, selectedEndIndex);
    const cutB = Math.max(selectedStartIndex, selectedEndIndex);
    const selectionReversed = selectedStartIndex > selectedEndIndex;

    const selectedStart = {
        lat: Number(baseOption.route[selectedStartIndex].lat),
        lon: Number(baseOption.route[selectedStartIndex].lon)
    };
    // V43: the final green edge can rejoin the red route in the MIDDLE of
    // both polylines. Use the exact projected route-side rejoin point instead
    // of forcing the edit to a stored route vertex.
    const selectedEnd = inferredRejoin.routePoint
        ? {
            lat: Number(inferredRejoin.routePoint.lat),
            lon: Number(inferredRejoin.routePoint.lon)
        }
        : {
            lat: Number(baseOption.route[selectedEndIndex].lat),
            lon: Number(baseOption.route[selectedEndIndex].lon)
        };

    applyReplacementButton.disabled = true;
    replaceSectionButton.disabled = true;
    generateButton.disabled = true;
    findMoreButton.disabled = true;

    try {
        // V42: no route generation happens here. The exact green geometries
        // highlighted on screen are literally spliced into the selected route.
        let replacement = buildDirectReplacementGeometry(
            selectedStart,
            selectedEnd
        );

        if (selectionReversed) replacement = [...replacement].reverse();
        replacement[0] = {
            lat: Number(baseOption.route[cutA].lat),
            lon: Number(baseOption.route[cutA].lon)
        };
        // Preserve the exact mid-segment rejoin. Do not snap it back to the
        // nearest stored red-route coordinate.
        replacement[replacement.length - 1] = {
            lat: Number(selectedEnd.lat),
            lon: Number(selectedEnd.lon)
        };

        const stitched = baseOption.route
            .slice(0, cutA + 1)
            .map(p => ({lat: Number(p.lat), lon: Number(p.lon)}));

        for (const point of replacement.slice(1)) {
            const last = stitched[stitched.length - 1];
            if (last && routeEditDistanceMeters(last, point) < 0.15) {
                stitched[stitched.length - 1] = {...point};
            } else {
                stitched.push({...point});
            }
        }

        const suffixStartIndex = (
            !selectionReversed &&
            Number.isInteger(inferredRejoin.routeSegmentIndex)
        )
            ? inferredRejoin.routeSegmentIndex + 1
            : cutB + 1;

        for (const p of baseOption.route.slice(suffixStartIndex)) {
            const point = {lat: Number(p.lat), lon: Number(p.lon)};
            const last = stitched[stitched.length - 1];
            if (last && routeEditDistanceMeters(last, point) < 0.15) {
                stitched[stitched.length - 1] = point;
            } else {
                stitched.push(point);
            }
        }

        const editedIndex = replacementBaseRouteIndex;
        const provisional = {
            ...baseOption,
            route: stitched,
            gpx_export_points: stitched.map(p => ({
                lat: Number(p.lat),
                lon: Number(p.lon)
            })),
            actual_distance_miles: Number(directRouteDistanceMiles(stitched).toFixed(2)),
            distance_error_miles: Number(
                Math.abs(
                    directRouteDistanceMiles(stitched) -
                    Number(getInputData().target_distance_miles)
                ).toFixed(2)
            ),
            elevation_profile: [],
            is_edited: true,
            edit_type: "section-replacement-direct-splice",
            route_signature: `direct-${Date.now()}-${stitched.length}`
        };

        provisional.name = baseOption.name || "Edited route";
        provisional.option_index = editedIndex + 1;
        lastGeneratedRoute.route_options[editedIndex] = provisional;
        reindexRouteOptions(lastGeneratedRoute.route_options);
        lastGeneratedRoute.route_options_count = lastGeneratedRoute.route_options.length;

        // IMPORTANT: update the red route immediately, before any server work.
        cancelRouteSectionReplacement(false);
        selectedRouteOptionIndex = editedIndex;
        renderRouteResults("Route section replaced");
        drawRouteOptions(lastGeneratedRoute, editedIndex, false);
        selectRouteOption(editedIndex, false);
        status.textContent =
            "Section replaced immediately. Refreshing elevation and route statistics...";

        // Metrics/elevation are refreshed afterward. Failure here does NOT undo
        // the edit the user can already see.
        const input = getInputData();
        const response = await fetch("/recalculate-edited-route", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
                start_lat: input.start_lat,
                start_lon: input.start_lon,
                target_distance_miles: input.target_distance_miles,
                target_gain_ft: input.target_gain_ft,
                current_route: stitched,
                avoid_areas: input.avoid_areas,
                avoid_segments: input.avoid_segments,
                prefer_segments: input.prefer_segments
            })
        });

        const result = await readJsonResponse(response);
        const recalculated = result.option;
        if (recalculated && (recalculated.route || []).length) {
            recalculated.name = provisional.name;
            recalculated.option_index = editedIndex + 1;
            recalculated.is_edited = true;
            recalculated.edit_type = "section-replacement-direct-splice";
            lastGeneratedRoute.route_options[editedIndex] = recalculated;
            reindexRouteOptions(lastGeneratedRoute.route_options);
            renderRouteResults("Route section replaced");
            drawRouteOptions(lastGeneratedRoute, editedIndex, false);
            selectRouteOption(editedIndex, false);
            status.textContent =
                "Section replaced. Elevation and route statistics refreshed.";
        }
    } catch (error) {
        // If the direct geometry validation fails before splicing, keep the
        // green selection so the user can add the missing piece.
        if (routeReplacementStage !== "idle") {
            routeReplacementStage = "guide";
            replacementCutEndIndex = null;
            status.textContent = "Replacement error: " + error.message;
            updateRouteReplacementControls();
            drawRouteReplacementLayers();
        } else {
            // The visible route has already been changed; only metric refresh failed.
            status.textContent =
                "Section replaced, but statistics could not refresh: " + error.message;
        }
    } finally {
        generateButton.disabled = false;
        findMoreButton.disabled = !lastGeneratedRoute;
        replaceSectionButton.disabled = !getSelectedRouteOption();
        if (routeReplacementStage !== "idle") updateRouteReplacementControls();
    }
}

function beginRouteEditMode() {
    if (!getSelectedRouteOption()) return;
    clearPlacementModes();
    routeEditMode = true;
    editRouteButton.textContent = "Click the red route...";
    document.getElementById("routeEditStatus").textContent =
        "Click the selected red route where you want an edit handle, then drag that handle onto another trail corridor.";
}


function addRouteEditPointFromRoute(latlng) {
    if (routeEditPoints.length >= 5) {
        document.getElementById("routeEditStatus").textContent = "Maximum of 5 direct route edit points reached.";
        routeEditMode = false;
        editRouteButton.textContent = "Edit selected route";
        return;
    }
    routeEditPoints.push({
        id: nextRouteEditId++,
        lat: Number(latlng.lat.toFixed(7)),
        lon: Number(latlng.lng.toFixed(7)),
        tolerance_miles: 0.03
    });
    routeEditMode = false;
    editRouteButton.textContent = "Edit selected route";
    renderRouteEditRows();
    drawRouteEditLayers();
    commitPlannerState();
    document.getElementById("routeEditStatus").textContent =
        "Edit handle added. Drag it onto the trail corridor you want; rerouting starts when you release it.";
}


function renderRouteEditRows() {
    const container = document.getElementById("routeEditRows");
    if (!container) return;
    container.innerHTML = "";
    routeEditPoints.forEach((point, index) => {
        const row = document.createElement("div");
        row.className = "route-edit-row";
        row.innerHTML = `
            <span>Edit ${index + 1} · ${Number(point.lat).toFixed(5)}, ${Number(point.lon).toFixed(5)}</span>
            <button type="button" class="route-edit-remove" data-route-edit-id="${point.id}">Remove</button>
        `;
        container.appendChild(row);
    });
    container.querySelectorAll("button[data-route-edit-id]").forEach(button => {
        button.addEventListener("click", () => {
            const id = Number(button.dataset.routeEditId);
            routeEditPoints = routeEditPoints.filter(item => item.id !== id);
            renderRouteEditRows();
            drawRouteEditLayers();
            commitPlannerState();
            document.getElementById("routeEditStatus").textContent =
                "Edit point removed. Generate again to recalculate without it.";
        });
    });
    clearRouteEditsButton.disabled = routeEditPoints.length === 0;
}


function drawRouteEditLayers() {
    for (const layer of routeEditLayers) {
        if (layer && map.hasLayer(layer)) map.removeLayer(layer);
    }
    routeEditLayers = [];
    routeEditPoints.forEach((point, index) => {
        const marker = L.marker([Number(point.lat), Number(point.lon)], {
            draggable: true,
            title: `Route edit ${index + 1}`
        }).addTo(map).bindTooltip(`Route edit ${index + 1} · drag to reroute`);
        marker.on("dragstart", event => {
            const start = event.target.getLatLng();
            event.target._routeEditDragOrigin = L.latLng(start.lat, start.lng);
        });
        marker.on("dragend", async event => {
            const markerLayer = event.target;
            const moved = markerLayer.getLatLng();
            const nearest = findNearestOverlayTrailSegment(moved);
            if (!nearest || nearest.distancePx > 40) {
                if (markerLayer._routeEditDragOrigin) markerLayer.setLatLng(markerLayer._routeEditDragOrigin);
                document.getElementById("routeEditStatus").textContent =
                    "No trail was close enough. Zoom in and drag the handle directly onto the desired trail.";
                return;
            }
            point.lat = Number(nearest.lat.toFixed(7));
            point.lon = Number(nearest.lon.toFixed(7));
            markerLayer.setLatLng([point.lat, point.lon]);
            renderRouteEditRows();
            commitPlannerState();
            await rerouteThroughEditPoints();
        });
        routeEditLayers.push(marker);
    });
    clearRouteEditsButton.disabled = routeEditPoints.length === 0;
}


function clearRouteEditPoints() {
    if (!routeEditPoints.length) return;
    routeEditPoints = [];
    renderRouteEditRows();
    drawRouteEditLayers();
    commitPlannerState();
    document.getElementById("routeEditStatus").textContent =
        "Direct edit points cleared. Existing route choices are kept; Generate recalculates without edit constraints.";
}


async function rerouteThroughEditPoints() {
    if (!lastGeneratedRoute || !routeEditPoints.length) return;
    const status = document.getElementById("routeEditStatus");
    const data = getInputData();
    currentSearchSeed = Math.floor((Date.now() + Math.random() * 1000000) % 2147483647);
    data.search_seed = currentSearchSeed;
    generateButton.disabled = true;
    findMoreButton.disabled = true;
    editRouteButton.disabled = true;
    status.textContent = "Rerouting selected route through the dragged trail corridor...";
    try {
        const result = await requestRouteBatch(data);
        const incoming = ensureRouteOptions(result);
        if (!incoming.length) throw new Error("No edited route was returned.");
        const edited = incoming.find(option => routePassesQualityFilters(option)) || incoming[0];
        edited.is_edited = true;
        const signature = browserRouteSignature(edited);
        let selectedIndex = (lastGeneratedRoute.route_options || []).findIndex(option => browserRouteSignature(option) === signature);
        if (selectedIndex < 0) {
            lastGeneratedRoute.route_options.push(edited);
            reindexRouteOptions(lastGeneratedRoute.route_options);
            selectedIndex = lastGeneratedRoute.route_options.length - 1;
        }
        lastGeneratedRoute.route_options_count = lastGeneratedRoute.route_options.length;
        lastGeneratedRoute.route_edit_points_count = routeEditPoints.length;
        lastSearchConfigKey = searchConfigKey(data);
        renderRouteResults("Route edited through dragged corridor");
        if (routePassesQualityFilters(lastGeneratedRoute.route_options[selectedIndex])) {
            drawRouteOptions(lastGeneratedRoute, selectedIndex, false);
            selectRouteOption(selectedIndex, false);
            status.textContent = "Edited route added and selected.";
        } else {
            drawRouteOptions(lastGeneratedRoute, selectedRouteOptionIndex, false);
            status.textContent = "Edited route was added, but your quality filters currently hide it.";
        }
    } catch (error) {
        status.textContent = "Route edit error: " + error.message;
    } finally {
        generateButton.disabled = false;
        findMoreButton.disabled = !lastGeneratedRoute;
        editRouteButton.disabled = !getSelectedRouteOption();
    }
}


function resetTrailSegmentPlacementButtons() {
    avoidTrailSegmentButton.textContent = "Avoid trail segment";
    preferTrailSegmentButton.textContent = "Prefer trail segment";
}


function clearPlacementModes() {
    if (routeReplacementStage !== "idle") {
        cancelRouteSectionReplacement(false);
    }
    startPointPlacementMode = false;
    passPointPlacementMode = false;
    avoidAreaPlacementMode = false;
    trailSegmentPlacementMode = null;
    routeEditMode = false;
    chooseStartButton.textContent = "Choose start on map";
    addPassPointButton.textContent = "Add pass-through point";
    addAvoidAreaButton.textContent = "Add avoid area";
    editRouteButton.textContent = "Edit selected route";
    resetTrailSegmentPlacementButtons();
}


function beginStartPointPlacement() {
    clearPlacementModes();
    startPointPlacementMode = true;
    chooseStartButton.textContent = "Click map to place start...";
    document.getElementById("startPointStatus").textContent =
        "Click anywhere on the map to choose the starting point.";
}


function beginPassPointPlacement() {
    clearPlacementModes();
    if (passPoints.length >= 5) {
        document.getElementById("passPointStatus").textContent = "Maximum of 5 pass-through points reached.";
        return;
    }
    passPointPlacementMode = true;
    addPassPointButton.textContent = "Click map to place point...";
    document.getElementById("passPointStatus").textContent =
        "Click anywhere on the map. The route will use a nearby natural trail inside the tolerance circle.";
}


function beginAvoidAreaPlacement() {
    clearPlacementModes();
    if (avoidAreas.length >= 5) {
        document.getElementById("avoidAreaStatus").textContent = "Maximum of 5 avoid areas reached.";
        return;
    }
    avoidAreaPlacementMode = true;
    addAvoidAreaButton.textContent = "Click map to place area...";
    document.getElementById("avoidAreaStatus").textContent =
        "Click the center of the area you want the generated route to avoid.";
}


function beginTrailSegmentPlacement(kind) {
    clearPlacementModes();
    const list = kind === "avoid" ? avoidSegments : preferSegments;
    const maxCount = 12;
    if (list.length >= maxCount) {
        document.getElementById("segmentStatus").textContent =
            `Maximum of ${maxCount} ${kind === "avoid" ? "avoided" : "preferred"} trail segments reached.`;
        return;
    }
    trailSegmentPlacementMode = kind;
    if (kind === "avoid") {
        avoidTrailSegmentButton.textContent = "Click gray trail to avoid...";
    } else {
        preferTrailSegmentButton.textContent = "Click gray trail to prefer...";
    }
    document.getElementById("segmentStatus").textContent =
        "Click directly on the gray trail segment you want to " + (kind === "avoid" ? "avoid." : "prefer.");
}


function invalidateAfterStartMove() {
    loadedWorkspaceStartKey = null;
    lastWorkspaceResult = null;
    lastGeneratedRoute = null;
    selectedRouteOptionIndex = 0;
    findMoreBatch = 0;
    downloadGpxButton.disabled = true;
    findMoreButton.disabled = true;
    clearGeneratedRouteLines();
    if (snappedStartMarker) map.removeLayer(snappedStartMarker);
    if (snapLine) map.removeLayer(snapLine);
    snappedStartMarker = null;
    snapLine = null;
}


function createRequestedStartMarker(lat, lon, popupText = "Selected start", openPopup = false) {
    if (requestedStartMarker && map.hasLayer(requestedStartMarker)) {
        map.removeLayer(requestedStartMarker);
    }
    requestedStartMarker = L.marker([lat, lon], {draggable: true})
        .addTo(map)
        .bindPopup(popupText);
    requestedStartMarker.on("dragend", event => {
        setStartPoint(event.target.getLatLng(), "Start moved. Generate a route or load the start area.");
    });
    if (openPopup) requestedStartMarker.openPopup();
}


function setStartPoint(latlng, statusText = "Start selected. Generate a route or load the start area.") {
    const startLatInput = document.getElementById("start_lat");
    const startLonInput = document.getElementById("start_lon");
    const endLatInput = document.getElementById("end_lat");
    const endLonInput = document.getElementById("end_lon");

    const oldStartLat = Number(startLatInput.value);
    const oldStartLon = Number(startLonInput.value);
    const oldEndLat = Number(endLatInput.value);
    const oldEndLon = Number(endLonInput.value);
    const endWasFollowingStart =
        Number.isFinite(oldStartLat) && Number.isFinite(oldStartLon) &&
        Number.isFinite(oldEndLat) && Number.isFinite(oldEndLon) &&
        Math.abs(oldEndLat - oldStartLat) < 0.000001 &&
        Math.abs(oldEndLon - oldStartLon) < 0.000001;

    const lat = Number(latlng.lat.toFixed(7));
    const lon = Number(latlng.lng.toFixed(7));
    startLatInput.value = lat;
    startLonInput.value = lon;
    if (endWasFollowingStart) {
        endLatInput.value = lat;
        endLonInput.value = lon;
    }

    invalidateAfterStartMove();
    createRequestedStartMarker(lat, lon, "Selected start · drag to adjust", true);
    clearPlacementModes();
    document.getElementById("startPointStatus").textContent = statusText;
    document.getElementById("results").innerHTML =
        '<span class="success">Start selected from map. Drag the marker to fine-tune it.</span>';
    commitPlannerState();
    refresh3DMapData();
}


function findNearestOverlayTrailSegment(latlng) {
    const records = (masterTrailRecords && masterTrailRecords.length)
        ? masterTrailRecords
        : (masterTrailSegments || []).map(segment => ({
            tile_id: null,
            u: null,
            v: null,
            key: null,
            geometry: segment
        }));

    if (!records.length) return null;

    const clickPoint = map.latLngToLayerPoint(latlng);
    let best = null;

    for (const record of records) {
        const segment = record.geometry || [];
        if (!segment || segment.length < 2) continue;

        for (let i = 0; i < segment.length - 1; i++) {
            const aLL = L.latLng(Number(segment[i][0]), Number(segment[i][1]));
            const bLL = L.latLng(Number(segment[i + 1][0]), Number(segment[i + 1][1]));
            const a = map.latLngToLayerPoint(aLL);
            const b = map.latLngToLayerPoint(bLL);
            const dx = b.x - a.x;
            const dy = b.y - a.y;
            const denom = dx * dx + dy * dy;
            let t = denom > 0
                ? ((clickPoint.x - a.x) * dx + (clickPoint.y - a.y) * dy) / denom
                : 0;
            t = Math.max(0, Math.min(1, t));

            const px = a.x + t * dx;
            const py = a.y + t * dy;
            const distPx = Math.hypot(clickPoint.x - px, clickPoint.y - py);

            if (!best || distPx < best.distancePx) {
                best = {
                    distancePx: distPx,
                    lat: aLL.lat + (bLL.lat - aLL.lat) * t,
                    lon: aLL.lng + (bLL.lng - aLL.lng) * t,
                    geometry: segment.map(point => [Number(point[0]), Number(point[1])]),
                    tile_id: record.tile_id ?? null,
                    edge_u: record.u === null || record.u === undefined ? null : Number(record.u),
                    edge_v: record.v === null || record.v === undefined ? null : Number(record.v),
                    edge_key: record.key === null || record.key === undefined ? null : String(record.key)
                };
            }
        }
    }
    return best;
}

function trailGeometrySignature(geometry) {
    if (!geometry || geometry.length < 2) return "";
    const first = geometry[0];
    const last = geometry[geometry.length - 1];
    const a = `${Number(first[0]).toFixed(6)},${Number(first[1]).toFixed(6)}`;
    const b = `${Number(last[0]).toFixed(6)},${Number(last[1]).toFixed(6)}`;
    return a < b ? `${a}|${b}` : `${b}|${a}`;
}


function addClickedTrailSegment(kind, latlng) {
    const nearest = findNearestOverlayTrailSegment(latlng);
    if (!nearest || nearest.distancePx > 22) {
        document.getElementById("segmentStatus").textContent =
            "No gray trail was close enough to that click. Zoom in and click directly on the trail.";
        return;
    }

    const list = kind === "avoid" ? avoidSegments : preferSegments;
    const opposite = kind === "avoid" ? preferSegments : avoidSegments;
    const chosenLatLng = L.latLng(nearest.lat, nearest.lon);
    const geometryKey = trailGeometrySignature(nearest.geometry);
    const duplicate = list.some(item =>
        trailGeometrySignature(item.geometry) === geometryKey ||
        map.distance(chosenLatLng, L.latLng(item.lat, item.lon)) < 3
    );
    if (duplicate) {
        document.getElementById("segmentStatus").textContent = "That trail segment is already selected.";
        return;
    }

    // If the same exact segment was previously selected in the opposite mode,
    // remove it there so the intent is never ambiguous.
    const filteredOpposite = opposite.filter(item =>
        trailGeometrySignature(item.geometry) !== geometryKey &&
        map.distance(chosenLatLng, L.latLng(item.lat, item.lon)) >= 3
    );
    if (kind === "avoid") {
        preferSegments = filteredOpposite;
    } else {
        avoidSegments = filteredOpposite;
    }

    list.push({
        id: nextTrailSegmentId++,
        lat: Number(nearest.lat.toFixed(7)),
        lon: Number(nearest.lon.toFixed(7)),
        geometry: nearest.geometry,
        tile_id: nearest.tile_id,
        edge_u: nearest.edge_u,
        edge_v: nearest.edge_v,
        edge_key: nearest.edge_key
    });

    trailSegmentPlacementMode = null;
    resetTrailSegmentPlacementButtons();
    renderTrailSegmentRows();
    drawTrailSegmentLayers();
    document.getElementById("segmentStatus").textContent =
        kind === "avoid"
            ? "Trail segment marked to avoid."
            : "Trail segment marked as preferred (soft preference only).";
    commitPlannerState();
}


function handleMapPlacementClick(event) {
    if (routeReplacementMode && routeReplacementStage === "guide") {
        addReplacementTrailSegment(event.latlng);
        return;
    }

    if (trailSegmentPlacementMode) {
        const kind = trailSegmentPlacementMode;
        addClickedTrailSegment(kind, event.latlng);
        return;
    }

    if (passPointPlacementMode) {
        passPointPlacementMode = false;
        addPassPointButton.textContent = "Add pass-through point";
        passPoints.push({
            id: nextPassPointId++,
            lat: Number(event.latlng.lat.toFixed(7)),
            lon: Number(event.latlng.lng.toFixed(7)),
            tolerance_miles: 0.25
        });
        renderPassPointRows();
        drawPassPointLayers();
        document.getElementById("passPointStatus").textContent =
            "Pass-through point added. Drag its marker or edit the values below.";
        commitPlannerState();
        return;
    }

    if (avoidAreaPlacementMode) {
        avoidAreaPlacementMode = false;
        addAvoidAreaButton.textContent = "Add avoid area";
        avoidAreas.push({
            id: nextAvoidAreaId++,
            lat: Number(event.latlng.lat.toFixed(7)),
            lon: Number(event.latlng.lng.toFixed(7)),
            radius_miles: 0.25
        });
        renderAvoidAreaRows();
        drawAvoidAreaLayers();
        document.getElementById("avoidAreaStatus").textContent =
            "Avoid area added. Drag its marker or edit the center/radius below.";
        commitPlannerState();
        return;
    }

    if (startPointPlacementMode) {
        setStartPoint(event.latlng);
    }
}


function renderTrailSegmentRows() {
    const container = document.getElementById("segmentRows");
    container.innerHTML = "";
    const rows = [
        ...avoidSegments.map(item => ({...item, kind: "avoid"})),
        ...preferSegments.map(item => ({...item, kind: "prefer"}))
    ];
    rows.forEach((item, index) => {
        const row = document.createElement("div");
        row.className = `segment-row ${item.kind}`;
        row.innerHTML = `
            <span class="segment-kind">${item.kind === "avoid" ? "Avoid" : "Prefer"} segment · ${Number(item.lat).toFixed(5)}, ${Number(item.lon).toFixed(5)}</span>
            <button type="button" class="segment-remove" data-segment-kind="${item.kind}" data-segment-id="${item.id}">Remove</button>
        `;
        container.appendChild(row);
    });
    container.querySelectorAll("button[data-segment-id]").forEach(button => {
        button.addEventListener("click", () => {
            const id = Number(button.dataset.segmentId);
            if (button.dataset.segmentKind === "avoid") {
                avoidSegments = avoidSegments.filter(item => item.id !== id);
            } else {
                preferSegments = preferSegments.filter(item => item.id !== id);
            }
            renderTrailSegmentRows();
            drawTrailSegmentLayers();
            commitPlannerState();
        });
    });
}


function drawTrailSegmentLayers() {
    for (const layer of trailSegmentLayers) {
        if (map.hasLayer(layer)) map.removeLayer(layer);
    }
    trailSegmentLayers = [];

    avoidSegments.forEach((item, index) => {
        const line = L.polyline(item.geometry, {
            color: "#ea580c",
            weight: 7,
            opacity: 0.9,
            dashArray: "8 6",
            interactive: false
        }).addTo(map).bindTooltip(`Avoid trail segment ${index + 1}`);
        trailSegmentLayers.push(line);
    });

    preferSegments.forEach((item, index) => {
        const line = L.polyline(item.geometry, {
            color: "#16a34a",
            weight: 7,
            opacity: 0.85,
            interactive: false
        }).addTo(map).bindTooltip(`Preferred trail segment ${index + 1}`);
        trailSegmentLayers.push(line);
    });
}


function renderPassPointRows() {
    const container = document.getElementById("passPointRows");
    container.innerHTML = "";
    passPoints.forEach((point, index) => {
        const row = document.createElement("div");
        row.className = "pass-point-row";
        row.innerHTML = `
            <div class="input-group">
                <label>Point ${index + 1} latitude</label>
                <input type="number" step="any" data-pass-id="${point.id}" data-field="lat" value="${point.lat}">
            </div>
            <div class="input-group">
                <label>Point ${index + 1} longitude</label>
                <input type="number" step="any" data-pass-id="${point.id}" data-field="lon" value="${point.lon}">
            </div>
            <div class="input-group">
                <label>Tolerance (mi)</label>
                <input type="number" min="0.01" max="5" step="0.05" data-pass-id="${point.id}" data-field="tolerance_miles" value="${point.tolerance_miles}">
            </div>
            <button type="button" class="pass-point-remove" data-remove-pass-id="${point.id}">Remove</button>
        `;
        container.appendChild(row);
    });

    container.querySelectorAll("input[data-pass-id]").forEach(input => {
        input.addEventListener("change", () => {
            const id = Number(input.dataset.passId);
            const point = passPoints.find(item => item.id === id);
            if (!point) return;
            point[input.dataset.field] = Number(input.value);
            drawPassPointLayers();
            commitPlannerState();
        });
    });

    container.querySelectorAll("button[data-remove-pass-id]").forEach(button => {
        button.addEventListener("click", () => {
            const id = Number(button.dataset.removePassId);
            passPoints = passPoints.filter(item => item.id !== id);
            renderPassPointRows();
            drawPassPointLayers();
            commitPlannerState();
        });
    });
}


function drawPassPointLayers() {
    for (const layer of passPointLayers) {
        if (map.hasLayer(layer)) map.removeLayer(layer);
    }
    passPointLayers = [];

    passPoints.forEach((point, index) => {
        const lat = Number(point.lat);
        const lon = Number(point.lon);
        const toleranceMiles = Number(point.tolerance_miles);
        if (!Number.isFinite(lat) || !Number.isFinite(lon) || !Number.isFinite(toleranceMiles)) return;

        const circle = L.circle([lat, lon], {
            radius: Math.max(1, toleranceMiles * 1609.344),
            color: "#7e22ce",
            weight: 2,
            opacity: 0.75,
            fillColor: "#a855f7",
            fillOpacity: 0.08
        }).addTo(map);
        const marker = L.marker([lat, lon], {draggable: true})
            .bindTooltip(`Required point ${index + 1} · drag to adjust`)
            .addTo(map);
        marker.on("dragend", event => {
            const moved = event.target.getLatLng();
            point.lat = Number(moved.lat.toFixed(7));
            point.lon = Number(moved.lng.toFixed(7));
            renderPassPointRows();
            drawPassPointLayers();
            commitPlannerState();
        });
        passPointLayers.push(circle, marker);
    });
}


function renderAvoidAreaRows() {
    const container = document.getElementById("avoidAreaRows");
    container.innerHTML = "";
    avoidAreas.forEach((area, index) => {
        const row = document.createElement("div");
        row.className = "avoid-area-row";
        row.innerHTML = `
            <div class="input-group">
                <label>Area ${index + 1} latitude</label>
                <input type="number" step="any" data-avoid-id="${area.id}" data-field="lat" value="${area.lat}">
            </div>
            <div class="input-group">
                <label>Area ${index + 1} longitude</label>
                <input type="number" step="any" data-avoid-id="${area.id}" data-field="lon" value="${area.lon}">
            </div>
            <div class="input-group">
                <label>Radius (mi)</label>
                <input type="number" min="0.01" max="10" step="0.05" data-avoid-id="${area.id}" data-field="radius_miles" value="${area.radius_miles}">
            </div>
            <button type="button" class="avoid-area-remove" data-remove-avoid-id="${area.id}">Remove</button>
        `;
        container.appendChild(row);
    });

    container.querySelectorAll("input[data-avoid-id]").forEach(input => {
        input.addEventListener("change", () => {
            const id = Number(input.dataset.avoidId);
            const area = avoidAreas.find(item => item.id === id);
            if (!area) return;
            area[input.dataset.field] = Number(input.value);
            drawAvoidAreaLayers();
            commitPlannerState();
        });
    });

    container.querySelectorAll("button[data-remove-avoid-id]").forEach(button => {
        button.addEventListener("click", () => {
            const id = Number(button.dataset.removeAvoidId);
            avoidAreas = avoidAreas.filter(item => item.id !== id);
            renderAvoidAreaRows();
            drawAvoidAreaLayers();
            commitPlannerState();
        });
    });
}


function drawAvoidAreaLayers() {
    for (const layer of avoidAreaLayers) {
        if (map.hasLayer(layer)) map.removeLayer(layer);
    }
    avoidAreaLayers = [];

    avoidAreas.forEach((area, index) => {
        const lat = Number(area.lat);
        const lon = Number(area.lon);
        const radiusMiles = Number(area.radius_miles);
        if (!Number.isFinite(lat) || !Number.isFinite(lon) || !Number.isFinite(radiusMiles)) return;

        const circle = L.circle([lat, lon], {
            radius: Math.max(1, radiusMiles * 1609.344),
            color: "#c2410c",
            weight: 2,
            opacity: 0.9,
            dashArray: "7 5",
            fillColor: "#f97316",
            fillOpacity: 0.12
        }).addTo(map);
        const marker = L.marker([lat, lon], {draggable: true})
            .bindTooltip(`Avoid area ${index + 1} · drag to adjust`)
            .addTo(map);
        marker.on("dragend", event => {
            const moved = event.target.getLatLng();
            area.lat = Number(moved.lat.toFixed(7));
            area.lon = Number(moved.lng.toFixed(7));
            renderAvoidAreaRows();
            drawAvoidAreaLayers();
            commitPlannerState();
        });
        avoidAreaLayers.push(circle, marker);
    });
}


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
        if (selectedRouteOptionIndex < 0 || selectedRouteOptionIndex >= options.length) return null;
        return options[selectedRouteOptionIndex];
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


function clearElevationHover() {
    if (elevationHoverMarker && map.hasLayer(elevationHoverMarker)) {
        map.removeLayer(elevationHoverMarker);
    }
    elevationHoverMarker = null;
    if (elevationRenderState) {
        const line = document.getElementById("profileHoverLine");
        const dot = document.getElementById("profileHoverDot");
        const label = document.getElementById("profileHoverLabel");
        if (line) line.setAttribute("visibility", "hidden");
        if (dot) dot.setAttribute("visibility", "hidden");
        if (label) label.setAttribute("visibility", "hidden");
    }
}


function clearGeneratedRouteLines() {
    clearElevationHover();
    for (const line of routeOptionLines) {
        if (line && map.hasLayer(line)) {
            map.removeLayer(line);
        }
    }
    routeOptionLines = [];

    if (routeLine && map.hasLayer(routeLine)) {
        map.removeLayer(routeLine);
    }
    routeLine = null;
}


function nearestProfileIndexToLatLng(profile, latlng) {
    let bestIndex = -1;
    let bestDistance = Infinity;
    for (let i = 0; i < profile.length; i++) {
        const p = profile[i];
        if (!Number.isFinite(Number(p.lat)) || !Number.isFinite(Number(p.lon))) continue;
        const d = map.distance(latlng, L.latLng(Number(p.lat), Number(p.lon)));
        if (d < bestDistance) {
            bestDistance = d;
            bestIndex = i;
        }
    }
    return bestIndex;
}


function updateElevationHover(index, showMapMarker = true) {
    if (!elevationRenderState) return;
    const {profile, x, y, margin, plotH} = elevationRenderState;
    if (index < 0 || index >= profile.length) return;
    const p = profile[index];
    const px = x(Number(p.distance_miles));
    const py = y(Number(p.elevation_ft));
    const hoverLine = document.getElementById("profileHoverLine");
    const hoverDot = document.getElementById("profileHoverDot");
    const hoverLabel = document.getElementById("profileHoverLabel");
    if (hoverLine) {
        hoverLine.setAttribute("x1", px);
        hoverLine.setAttribute("x2", px);
        hoverLine.setAttribute("y1", margin.top);
        hoverLine.setAttribute("y2", margin.top + plotH);
        hoverLine.setAttribute("visibility", "visible");
    }
    if (hoverDot) {
        hoverDot.setAttribute("cx", px);
        hoverDot.setAttribute("cy", py);
        hoverDot.setAttribute("visibility", "visible");
    }
    if (hoverLabel) {
        hoverLabel.setAttribute("x", Math.min(px + 8, elevationRenderState.width - 125));
        hoverLabel.setAttribute("y", Math.max(14, py - 8));
        hoverLabel.textContent = `${Number(p.distance_miles).toFixed(2)} mi · ${Math.round(Number(p.elevation_ft))} ft`;
        hoverLabel.setAttribute("visibility", "visible");
    }

    if (showMapMarker && Number.isFinite(Number(p.lat)) && Number.isFinite(Number(p.lon))) {
        const ll = [Number(p.lat), Number(p.lon)];
        if (!elevationHoverMarker) {
            elevationHoverMarker = L.circleMarker(ll, {
                radius: 6,
                weight: 2,
                color: "#111827",
                fillColor: "#ffffff",
                fillOpacity: 1,
                interactive: false
            }).addTo(map);
            elevationHoverMarker.bindTooltip("", {
                direction: "top",
                offset: [0, -6],
                className: "elevation-hover-tooltip"
            });
        } else {
            elevationHoverMarker.setLatLng(ll);
        }
        elevationHoverMarker.setTooltipContent(
            `${Number(p.distance_miles).toFixed(2)} mi · ${Math.round(Number(p.elevation_ft))} ft`
        );
        elevationHoverMarker.openTooltip();
        elevationHoverMarker.bringToFront();
    }
}


function handleSelectedRouteMapHover(latlng) {
    const selected = getSelectedRouteOption();
    if (!selected) return;
    const profile = selected.elevation_profile || [];
    const index = nearestProfileIndexToLatLng(profile, latlng);
    if (index >= 0) updateElevationHover(index, true);
}


function renderElevationProfile(option) {
    const svg = document.getElementById("elevationProfileSvg");
    const title = document.getElementById("elevation-profile-title");
    if (!svg || !title) return;

    clearElevationHover();
    const profile = (option && option.elevation_profile) || [];
    if (!profile.length) {
        title.textContent = "Elevation profile · no profile available";
        svg.innerHTML = "";
        elevationRenderState = null;
        return;
    }

    title.textContent = `${option.name} elevation profile · ${option.actual_distance_miles} mi · ${option.actual_gain_ft} ft gain · hover profile or route`;

    const width = Math.max(320, svg.clientWidth || 900);
    const height = Math.max(100, svg.clientHeight || 150);
    const margin = {left: 48, right: 16, top: 8, bottom: 24};
    const plotW = Math.max(1, width - margin.left - margin.right);
    const plotH = Math.max(1, height - margin.top - margin.bottom);

    const distances = profile.map(p => Number(p.distance_miles));
    const elevations = profile.map(p => Number(p.elevation_ft));
    const maxDistance = Math.max(...distances, 0.001);
    let minElevation = Math.min(...elevations);
    let maxElevation = Math.max(...elevations);
    if (maxElevation - minElevation < 10) {
        minElevation -= 5;
        maxElevation += 5;
    }
    const elevationRange = Math.max(1, maxElevation - minElevation);

    const x = d => margin.left + (d / maxDistance) * plotW;
    const y = e => margin.top + (1 - (e - minElevation) / elevationRange) * plotH;

    const pathData = profile.map((p, i) => {
        const command = i === 0 ? "M" : "L";
        return `${command}${x(Number(p.distance_miles)).toFixed(1)},${y(Number(p.elevation_ft)).toFixed(1)}`;
    }).join(" ");

    const baseY = margin.top + plotH;
    const firstX = x(Number(profile[0].distance_miles));
    const lastX = x(Number(profile[profile.length - 1].distance_miles));
    const areaPath = `${pathData} L${lastX.toFixed(1)},${baseY.toFixed(1)} L${firstX.toFixed(1)},${baseY.toFixed(1)} Z`;

    const midDistance = maxDistance / 2;
    const midElevation = (minElevation + maxElevation) / 2;

    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.innerHTML = `
        <line x1="${margin.left}" y1="${baseY}" x2="${width - margin.right}" y2="${baseY}" stroke="#aaa" stroke-width="1"/>
        <line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${baseY}" stroke="#aaa" stroke-width="1"/>
        <line x1="${margin.left}" y1="${y(midElevation)}" x2="${width - margin.right}" y2="${y(midElevation)}" stroke="#e5e5e5" stroke-width="1"/>
        <path d="${areaPath}" fill="#dbeafe" opacity="0.75"></path>
        <path d="${pathData}" fill="none" stroke="#b91c1c" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"></path>
        <line id="profileHoverLine" visibility="hidden" stroke="#111827" stroke-width="1.3" stroke-dasharray="4 3"></line>
        <circle id="profileHoverDot" visibility="hidden" r="4" fill="#111827" stroke="#ffffff" stroke-width="1.5"></circle>
        <text id="profileHoverLabel" visibility="hidden" class="elevation-axis-label" font-weight="bold"></text>
        <rect id="profileHoverCapture" x="${margin.left}" y="${margin.top}" width="${plotW}" height="${plotH}" fill="transparent" style="cursor:crosshair"></rect>
        <text x="${margin.left}" y="${height - 5}" class="elevation-axis-label">0 mi</text>
        <text x="${x(midDistance)}" y="${height - 5}" text-anchor="middle" class="elevation-axis-label">${midDistance.toFixed(1)} mi</text>
        <text x="${width - margin.right}" y="${height - 5}" text-anchor="end" class="elevation-axis-label">${maxDistance.toFixed(1)} mi</text>
        <text x="${margin.left - 6}" y="${margin.top + 9}" text-anchor="end" class="elevation-axis-label">${Math.round(maxElevation)} ft</text>
        <text x="${margin.left - 6}" y="${y(midElevation) + 4}" text-anchor="end" class="elevation-axis-label">${Math.round(midElevation)} ft</text>
        <text x="${margin.left - 6}" y="${baseY}" text-anchor="end" class="elevation-axis-label">${Math.round(minElevation)} ft</text>
    `;

    elevationRenderState = {profile, x, y, margin, plotW, plotH, width, height, maxDistance};
    const capture = document.getElementById("profileHoverCapture");
    if (capture) {
        capture.addEventListener("mousemove", event => {
            const rect = svg.getBoundingClientRect();
            const viewX = ((event.clientX - rect.left) / Math.max(rect.width, 1)) * width;
            const targetDistance = Math.max(0, Math.min(maxDistance, ((viewX - margin.left) / plotW) * maxDistance));
            let bestIndex = 0;
            let bestError = Infinity;
            for (let i = 0; i < profile.length; i++) {
                const error = Math.abs(Number(profile[i].distance_miles) - targetDistance);
                if (error < bestError) {
                    bestError = error;
                    bestIndex = i;
                }
            }
            updateElevationHover(bestIndex, true);
        });
        capture.addEventListener("mouseleave", clearElevationHover);
    }
}


function renderSelectedRouteDetails(option) {
    const details = document.getElementById("selectedRouteDetails");
    if (!details || !option) return;

    details.innerHTML =
        "<b>Selected:</b> " + option.name + "<br>" +
        "<b>Actual distance:</b> " + option.actual_distance_miles + " mi " +
        "(error " + option.distance_error_miles + " mi)<br>" +
        "<b>Elevation gain:</b> " + option.actual_gain_ft + " ft " +
        "(error " + option.elevation_error_ft + " ft)<br>" +
        "<b>Trail:</b> " + option.trail_percent + "% · " +
        "<b>Retrace:</b> " + (option.retrace_percent ?? 0) + "% · " +
        "<b>Connector:</b> " + (option.connector_percent ?? 0) + "%<br>" +
        "<b>Max reach:</b> " + (option.max_reach_miles ?? 0) + " mi · " +
        "<b>Preferred segments hit:</b> " + (option.preferred_hit_count ?? 0) + "<br>" +
        "<b>Score:</b> " + option.route_score +
        ((option.required_pass_points || []).length > 0
            ? "<br><b>Required points:</b> " + option.required_pass_points.map((point, index) =>
                `P${index + 1}: ${point.nearest_route_distance_m} m / ${point.tolerance_m} m allowed`
              ).join(" · ")
            : "");
}


function selectRouteOption(index, fitMap = true) {
    if (!lastGeneratedRoute || !lastGeneratedRoute.route_options) return;
    if (routeReplacementMode && replacementBaseRouteIndex !== null && index !== replacementBaseRouteIndex) {
        cancelRouteSectionReplacement(false);
    }
    const options = lastGeneratedRoute.route_options;
    if (index < 0 || index >= options.length) return;
    if (!routePassesQualityFilters(options[index])) return;

    selectedRouteOptionIndex = index;
    clearElevationHover();

    routeOptionLines.forEach((line, lineIndex) => {
        if (!line) return;
        if (lineIndex === index) {
            line.setStyle({weight: 7, opacity: 0.96, color: "#d60000"});
            line.bringToFront();
            routeLine = line;
        } else {
            line.setStyle({weight: 4, opacity: 0.24, color: "#4455aa"});
        }
    });

    document.querySelectorAll(".route-choice").forEach(button => {
        button.classList.toggle("selected", Number(button.dataset.routeIndex) === index);
    });

    const selected = options[index];
    renderSelectedRouteDetails(selected);
    renderElevationProfile(selected);
    refresh3DMapData();
    downloadGpxButton.disabled = false;
    editRouteButton.disabled = false;
    replaceSectionButton.disabled = false;

    if (fitMap && routeLine) {
        map.fitBounds(routeLine.getBounds(), {padding: [30, 30]});
    }
}


function drawRouteOptions(result, selectedIndex = 0, fitMap = true) {
    clearGeneratedRouteLines();
    const options = result.route_options || [];
    routeOptionLines = new Array(options.length).fill(null);

    options.forEach((option, index) => {
        if (!routePassesQualityFilters(option)) return;
        const coordinates = option.route.map(point => [point.lat, point.lon]);
        const line = L.polyline(coordinates, {
            weight: 4,
            opacity: 0.24,
            color: "#4455aa",
            interactive: true
        }).addTo(map);
        line.on("click", event => {
            // Route choice changes are intentionally list-only. Clicking a route
            // on the map must never switch the selected route. Map clicks on the
            // currently selected route are still available to explicit editing
            // tools such as section replacement / legacy edit mode.
            L.DomEvent.stopPropagation(event);
            if (routeReplacementMode && index === selectedRouteOptionIndex) {
                handleRouteReplacementRouteClick(event.latlng, index);
                return;
            }
            if (routeEditMode && index === selectedRouteOptionIndex) {
                addRouteEditPointFromRoute(event.latlng);
                return;
            }
            // Do nothing. Select a different route only from the route list.
        });
        line.on("mousemove", event => {
            if (index === selectedRouteOptionIndex) {
                handleSelectedRouteMapHover(event.latlng);
            }
        });
        line.on("mouseout", () => {
            if (index === selectedRouteOptionIndex) clearElevationHover();
        });
        routeOptionLines[index] = line;
    });

    const visible = getVisibleRouteIndices();
    if (visible.length > 0) {
        const requestedIndex = visible.includes(selectedIndex) ? selectedIndex : visible[0];
        selectRouteOption(requestedIndex, fitMap);
    } else {
        selectedRouteOptionIndex = -1;
        routeLine = null;
        downloadGpxButton.disabled = true;
        editRouteButton.disabled = true;
        replaceSectionButton.disabled = true;
        renderElevationProfile(null);
        const details = document.getElementById("selectedRouteDetails");
        if (details) details.innerHTML = '<span class="warning">No route choices meet the active quality filters.</span>';
    }
    drawRouteEditLayers();
    drawRouteReplacementLayers();
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
        target_gain_ft: parseFloat(document.getElementById("gain").value),
        pass_points: passPoints.map(point => ({
            lat: Number(point.lat),
            lon: Number(point.lon),
            tolerance_miles: Number(point.tolerance_miles)
        })),
        route_edit_points: routeEditPoints.map(point => ({
            lat: Number(point.lat),
            lon: Number(point.lon),
            tolerance_miles: Number(point.tolerance_miles || 0.03)
        })),
        avoid_areas: avoidAreas.map(area => ({
            lat: Number(area.lat),
            lon: Number(area.lon),
            radius_miles: Number(area.radius_miles)
        })),
        avoid_segments: avoidSegments.map(item => ({
            lat: Number(item.lat),
            lon: Number(item.lon)
        })),
        prefer_segments: preferSegments.map(item => ({
            lat: Number(item.lat),
            lon: Number(item.lon)
        })),
        route_diversity: Number(routeDiversityInput.value),
        search_seed: currentSearchSeed
    };
}


async function readJsonResponse(response) {
    const text = await response.text();

    if (!text) {
        throw new Error(
            "Server connection closed before a response was returned. " +
            "This usually means the server worker restarted or ran out of memory."
        );
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


function currentTrailOverlayBounds() {
    const b = map.getBounds();
    return {
        west: b.getWest(),
        south: b.getSouth(),
        east: b.getEast(),
        north: b.getNorth()
    };
}


function trailOverlayBoundsKey(bounds) {
    return [bounds.west, bounds.south, bounds.east, bounds.north]
        .map(value => Number(value).toFixed(4))
        .join(",") + "@" + map.getZoom();
}


async function loadMasterTrailOverlayOnce(force = false) {
    // Kept under the old function name so the rest of the UI needs almost no
    // changes. V38 now loads only the current viewport, not the whole region.
    if (map.getZoom() < MIN_TRAIL_OVERLAY_ZOOM) {
        networkLayer.clearLayers();
        masterTrailSegments = [];
        masterTrailRecords = [];
        masterTrailOverlayLastKey = "";
        return null;
    }

    const bounds = currentTrailOverlayBounds();
    const key = trailOverlayBoundsKey(bounds);
    if (!force && key === masterTrailOverlayLastKey) {
        return null;
    }

    const requestId = ++masterTrailOverlayRequestId;
    const params = new URLSearchParams({
        west: String(bounds.west),
        south: String(bounds.south),
        east: String(bounds.east),
        north: String(bounds.north)
    });

    masterTrailOverlayPromise = (async () => {
        const response = await fetch("/trail-overlay?" + params.toString());
        const result = await readJsonResponse(response);

        // Ignore a stale response if the user panned/zoomed while it was loading.
        if (requestId !== masterTrailOverlayRequestId) {
            return result;
        }

        networkLayer.clearLayers();
        masterTrailSegments = [];
        masterTrailRecords = [];

        if (result.viewport_too_wide) {
            masterTrailOverlayLastKey = key;
            updateNetworkVisibility();
            return result;
        }

        const tileRows = result.overlay_tiles || [];
        const tileResults = await Promise.all(
            tileRows.map(async tile => {
                const tileResponse = await fetch(tile.url);
                return await readJsonResponse(tileResponse);
            })
        );

        if (requestId !== masterTrailOverlayRequestId) {
            return result;
        }

        for (const tileResult of tileResults) {
            const segments = tileResult.allowed_trails || [];
            const records = tileResult.trail_records || [];
            if (segments.length === 0) continue;

            masterTrailSegments.push(...segments);
            if (records.length) {
                masterTrailRecords.push(...records);
            } else {
                masterTrailRecords.push(...segments.map(segment => ({
                    tile_id: tileResult.tile_id || null,
                    u: null,
                    v: null,
                    key: null,
                    geometry: segment
                })));
            }

            L.polyline(
                segments,
                {
                    weight: 3,
                    opacity: 0.42,
                    color: "#666666",
                    interactive: false
                }
            ).addTo(networkLayer);
        }

        masterTrailOverlayLastKey = key;
        updateNetworkVisibility();
        refresh3DMapData();
        return result;
    })();

    try {
        return await masterTrailOverlayPromise;
    } finally {
        if (requestId === masterTrailOverlayRequestId) {
            masterTrailOverlayPromise = null;
        }
    }
}


function scheduleTrailOverlayRefresh(delayMs = 180) {
    if (masterTrailOverlayTimer) {
        clearTimeout(masterTrailOverlayTimer);
    }
    masterTrailOverlayTimer = setTimeout(() => {
        masterTrailOverlayTimer = null;
        loadMasterTrailOverlayOnce(true).catch(error => {
            console.warn("Trail overlay load failed:", error);
        });
    }, delayMs);
}


map.on("moveend zoomend", () => {
    scheduleTrailOverlayRefresh();
});


async function loadTrailNetwork(data) {
    const startKey = getWorkspaceStartKey(data);

    // Refresh the current viewport overlay in parallel. It is visualization only
    // and must never block routing-area preparation or route search.
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

    createRequestedStartMarker(
        Number(result.requested_start.lat),
        Number(result.requested_start.lon),
        "Requested start · drag to adjust",
        false
    );

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


    return result;
}


async function reloadNetwork() {
    const data = getInputData();
    const results = document.getElementById("results");
    const startKey = getWorkspaceStartKey(data);

    networkButton.disabled = true;

    if (loadedWorkspaceStartKey === startKey && lastWorkspaceResult) {
        results.innerHTML = '<span class="success">Routing area is already ready for this start.</span>';
        networkButton.disabled = false;
        return;
    }

    results.innerHTML = '<span class="warning">Preparing routing area for this start...</span>';

    try {
        await loadTrailNetwork(data);
        results.innerHTML = '<span class="success">Routing area ready.</span>';
    } catch (error) {
        results.innerHTML = '<span class="error"><b>Error:</b> ' + error.message + '</span>';
    } finally {
        networkButton.disabled = false;
    }
}


function ensureRouteOptions(result) {
    const options = result.route_options || [];
    if (options.length === 0 && result.route && result.route.length >= 2) {
        const distance = Number(result.actual_distance_miles || 0);
        const repeatMiles = Number(result.repeated_distance_miles || 0);
        const connectorMiles = Number(result.connector_distance_miles || 0);
        result.route_options = [{
            index: 0,
            name: "Route 1",
            route_signature: null,
            actual_distance_miles: distance,
            distance_error_miles: result.distance_error_miles,
            actual_gain_ft: result.actual_gain_ft,
            actual_descent_ft: result.actual_descent_ft,
            elevation_error_ft: result.elevation_error_ft,
            route: result.route,
            gpx_export_points: result.gpx_export_points,
            elevation_profile: result.elevation_profile || [],
            repeated_edges: result.repeated_edges,
            repeated_distance_miles: repeatMiles,
            retrace_percent: distance > 0 ? Number((repeatMiles / distance * 100).toFixed(1)) : 0,
            repeated_nodes: result.repeated_nodes,
            immediate_reversals: result.immediate_reversals,
            connector_distance_miles: connectorMiles,
            connector_percent: distance > 0 ? Number((connectorMiles / distance * 100).toFixed(1)) : 0,
            trail_percent: result.trail_percent,
            independent_loops: result.independent_loops,
            extra_subloops: result.extra_subloops,
            branch_points: result.branch_points,
            max_reach_miles: result.max_reach_miles,
            footprint_sq_miles: result.footprint_sq_miles,
            shape_penalty: result.shape_penalty,
            route_score: result.route_score,
            preferred_hit_count: 0,
            preferred_distance_miles: 0,
            partial_edge_used: result.partial_edge_used,
            partial_added_distance_miles: result.partial_added_distance_miles,
            partial_outward_distance_meters: result.partial_outward_distance_meters,
            required_pass_points: result.required_pass_points || []
        }];
    }
    sortRouteOptionsByDistance(result.route_options || []);
    reindexRouteOptions(result.route_options || []);
    return result.route_options || [];
}


function sortRouteOptionsByDistance(options) {
    options.sort((a, b) => {
        const da = Number(a.actual_distance_miles || 0);
        const db = Number(b.actual_distance_miles || 0);
        if (da !== db) return da - db;

        const ga = Number(a.actual_gain_ft || 0);
        const gb = Number(b.actual_gain_ft || 0);
        if (ga !== gb) return ga - gb;

        return Number(a.route_score || 0) - Number(b.route_score || 0);
    });
    return options;
}


function reindexRouteOptions(options) {
    options.forEach((option, index) => {
        option.index = index;
        option.name = `Route ${index + 1}`;
        if (option.retrace_percent === undefined || option.retrace_percent === null) {
            const d = Number(option.actual_distance_miles || 0);
            option.retrace_percent = d > 0 ? Number((Number(option.repeated_distance_miles || 0) / d * 100).toFixed(1)) : 0;
        }
        if (option.connector_percent === undefined || option.connector_percent === null) {
            const d = Number(option.actual_distance_miles || 0);
            option.connector_percent = d > 0 ? Number((Number(option.connector_distance_miles || 0) / d * 100).toFixed(1)) : 0;
        }
    });
}


function browserRouteSignature(option) {
    if (option.route_signature) return String(option.route_signature);
    const route = option.route || [];
    return route.map(point => `${Number(point.lat).toFixed(5)},${Number(point.lon).toFixed(5)}`).join("|");
}


function searchConfigKey(data) {
    const clone = {...data};
    delete clone.search_seed;
    return JSON.stringify(clone);
}


function routeChoiceCardHtml(option, index) {
    return `
        <button type="button" class="route-choice" data-route-index="${index}">
            <span class="route-card-title-line">
                <span class="route-card-title">${option.name}</span>
                ${option.is_edited ? '<span class="edited-badge">edited</span>' : ''}
            </span>
            <span class="route-card-primary">${option.actual_distance_miles} mi · ${option.actual_gain_ft} ft gain</span>
            <span class="route-card-secondary">Retrace ${option.retrace_percent ?? 0}% · Connector ${option.connector_percent ?? 0}% · Reach ${option.max_reach_miles ?? 0} mi · Trail ${option.trail_percent ?? 0}%</span>
        </button>
    `;
}


function renderRouteResults(statusMessage = "Route search complete") {
    if (!lastGeneratedRoute) return;
    const results = document.getElementById("results");

    const selectedSignatureBeforeSort =
        lastGeneratedRoute.route_options?.[selectedRouteOptionIndex]
            ? browserRouteSignature(lastGeneratedRoute.route_options[selectedRouteOptionIndex])
            : null;

    sortRouteOptionsByDistance(lastGeneratedRoute.route_options || []);
    reindexRouteOptions(lastGeneratedRoute.route_options || []);

    if (selectedSignatureBeforeSort) {
        const movedIndex = (lastGeneratedRoute.route_options || []).findIndex(
            option => browserRouteSignature(option) === selectedSignatureBeforeSort
        );
        if (movedIndex >= 0) selectedRouteOptionIndex = movedIndex;
    }

    const options = lastGeneratedRoute.route_options || [];
    const visibleEntries = options
        .map((option, index) => ({option, index}))
        .filter(entry => routePassesQualityFilters(entry.option));
    const routeButtons = visibleEntries.map(entry => routeChoiceCardHtml(entry.option, entry.index)).join("");
    const filters = getQualityFilterConfig();
    const filterLine = filters.enabled
        ? `<span class="filter-badge">${visibleEntries.length}/${options.length} pass filters</span>`
        : "";

    results.innerHTML =
        `<span class="success"><b>${statusMessage}</b></span> ${filterLine}<br>` +
        `<b>Route choices:</b> ${visibleEntries.length}${filters.enabled ? ` shown · ${options.length} found` : ""}<br>` +
        (visibleEntries.length
            ? `<div class="route-choice-grid">${routeButtons}</div><div id="selectedRouteDetails"></div><br>`
            : '<div class="warning" style="margin:8px 0;">No routes meet the active quality filters. Relax the limits or turn filters off; the found routes are still kept.</div><div id="selectedRouteDetails"></div><br>') +
        `<b>Target:</b> ${lastGeneratedRoute.requested_distance_miles} mi · ${lastGeneratedRoute.requested_gain_ft} ft<br>` +
        `<b>Diversity:</b> ${Math.round(Number(lastGeneratedRoute.route_diversity ?? routeDiversityInput.value))}/100<br>` +
        (lastGeneratedRoute.required_pass_points_count
            ? `<b>Required routing points:</b> ${lastGeneratedRoute.required_pass_points_count}<br>`
            : "") +
        (routeEditPoints.length ? `<b>Direct edit points:</b> ${routeEditPoints.length}<br>` : "") +
        (avoidSegments.length ? `<b>Avoided trail segments:</b> ${avoidSegments.length}<br>` : "") +
        (preferSegments.length ? `<b>Preferred trail segments:</b> ${preferSegments.length}<br>` : "");

    document.querySelectorAll(".route-choice").forEach(button => {
        button.addEventListener("click", () => {
            selectRouteOption(Number(button.dataset.routeIndex), true);
        });
    });

    const selected = getSelectedRouteOption();
    if (selected && routePassesQualityFilters(selected)) {
        renderSelectedRouteDetails(selected);
        document.querySelectorAll(".route-choice").forEach(button => {
            button.classList.toggle("selected", Number(button.dataset.routeIndex) === selectedRouteOptionIndex);
        });
    }
    const filterStatus = document.getElementById("qualityFilterStatus");
    if (filterStatus) {
        filterStatus.textContent = filters.enabled
            ? `${visibleEntries.length} of ${options.length} route choices meet these limits.`
            : "";
    }
}


async function requestRouteBatch(data) {
    const response = await fetch(
        "/generate-route",
        {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(data)
        }
    );
    return await readJsonResponse(response);
}


async function generateRoute() {
    const results = document.getElementById("results");
    cancelRouteSectionReplacement(false);
    currentSearchSeed = Math.floor((Date.now() + Math.random() * 1000000) % 2147483647);
    findMoreBatch = 0;
    const data = getInputData();
    data.search_seed = currentSearchSeed;

    results.innerHTML = '<span class="warning">Loading allowed trails...</span>';
    generateButton.disabled = true;
    findMoreButton.disabled = true;
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
        const result = await requestRouteBatch(data);
        ensureRouteOptions(result);

        if (!result.route || result.route.length < 2) {
            throw new Error("Server returned an empty route.");
        }

        lastGeneratedRoute = result;
        lastSearchConfigKey = searchConfigKey(data);
        selectedRouteOptionIndex = 0;
        renderRouteResults("Route search complete");
        drawRouteOptions(lastGeneratedRoute, 0, true);
        findMoreButton.disabled = false;

    } catch (error) {
        results.innerHTML = '<span class="error"><b>Error:</b> ' + error.message + '</span>';
    } finally {
        generateButton.disabled = false;
    }
}


async function findMoreRoutes() {
    const results = document.getElementById("results");
    if (!lastGeneratedRoute || !(lastGeneratedRoute.route_options || []).length) return;

    const data = getInputData();
    const configKey = searchConfigKey(data);
    if (lastSearchConfigKey && configKey !== lastSearchConfigKey) {
        results.insertAdjacentHTML("afterbegin", '<span class="warning"><b>Settings changed.</b> Click Generate Trail Route first, then Find More.</span><br>');
        return;
    }

    findMoreButton.disabled = true;
    generateButton.disabled = true;
    findMoreBatch += 1;
    currentSearchSeed = Math.floor((Date.now() + findMoreBatch * 104729 + Math.random() * 1000000) % 2147483647);
    data.search_seed = currentSearchSeed;
    const priorSelected = selectedRouteOptionIndex;
    const priorCount = lastGeneratedRoute.route_options.length;

    try {
        results.insertAdjacentHTML("afterbegin", '<span id="findMoreStatus" class="warning">Searching another route batch...</span><br>');
        const result = await requestRouteBatch(data);
        const incoming = ensureRouteOptions(result);
        const existingSignatures = new Set(lastGeneratedRoute.route_options.map(browserRouteSignature));
        let added = 0;

        for (const option of incoming) {
            const signature = browserRouteSignature(option);
            if (existingSignatures.has(signature)) continue;
            existingSignatures.add(signature);
            lastGeneratedRoute.route_options.push(option);
            added += 1;
        }

        const selectedSignatureBeforeSort =
            lastGeneratedRoute.route_options?.[priorSelected]
                ? browserRouteSignature(lastGeneratedRoute.route_options[priorSelected])
                : null;

        sortRouteOptionsByDistance(lastGeneratedRoute.route_options);
        reindexRouteOptions(lastGeneratedRoute.route_options);
        lastGeneratedRoute.route_options_count = lastGeneratedRoute.route_options.length;

        let sortedSelectedIndex = 0;
        if (selectedSignatureBeforeSort) {
            const movedIndex = lastGeneratedRoute.route_options.findIndex(
                option => browserRouteSignature(option) === selectedSignatureBeforeSort
            );
            if (movedIndex >= 0) sortedSelectedIndex = movedIndex;
        }
        selectedRouteOptionIndex = sortedSelectedIndex;

        renderRouteResults(
            added > 0
                ? `Added ${added} new route${added === 1 ? "" : "s"} · ${lastGeneratedRoute.route_options.length} total`
                : `No new distinct routes in this batch · ${priorCount} kept`
        );
        drawRouteOptions(lastGeneratedRoute, sortedSelectedIndex, false);
    } catch (error) {
        const status = document.getElementById("findMoreStatus");
        if (status) status.outerHTML = '<span class="error"><b>Find More error:</b> ' + error.message + '</span>';
        else results.insertAdjacentHTML("afterbegin", '<span class="error"><b>Find More error:</b> ' + error.message + '</span><br>');
    } finally {
        findMoreButton.disabled = false;
        generateButton.disabled = false;
    }
}

// Load only gray trails visible in the initial map viewport. Do not spend
// time preparing the old default start: the first normal map click chooses the
// user's start point, then Generate/Load prepares that start workspace.
updateQualityFilterUi();
renderRouteEditRows();
loadPersistentBlocklist();
resetPlannerHistory();
loadMasterTrailOverlayOnce().catch(error => {
    console.warn("Trail overlay load failed:", error);
});
beginStartPointPlacement();
document.getElementById("results").innerHTML =
    '<span class="warning">Click the map to choose a starting point.</span>';
</script>

</body>
</html>
"""


# ============================================================
# ONE-TIME OFFLINE BUILD CLI
# ============================================================

if __name__ == "__main__":
    if "--build-master" in sys.argv or "--build-routing" in sys.argv or "--build-local" in sys.argv:
        try:
            # Legacy master-graph builder retained for development compatibility.
            # No Overpass/API calls are made. The command rebuilds both the
            # natural-trail master and final sparse routing graph in one pass.
            build_master_routing_graph(rebuild_trails=True)
        except Exception as exc:
            print(f"LOCAL ROUTING BUILD FAILED: {exc}", file=sys.stderr)
            raise SystemExit(1)
        raise SystemExit(0)

    print(
        "This file is the FastAPI app. Start it with uvicorn. To build the "
        "legacy monolithic offline routing graph, run: "
        "python main.py --build-routing"
    )
