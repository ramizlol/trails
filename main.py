from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import math
import os
import random
import time
import xml.etree.ElementTree as ET

import networkx as nx
import numpy as np
import osmnx as ox
import rasterio
from pyproj import Transformer
from rasterio.warp import transform as rio_transform


app = FastAPI()


# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEM_PATH = os.path.join(BASE_DIR, "output_USGS30m.tif")

METERS_PER_MILE = 1609.344
FEET_PER_METER = 3.28084

DEFAULT_LAT = 33.589281
DEFAULT_LON = -112.091148
APP_VERSION = "2026-08-09-search-budget-v2"

# Sample along trail/GPX geometry every 5 m.
# The source DEM is still ~30 m resolution.
ELEVATION_SAMPLE_SPACING_M = 5.0

# Smooth the 5 m samples across ~55 m before accumulating ascent/descent.
# This reduces staircase/noise inflation from repeatedly sampling a ~30 m DEM.
ELEVATION_SMOOTHING_POINTS = 11

# GPX points within this distance of an allowed trail count as covered.
GPX_TRAIL_MATCH_TOLERANCE_M = 25.0

MAX_CACHED_GRAPHS = 5
GRAPH_CACHE = {}


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

            # Keep the short-route search manageable on Render.
            "beam_width": 500,
            "beam_max_steps": 80,

            # Hard computation budgets.
            "max_search_seconds": 18.0,
            "max_expanded_states": 100000,

            # Only run the expensive whole-route elevation calculation
            # on the best closed-loop candidates.
            "max_closed_candidates": 60,
        }

    if target_distance_miles < 8.0:
        return {
            "name": "medium-waypoint",
            "search_radius_m": 2500,
            "attempts": 1200,
            "anchor_counts": [2, 3, 3, 3],
            "min_anchor_distance_m": 150,
            "min_anchor_separation_m": 140,
        }

    if target_distance_miles < 15.0:
        return {
            "name": "long-waypoint",
            "search_radius_m": 3200,
            "attempts": 900,
            "anchor_counts": [3, 4, 4, 4],
            "min_anchor_distance_m": 300,
            "min_anchor_separation_m": 250,
        }

    return {
        "name": "ultra-waypoint",
        "search_radius_m": min(5000, int(target_distance_miles * 300)),
        "attempts": 700,
        "anchor_counts": [4, 4, 5],
        "min_anchor_distance_m": 400,
        "min_anchor_separation_m": 300,
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

def smooth_elevations(values, window_points=ELEVATION_SMOOTHING_POINTS):
    """
    Centered moving average used for both generated routes and GPX analysis.

    With 5 m samples and an 11-point window, the effective smoothing width is
    about 55 m. This is intentionally wider than one 30 m DEM cell so small
    raster stair-steps are less likely to be counted as real climbing.
    """
    if len(values) < 3:
        return [float(v) for v in values]

    window_points = max(1, int(window_points))

    if window_points % 2 == 0:
        window_points += 1

    radius = window_points // 2
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

    keys = list(unique.keys())
    lons = [unique[key][0] for key in keys]
    lats = [unique[key][1] for key in keys]

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

        lookup = {}

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

            lookup[key] = dem_value_to_meters(src, value)

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


def route_raw_elevation_samples(G, route_nodes):
    """
    Concatenate the raw 5 m DEM samples from every traversed edge into one
    continuous route profile. This is deliberately done BEFORE smoothing so
    edge boundaries cannot create artificial ascent/descent resets.
    """
    route_coords = []
    route_elevations = []

    for i in range(len(route_nodes) - 1):
        u = route_nodes[i]
        v = route_nodes[i + 1]
        edge = get_shortest_edge(G, u, v)

        if edge is None:
            continue

        edge_coords = edge.get("dem_sample_coords")
        edge_elevations = edge.get("dem_raw_elevations_m")

        # Fallback for any graph created before these cached fields existed.
        if not edge_coords or not edge_elevations:
            edge_coords = densify_polyline(
                oriented_edge_coords(G, u, v, edge),
                ELEVATION_SAMPLE_SPACING_M,
            )
            edge_elevations = elevations_for_coords(edge_coords)

        edge_coords = list(edge_coords)
        edge_elevations = [float(value) for value in edge_elevations]

        if not edge_coords or not edge_elevations:
            continue

        # Adjacent directed edges normally share the junction sample.
        # Drop that one duplicate so it is not over-weighted by smoothing.
        start_index = 0

        if route_coords:
            last_lon, last_lat = route_coords[-1]
            first_lon, first_lat = edge_coords[0]

            if haversine_meters(
                last_lat,
                last_lon,
                first_lat,
                first_lon,
            ) < 0.5:
                start_index = 1

        route_coords.extend(edge_coords[start_index:])
        route_elevations.extend(edge_elevations[start_index:])

    return route_coords, route_elevations


def route_elevation_metrics(G, route_nodes):
    """
    Calculate final route ascent/descent from one continuous DEM profile.

    This is the authoritative elevation calculation for generated routes.
    Edge-level ascent is retained only as a cheap beam-search heuristic.
    """
    coords, raw_elevations = route_raw_elevation_samples(G, route_nodes)

    if len(raw_elevations) < 2:
        return {
            "ascent_m": 0.0,
            "descent_m": 0.0,
            "sample_count": len(raw_elevations),
            "raw_elevations_m": raw_elevations,
            "smoothed_elevations_m": raw_elevations,
            "coords": coords,
        }

    smoothed = smooth_elevations(
        raw_elevations,
        window_points=ELEVATION_SMOOTHING_POINTS,
    )

    ascent_m, descent_m = calculate_ascent_descent(smoothed)

    return {
        "ascent_m": float(ascent_m),
        "descent_m": float(descent_m),
        "sample_count": len(raw_elevations),
        "raw_elevations_m": raw_elevations,
        "smoothed_elevations_m": smoothed,
        "coords": coords,
    }


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

        # Keep the raw samples on each directed edge. Generated routes later
        # concatenate these into one continuous profile and smooth only once.
        G[u][v][key]["dem_sample_coords"] = list(samples)
        G[u][v][key]["dem_raw_elevations_m"] = list(elevations)

        # Edge ascent is only a beam-search heuristic. The final route gain is
        # always recalculated across the whole continuous route profile.
        heuristic_elevations = smooth_elevations(
            elevations,
            window_points=ELEVATION_SMOOTHING_POINTS,
        )
        ascent, descent = calculate_ascent_descent(heuristic_elevations)

        G[u][v][key]["ascent_m"] = float(ascent)
        G[u][v][key]["descent_m"] = float(descent)
        G[u][v][key]["elevation_sample_count"] = len(samples)

    return G, len(elevation_lookup)


# ============================================================
# DOWNLOAD / CACHE TRAIL GRAPH
# ============================================================

def download_trail_graph(lat, lon, radius_meters):
    cache_key = (
        round(float(lat), 5),
        round(float(lon), 5),
        int(radius_meters),
        ELEVATION_SAMPLE_SPACING_M,
        ELEVATION_SMOOTHING_POINTS,
    )

    if cache_key in GRAPH_CACHE:
        cached = GRAPH_CACHE[cache_key]
        return (
            cached["graph"],
            cached["filtered_edges_removed"],
            True,
            cached["unique_elevation_samples"],
        )

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

    trail_filter = '["highway"~"path|track|steps"]'

    G = ox.graph.graph_from_point(
        (lat, lon),
        dist=radius_meters,
        network_type="walk",
        custom_filter=trail_filter,
        simplify=True,
        retain_all=True,
    )

    original_edges = G.number_of_edges()

    remove_edges = []

    for u, v, key, data in G.edges(keys=True, data=True):
        if not edge_is_allowed_trail(data):
            remove_edges.append((u, v, key))

    G.remove_edges_from(remove_edges)
    G.remove_nodes_from(list(nx.isolates(G)))

    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        raise HTTPException(
            status_code=400,
            detail="No usable trail network found.",
        )

    nearest = ox.distance.nearest_nodes(
        G,
        X=lon,
        Y=lat,
    )

    component = nx.node_connected_component(
        G.to_undirected(as_view=True),
        nearest,
    )

    G = G.subgraph(component).copy()

    filtered_edges_removed = original_edges - G.number_of_edges()

    G, unique_elevation_samples = add_local_dem_edge_elevations(G)

    if len(GRAPH_CACHE) >= MAX_CACHED_GRAPHS:
        oldest = next(iter(GRAPH_CACHE))
        GRAPH_CACHE.pop(oldest)

    GRAPH_CACHE[cache_key] = {
        "graph": G,
        "filtered_edges_removed": filtered_edges_removed,
        "unique_elevation_samples": unique_elevation_samples,
    }

    return (
        G,
        filtered_edges_removed,
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
# ROUTE SCORE
# ============================================================

def route_score(G, route_nodes, target_distance_meters, target_gain_meters):
    total_distance = path_distance_meters(G, route_nodes)

    # Authoritative final elevation calculation: concatenate the entire route's
    # raw 5 m DEM samples, smooth once across the full route, then accumulate.
    elevation_metrics = route_elevation_metrics(G, route_nodes)
    actual_gain = elevation_metrics["ascent_m"]
    actual_descent = elevation_metrics["descent_m"]

    if total_distance <= 0:
        return float("inf"), {}

    distance_error = abs(total_distance - target_distance_meters)
    distance_ratio = distance_error / target_distance_meters

    gain_error = abs(actual_gain - target_gain_meters)

    if target_gain_meters > 0:
        gain_ratio = gain_error / target_gain_meters
    else:
        gain_ratio = actual_gain / 30.48

    repeated_edges, repeated_distance = repeated_edge_stats(G, route_nodes)
    repeat_ratio = repeated_distance / total_distance
    repeated_nodes = repeated_node_occurrences(route_nodes)
    immediate_reversals = count_immediate_reversals(route_nodes)

    if target_distance_meters < 4 * METERS_PER_MILE:
        repeat_weight = 70.0
        node_weight = 8.0
    else:
        repeat_weight = 300.0
        node_weight = 25.0

    score = (
        distance_ratio * 190.0
        + gain_ratio * 240.0
        + repeat_ratio * repeat_weight
        + repeated_nodes * node_weight
        + immediate_reversals * 15.0
    )

    return (
        score,
        {
            "total_distance_meters": total_distance,
            "actual_gain_meters": actual_gain,
            "actual_descent_meters": actual_descent,
            "route_elevation_sample_count": elevation_metrics["sample_count"],
            "distance_error_meters": distance_error,
            "gain_error_meters": gain_error,
            "repeated_edges": repeated_edges,
            "repeated_distance_meters": repeated_distance,
            "repeat_ratio": repeat_ratio,
            "repeated_nodes": repeated_nodes,
            "immediate_reversals": immediate_reversals,
            "score": score,
        },
    )


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
    """
    Time/state-budgeted true closed-loop beam search.

    During exploration:
      - trail distance is exact
      - edge ascent is only a cheap heuristic
      - there is no hard elevation prune
      - whole-route 5 m DEM elevation is NOT recalculated for every loop

    Once the search budget ends:
      - retain the best closed-loop candidates
      - accurately score only those finalists using the same continuous
        5 m DEM profile + smoothing pipeline used by GPX analysis
    """

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

    allowed_distance_error_m = (
        limits["distance_error_limit_miles"] * METERS_PER_MILE
    )

    max_acceptable_distance = (
        target_distance_meters + allowed_distance_error_m
    )

    beam_width = int(profile.get("beam_width", 500))
    max_steps = int(profile.get("beam_max_steps", 80))
    max_search_seconds = float(profile.get("max_search_seconds", 18.0))
    max_expanded_states = int(profile.get("max_expanded_states", 100000))
    max_closed_candidates = int(profile.get("max_closed_candidates", 60))

    target_gain_density = (
        target_gain_meters / target_distance_meters
        if target_distance_meters > 0
        else 0.0
    )

    start_time = time.perf_counter()
    states_expanded = 0
    last_depth = 0
    budget_reached = False

    beam = [
        {
            "route": (start_node,),
            "node": start_node,
            "distance": 0.0,
            "gain": 0.0,
            "used_edges": frozenset(),
            "repeat_distance": 0.0,
            "reversals": 0,
        }
    ]

    # Closed loops are initially ranked with cheap metrics only.
    # Accurate whole-route elevation is deferred until the search ends.
    closed_candidates = []
    closed_routes_seen = set()

    for depth in range(max_steps):
        last_depth = depth + 1
        expanded = []

        if time.perf_counter() - start_time >= max_search_seconds:
            budget_reached = True
            break

        if states_expanded >= max_expanded_states:
            budget_reached = True
            break

        for state in beam:
            if time.perf_counter() - start_time >= max_search_seconds:
                budget_reached = True
                break

            if states_expanded >= max_expanded_states:
                budget_reached = True
                break

            current = state["node"]

            for neighbor in S.successors(current):
                states_expanded += 1

                if states_expanded >= max_expanded_states:
                    budget_reached = True
                    break

                if time.perf_counter() - start_time >= max_search_seconds:
                    budget_reached = True
                    break

                edge_data = S[current][neighbor]
                edge_length = float(edge_data.get("length", 0) or 0)
                edge_gain = float(edge_data.get("ascent_m", 0) or 0)

                if edge_length <= 0:
                    continue

                new_distance = state["distance"] + edge_length
                new_gain = state["gain"] + edge_gain

                # Distance is safe to hard-prune because it can never decrease.
                if new_distance > max_acceptable_distance:
                    continue

                edge_key = undirected_edge_key(current, neighbor)
                already_used = edge_key in state["used_edges"]

                repeat_distance = (
                    state["repeat_distance"]
                    + (edge_length if already_used else 0.0)
                )

                used_edges = set(state["used_edges"])
                used_edges.add(edge_key)

                route = state["route"]

                immediate_reversal = 0
                if len(route) >= 2 and neighbor == route[-2]:
                    immediate_reversal = 1

                reversals = state["reversals"] + immediate_reversal
                new_route = route + (neighbor,)

                # ---------------------------------------------------------
                # A real closed loop has physically returned to start_node.
                # ---------------------------------------------------------
                if neighbor == start_node:
                    # Ignore tiny accidental circles near the trailhead.
                    if new_distance < target_distance_meters * 0.50:
                        continue

                    route_key = tuple(new_route)
                    if route_key in closed_routes_seen:
                        continue

                    closed_routes_seen.add(route_key)

                    distance_error_ratio = (
                        abs(new_distance - target_distance_meters)
                        / max(target_distance_meters, 1.0)
                    )

                    if target_gain_meters > 0:
                        approximate_gain_error_ratio = (
                            abs(new_gain - target_gain_meters)
                            / target_gain_meters
                        )
                    else:
                        approximate_gain_error_ratio = new_gain / 30.48

                    repeat_ratio = (
                        repeat_distance / max(new_distance, 1.0)
                    )

                    # Cheap pre-score. Distance is trusted. Edge-level elevation
                    # is deliberately weak because final elevation is calculated
                    # from one continuous whole-route DEM profile later.
                    cheap_score = (
                        distance_error_ratio * 200.0
                        + approximate_gain_error_ratio * 25.0
                        + repeat_ratio * 35.0
                        + reversals * 5.0
                    )

                    closed_candidates.append(
                        (cheap_score, list(new_route))
                    )

                    # Keep candidate storage bounded during the search.
                    if (
                        len(closed_candidates)
                        > max_closed_candidates * 4
                    ):
                        closed_candidates.sort(key=lambda item: item[0])
                        closed_candidates = closed_candidates[
                            :max_closed_candidates
                        ]

                    # Do not continue through the trailhead into a second loop.
                    continue

                # ---------------------------------------------------------
                # Shortest distance home is only a lower-bound feasibility test.
                # The shortest path itself is NEVER appended to the route.
                # ---------------------------------------------------------
                if neighbor not in return_distance:
                    continue

                distance_home_lower_bound = float(return_distance[neighbor])
                minimum_possible_final_distance = (
                    new_distance + distance_home_lower_bound
                )

                if minimum_possible_final_distance > max_acceptable_distance:
                    continue

                # ---------------------------------------------------------
                # Cheap partial-state priority.
                # ---------------------------------------------------------
                projected_distance_error = (
                    abs(
                        minimum_possible_final_distance
                        - target_distance_meters
                    )
                    / max(target_distance_meters, 1.0)
                )

                if new_distance > 0:
                    current_gain_density = new_gain / new_distance
                else:
                    current_gain_density = 0.0

                gain_density_denominator = max(
                    target_gain_density,
                    0.004,
                )

                gain_density_error = (
                    abs(current_gain_density - target_gain_density)
                    / gain_density_denominator
                )

                repeat_ratio = (
                    repeat_distance / max(new_distance, 1.0)
                )

                node_revisited = neighbor in route
                node_revisit_penalty = 0.02 if node_revisited else 0.0

                # Elevation is intentionally a weak ranking heuristic and never
                # a hard prune. This keeps potentially good low-vert routes alive
                # even when edge-level DEM ascent is noisy.
                priority = (
                    projected_distance_error * 4.0
                    + gain_density_error * 0.20
                    + repeat_ratio * 0.20
                    + node_revisit_penalty
                    + reversals * 0.02
                )

                expanded.append(
                    (
                        priority,
                        {
                            "route": new_route,
                            "node": neighbor,
                            "distance": new_distance,
                            "gain": new_gain,
                            "used_edges": frozenset(used_edges),
                            "repeat_distance": repeat_distance,
                            "reversals": reversals,
                        },
                    )
                )

            if budget_reached:
                break

        if budget_reached:
            break

        if not expanded:
            break

        # -------------------------------------------------------------
        # Beam reduction / state deduplication.
        # Preserve multiple materially different ways to reach a node.
        # -------------------------------------------------------------
        expanded.sort(key=lambda item: item[0])

        next_beam = []
        seen_buckets = set()

        for priority, state in expanded:
            route = state["route"]
            previous_node = route[-2] if len(route) >= 2 else None

            distance_bucket = int(state["distance"] / 30.0)
            gain_bucket = int(state["gain"] / 10.0)
            repeat_bucket = int(state["repeat_distance"] / 30.0)

            bucket = (
                state["node"],
                previous_node,
                distance_bucket,
                gain_bucket,
                repeat_bucket,
            )

            if bucket in seen_buckets:
                continue

            seen_buckets.add(bucket)
            next_beam.append(state)

            if len(next_beam) >= beam_width:
                break

        beam = next_beam

    # =============================================================
    # ACCURATE FINAL SCORING
    # =============================================================
    # Run the expensive 5 m continuous-route DEM + ~55 m smoothing
    # only on the best closed-loop finalists.

    if closed_candidates:
        closed_candidates.sort(key=lambda item: item[0])
        closed_candidates = closed_candidates[:max_closed_candidates]

    best_acceptable_route = None
    best_acceptable_metrics = None
    best_acceptable_score = float("inf")

    best_closed_route = None
    best_closed_metrics = None
    best_closed_score = float("inf")

    for cheap_score, candidate_route in closed_candidates:
        score, metrics = route_score(
            G,
            candidate_route,
            target_distance_meters,
            target_gain_meters,
        )

        if score < best_closed_score:
            best_closed_score = score
            best_closed_route = candidate_route
            best_closed_metrics = metrics

        distance_error_miles = (
            metrics["distance_error_meters"] / METERS_PER_MILE
        )

        gain_error_ft = (
            metrics["gain_error_meters"] * FEET_PER_METER
        )

        acceptable = (
            distance_error_miles
            <= limits["distance_error_limit_miles"]
            and gain_error_ft
            <= limits["gain_error_limit_ft"]
        )

        if acceptable and score < best_acceptable_score:
            best_acceptable_score = score
            best_acceptable_route = candidate_route
            best_acceptable_metrics = metrics

    if best_acceptable_route is not None:
        return (
            best_acceptable_route,
            best_acceptable_metrics,
            last_depth,
            states_expanded,
        )

    if best_closed_route is not None:
        best_distance_miles = (
            best_closed_metrics["total_distance_meters"] / METERS_PER_MILE
        )

        best_gain_ft = (
            best_closed_metrics["actual_gain_meters"] * FEET_PER_METER
        )

        budget_text = (
            " Search budget was reached."
            if budget_reached
            else ""
        )

        raise HTTPException(
            status_code=400,
            detail=(
                "Closed loops were found, but none met the requested "
                "quality limits. Best accurately scored route was "
                f"{best_distance_miles:.2f} mi / "
                f"{round(best_gain_ft)} ft gain."
                + budget_text
            ),
        )

    budget_text = (
        " Search budget was reached."
        if budget_reached
        else ""
    )

    raise HTTPException(
        status_code=400,
        detail=(
            "No closed loop was found within the search budget."
            + budget_text
        ),
    )


# ============================================================
# LONGER-ROUTE WAYPOINT PATH SEARCH
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


def generate_waypoint_loop(
    G,
    start_node,
    target_distance_meters,
    target_gain_meters,
    profile,
    limits,
):
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

    best_route = None
    best_metrics = None
    best_score = float("inf")

    best_any_route = None
    best_any_metrics = None
    best_any_score = float("inf")

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

        score, metrics = route_score(
            G,
            route,
            target_distance_meters,
            target_gain_meters,
        )

        distance = metrics["total_distance_meters"]

        if (
            distance < target_distance_meters * 0.72
            or distance > target_distance_meters * 1.25
        ):
            continue

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
                f"Best candidate was {best_distance:.2f} mi / "
                f"{round(best_gain)} ft gain."
            ),
        )

    raise HTTPException(
        status_code=400,
        detail="No suitable waypoint route found.",
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
            window_points=ELEVATION_SMOOTHING_POINTS,
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
            "elevation_smoothing_window_points": ELEVATION_SMOOTHING_POINTS,
            "elevation_smoothing_window_m_approx": round(
                ELEVATION_SMOOTHING_POINTS * ELEVATION_SAMPLE_SPACING_M
            ),
            "elevation_source": os.path.basename(DEM_PATH),
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

        start_node = ox.distance.nearest_nodes(
            G,
            X=request.start_lon,
            Y=request.start_lat,
        )

        snapped_lat = float(G.nodes[start_node]["y"])
        snapped_lon = float(G.nodes[start_node]["x"])

        snap_distance = haversine_meters(
            request.start_lat,
            request.start_lon,
            snapped_lat,
            snapped_lon,
        )

        segments = graph_debug_segments(G)

        return {
            "allowed_trails": segments,
            "allowed_trail_segments": len(segments),
            "network_nodes": G.number_of_nodes(),
            "network_edges": G.number_of_edges(),
            "search_radius_m": profile["search_radius_m"],
            "route_profile": profile["name"],
            "requested_start": {
                "lat": request.start_lat,
                "lon": request.start_lon,
            },
            "snapped_start": {
                "lat": snapped_lat,
                "lon": snapped_lon,
            },
            "snap_distance_m": round(snap_distance, 1),
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
            "elevation_smoothing_window_points": ELEVATION_SMOOTHING_POINTS,
            "elevation_smoothing_window_m_approx": round(
                ELEVATION_SMOOTHING_POINTS * ELEVATION_SAMPLE_SPACING_M
            ),
        }


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

        start_node = ox.distance.nearest_nodes(
            G,
            X=request.start_lon,
            Y=request.start_lat,
        )

        end_node = ox.distance.nearest_nodes(
            G,
            X=request.end_lon,
            Y=request.end_lat,
        )

        snapped_start_lat = float(G.nodes[start_node]["y"])
        snapped_start_lon = float(G.nodes[start_node]["x"])

        snap_distance_m = haversine_meters(
            request.start_lat,
            request.start_lon,
            snapped_start_lat,
            snapped_start_lon,
        )

        same_point = (
            abs(request.start_lat - request.end_lat) < 0.0001
            and abs(request.start_lon - request.end_lon) < 0.0001
        )

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

        actual_descent_ft = (
            metrics.get("actual_descent_meters", 0.0) * FEET_PER_METER
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

        coords = route_coordinates(G, route_nodes)

        return {
            "requested_distance_miles": request.target_distance_miles,
            "actual_distance_miles": round(route_distance_miles, 2),
            "distance_error_miles": round(distance_error_miles, 2),
            "requested_gain_ft": request.target_gain_ft,
            "actual_gain_ft": round(actual_gain_ft),
            "actual_descent_ft": round(actual_descent_ft),
            "elevation_error_ft": round(elevation_error_ft),
            "route_type": route_type,
            "search_method": search_method,
            "route_profile": profile["name"],
            "route": coords,
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
            "search_radius_m": profile["search_radius_m"],
            "network_nodes": G.number_of_nodes(),
            "network_edges": G.number_of_edges(),
            "filtered_edges_removed": filtered_edges_removed,
            "graph_from_cache": graph_from_cache,
            "unique_elevation_samples": unique_elevation_samples,
            "route_elevation_samples": metrics.get(
                "route_elevation_sample_count",
                0,
            ),
            "elevation_sample_spacing_m": ELEVATION_SAMPLE_SPACING_M,
            "elevation_smoothing_window_points": ELEVATION_SMOOTHING_POINTS,
            "elevation_smoothing_window_m_approx": round(
                ELEVATION_SMOOTHING_POINTS * ELEVATION_SAMPLE_SPACING_M
            ),
            "elevation_source": os.path.basename(DEM_PATH),
            "snapped_start_lat": snapped_start_lat,
            "snapped_start_lon": snapped_start_lon,
            "snap_distance_m": round(snap_distance_m, 1),
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
<div class="small"><b>Version:</b> 2026-08-09-search-budget-v2</div>

<div class="input-row">
    <div class="input-group">
        <label for="start_lat">Start latitude</label>
        <input id="start_lat" type="number" step="any" value="33.589281">
    </div>

    <div class="input-group">
        <label for="start_lon">Start longitude</label>
        <input id="start_lon" type="number" step="any" value="-112.091148">
    </div>

    <div class="input-group">
        <label for="end_lat">End latitude</label>
        <input id="end_lat" type="number" step="any" value="33.589281">
    </div>

    <div class="input-group">
        <label for="end_lon">End longitude</label>
        <input id="end_lon" type="number" step="any" value="-112.091148">
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
    [33.589281, -112.091148],
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
let requestedStartMarker = null;
let snappedStartMarker = null;
let snapLine = null;

const generateButton = document.getElementById("generateButton");
const networkButton = document.getElementById("networkButton");
const analyzeGpxButton = document.getElementById("analyzeGpxButton");
const clearGpxButton = document.getElementById("clearGpxButton");
const showNetworkCheckbox = document.getElementById("showNetwork");


generateButton.addEventListener("click", generateRoute);
networkButton.addEventListener("click", reloadNetwork);
analyzeGpxButton.addEventListener("click", analyzeGpx);
clearGpxButton.addEventListener("click", clearGpx);
showNetworkCheckbox.addEventListener("change", updateNetworkVisibility);


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
        "<b>Graph nodes:</b> " +
        result.network_nodes +
        "<br>" +
        "<b>Graph edges:</b> " +
        result.network_edges +
        "<br>" +
        "<b>Start snap distance:</b> " +
        result.snap_distance_m +
        " m<br>" +
        "<b>Search radius:</b> " +
        result.search_radius_m +
        " m<br>" +
        "<b>Profile:</b> " +
        result.route_profile;

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
            "<b>Elevation error:</b> " + result.elevation_error_ft + " ft<br><br>" +
            "<b>Search method:</b> " + result.search_method + "<br>" +
            "<b>Route profile:</b> " + result.route_profile + "<br>" +
            "<b>Search depth:</b> " + result.search_steps + "<br>" +
            "<b>States expanded:</b> " + expandedText + "<br>" +
            "<b>Start snap distance:</b> " + result.snap_distance_m + " m<br><br>" +
            "<b>Repeated trail distance:</b> " + result.repeated_distance_miles + " mi<br>" +
            "<b>Repeated edges:</b> " + result.repeated_edges + "<br>" +
            "<b>Repeated junctions:</b> " + result.repeated_nodes + "<br>" +
            "<b>Immediate reversals:</b> " + result.immediate_reversals + "<br>" +
            "<b>Route score:</b> " + result.route_score + "<br><br>" +
            "<b>Graph cached:</b> " + result.graph_from_cache + "<br>" +
            "<b>Graph elevation samples:</b> " + result.unique_elevation_samples + "<br>" +
            "<b>Route elevation samples:</b> " + result.route_elevation_samples + "<br>" +
            "<b>Elevation sample spacing:</b> ~" + result.elevation_sample_spacing_m + " m<br>" +
            "<b>Elevation smoothing:</b> " + result.elevation_smoothing_window_points +
            " points (~" + result.elevation_smoothing_window_m_approx + " m)<br>" +
            '<span class="small">Elevation source: ' + result.elevation_source + "</span>";

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
            "<b>Elevation smoothing window:</b> " + result.elevation_smoothing_window_points +
            " points (~" + result.elevation_smoothing_window_m_approx + " m)<br>" +
            '<span class="small">Blue = manual GPX, red = generated route, gray = allowed trail network.</span>';

    } catch (error) {
        gpxResults.innerHTML = '<span class="error"><b>Error:</b> ' + error.message + "</span>";
    } finally {
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
