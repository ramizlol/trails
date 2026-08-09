from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import math
import os
import random

import networkx as nx
import numpy as np
import osmnx as ox
import rasterio
from rasterio.warp import transform as rio_transform


app = FastAPI()


# =========================================================
# SETTINGS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DEM_PATH = os.path.join(
    BASE_DIR,
    "output_USGS30m.tif"
)

METERS_PER_MILE = 1609.344
FEET_PER_METER = 3.28084

# Your requested trail sampling spacing.
#
# Note:
# The source DEM is still ~30 m resolution.
# Sampling every 5 m makes sure we inspect the
# terrain continuously along curved trails.
ELEVATION_SAMPLE_SPACING_M = 5.0

MAX_CACHED_GRAPHS = 5

GRAPH_CACHE = {}


# =========================================================
# DEFAULT TEST LOCATION
# =========================================================

DEFAULT_LAT = 33.589281
DEFAULT_LON = -112.091148


# =========================================================
# REQUEST
# =========================================================

class RouteRequest(BaseModel):

    start_lat: float
    start_lon: float

    end_lat: float
    end_lon: float

    target_distance_miles: float
    target_gain_ft: float


# =========================================================
# HOME
# =========================================================

@app.get("/")
def home():

    return {

        "status":
            "Trail Running Creator API is running",

        "map":
            "/map",

        "docs":
            "/docs",

        "dem_info":
            "/dem-info",

        "default_start": {

            "lat":
                DEFAULT_LAT,

            "lon":
                DEFAULT_LON
        }
    }


# =========================================================
# BASIC HELPERS
# =========================================================

def normalize_tag_values(value):

    if value is None:
        return set()


    if not isinstance(
        value,
        (list, tuple, set)
    ):

        value = [value]


    result = set()


    for item in value:

        if item is None:
            continue


        for part in str(item).split(";"):

            part = (
                part.strip().lower()
            )


            if part:

                result.add(
                    part
                )


    return result


def undirected_edge_key(
    u,
    v
):

    return tuple(
        sorted(
            (
                int(u),
                int(v)
            )
        )
    )


def haversine_meters(
    lat1,
    lon1,
    lat2,
    lon2
):

    radius = 6371000.0


    p1 = math.radians(
        lat1
    )

    p2 = math.radians(
        lat2
    )


    dphi = math.radians(
        lat2 - lat1
    )


    dlambda = math.radians(
        lon2 - lon1
    )


    a = (

        math.sin(
            dphi / 2.0
        ) ** 2

        +

        math.cos(p1)
        *
        math.cos(p2)
        *
        math.sin(
            dlambda / 2.0
        ) ** 2
    )


    return (

        2.0
        *
        radius
        *
        math.atan2(

            math.sqrt(a),

            math.sqrt(
                1.0 - a
            )
        )
    )


def node_angle_from_start(
    G,
    start_node,
    node
):

    start_lat = float(
        G.nodes[start_node]["y"]
    )

    start_lon = float(
        G.nodes[start_node]["x"]
    )


    lat = float(
        G.nodes[node]["y"]
    )

    lon = float(
        G.nodes[node]["x"]
    )


    x = (

        lon - start_lon

    ) * math.cos(

        math.radians(
            start_lat
        )
    )


    y = (
        lat - start_lat
    )


    return math.atan2(
        y,
        x
    )


# =========================================================
# ROUTE PROFILE
# =========================================================

def get_route_profile(
    target_distance_miles
):

    # Short routes use beam search.
    if target_distance_miles < 4.0:

        return {

            "name":
                "short-beam",

            "search_radius_m":
                1800,

            "beam_width":
                450,

            "beam_max_steps":
                80,

            "candidate_max_ratio":
                1.16
        }


    # Medium routes still use the waypoint system.
    if target_distance_miles < 8.0:

        return {

            "name":
                "medium-waypoint",

            "search_radius_m":
                2500,

            "attempts":
                1200,

            "anchor_counts":
                [2, 3, 3, 3],

            "min_anchor_distance_m":
                150,

            "min_anchor_separation_m":
                140
        }


    if target_distance_miles < 15.0:

        return {

            "name":
                "long-waypoint",

            "search_radius_m":
                3200,

            "attempts":
                900,

            "anchor_counts":
                [3, 4, 4, 4],

            "min_anchor_distance_m":
                300,

            "min_anchor_separation_m":
                250
        }


    return {

        "name":
            "ultra-waypoint",

        "search_radius_m":
            min(
                5000,
                int(
                    target_distance_miles
                    *
                    300
                )
            ),

        "attempts":
            700,

        "anchor_counts":
            [4, 4, 5],

        "min_anchor_distance_m":
            400,

        "min_anchor_separation_m":
            300
    }


# =========================================================
# QUALITY LIMITS
# =========================================================

def get_route_quality_limits(
    target_distance_miles,
    target_gain_ft
):

    # Short routes should be quite close
    # to the requested mileage.
    if target_distance_miles < 4:

        distance_error_limit_miles = max(

            0.15,

            target_distance_miles
            *
            0.08
        )


    else:

        distance_error_limit_miles = min(

            0.50,

            max(
                0.20,
                target_distance_miles
                *
                0.06
            )
        )


    if target_gain_ft <= 300:

        gain_error_limit_ft = 100


    elif target_gain_ft <= 1000:

        gain_error_limit_ft = max(

            125,

            target_gain_ft
            *
            0.25
        )


    else:

        gain_error_limit_ft = max(

            150,

            target_gain_ft
            *
            0.18
        )


    return {

        "distance_error_limit_miles":
            distance_error_limit_miles,

        "gain_error_limit_ft":
            gain_error_limit_ft,

        "excellent_distance_error_miles":
            max(
                0.06,
                target_distance_miles
                *
                0.025
            ),

        "excellent_gain_error_ft":
            max(
                40,
                target_gain_ft
                *
                0.08
            )
    }


# =========================================================
# EDGE HELPERS
# =========================================================

def get_shortest_edge(
    G,
    u,
    v
):

    edge_data = (
        G.get_edge_data(
            u,
            v
        )
    )


    if not edge_data:
        return None


    return min(

        edge_data.values(),

        key=lambda edge:

            float(
                edge.get(
                    "length",
                    float("inf")
                )
            )
    )


def path_distance_meters(
    G,
    route_nodes
):

    total = 0.0


    for i in range(
        len(route_nodes) - 1
    ):

        edge = get_shortest_edge(

            G,

            route_nodes[i],

            route_nodes[
                i + 1
            ]
        )


        if edge is not None:

            total += float(

                edge.get(
                    "length",
                    0
                )

                or

                0
            )


    return total


def path_gain_meters(
    G,
    route_nodes
):

    total = 0.0


    for i in range(
        len(route_nodes) - 1
    ):

        edge = get_shortest_edge(

            G,

            route_nodes[i],

            route_nodes[
                i + 1
            ]
        )


        if edge is not None:

            total += float(

                edge.get(
                    "ascent_m",
                    0
                )

                or

                0
            )


    return total


# =========================================================
# OSM GEOMETRY
# =========================================================

def oriented_edge_coords(
    G,
    u,
    v,
    edge
):

    geometry = (
        edge.get(
            "geometry"
        )
    )


    if geometry is not None:

        coords = [

            (
                float(lon),
                float(lat)
            )

            for lon, lat
            in geometry.coords
        ]


    else:

        coords = [

            (
                float(
                    G.nodes[u]["x"]
                ),

                float(
                    G.nodes[u]["y"]
                )
            ),

            (
                float(
                    G.nodes[v]["x"]
                ),

                float(
                    G.nodes[v]["y"]
                )
            )
        ]


    if not coords:
        return []


    u_lon = float(
        G.nodes[u]["x"]
    )

    u_lat = float(
        G.nodes[u]["y"]
    )


    first_lon, first_lat = (
        coords[0]
    )

    last_lon, last_lat = (
        coords[-1]
    )


    first_distance = (

        abs(
            first_lon - u_lon
        )

        +

        abs(
            first_lat - u_lat
        )
    )


    last_distance = (

        abs(
            last_lon - u_lon
        )

        +

        abs(
            last_lat - u_lat
        )
    )


    if last_distance < first_distance:

        coords.reverse()


    return coords


def route_coordinates(
    G,
    route_nodes
):

    coordinates = []


    for i in range(
        len(route_nodes) - 1
    ):

        u = route_nodes[i]

        v = route_nodes[
            i + 1
        ]


        edge = get_shortest_edge(
            G,
            u,
            v
        )


        if edge is None:
            continue


        edge_coords = (
            oriented_edge_coords(
                G,
                u,
                v,
                edge
            )
        )


        for lon, lat in edge_coords:

            point = {

                "lat":
                    float(lat),

                "lon":
                    float(lon)
            }


            if coordinates:

                previous = (
                    coordinates[-1]
                )


                if (

                    abs(
                        previous["lat"]
                        -
                        point["lat"]
                    )
                    <
                    0.0000001

                    and

                    abs(
                        previous["lon"]
                        -
                        point["lon"]
                    )
                    <
                    0.0000001
                ):

                    continue


            coordinates.append(
                point
            )


    return coordinates


# =========================================================
# TRAIL FILTER
# =========================================================

def edge_is_allowed_trail(data):

    highways = (
        normalize_tag_values(
            data.get(
                "highway"
            )
        )
    )


    surfaces = (
        normalize_tag_values(
            data.get(
                "surface"
            )
        )
    )


    access = (
        normalize_tag_values(
            data.get(
                "access"
            )
        )
    )


    foot = (
        normalize_tag_values(
            data.get(
                "foot"
            )
        )
    )


    area = (
        normalize_tag_values(
            data.get(
                "area"
            )
        )
    )


    indoor = (
        normalize_tag_values(
            data.get(
                "indoor"
            )
        )
    )


    footway = (
        normalize_tag_values(
            data.get(
                "footway"
            )
        )
    )


    if not highways.intersection(
        {
            "path",
            "track",
            "steps"
        }
    ):

        return False


    if footway.intersection(
        {
            "sidewalk",
            "crossing"
        }
    ):

        return False


    hard_surfaces = {

        "asphalt",

        "concrete",

        "concrete:lanes",

        "concrete:plates",

        "paving_stones",

        "sett",

        "cobblestone"
    }


    if surfaces.intersection(
        hard_surfaces
    ):

        return False


    if "yes" in area:
        return False


    if "yes" in indoor:
        return False


    if access.intersection(
        {
            "no",
            "private"
        }
    ):

        if not foot.intersection(
            {
                "yes",
                "designated",
                "permissive"
            }
        ):

            return False


    return True


# =========================================================
# DENSIFY TRAIL GEOMETRY
# =========================================================

def densify_polyline(
    coords,
    spacing_m=
        ELEVATION_SAMPLE_SPACING_M
):

    if not coords:
        return []


    if len(coords) == 1:
        return coords[:]


    dense = [
        coords[0]
    ]


    for i in range(
        len(coords) - 1
    ):

        lon1, lat1 = (
            coords[i]
        )

        lon2, lat2 = (
            coords[i + 1]
        )


        segment_distance = (
            haversine_meters(
                lat1,
                lon1,
                lat2,
                lon2
            )
        )


        if segment_distance <= 0:
            continue


        steps = max(

            1,

            int(
                math.ceil(
                    segment_distance
                    /
                    spacing_m
                )
            )
        )


        for step in range(
            1,
            steps + 1
        ):

            fraction = (
                step
                /
                steps
            )


            lon = (

                lon1

                +

                (
                    lon2 - lon1
                )
                *
                fraction
            )


            lat = (

                lat1

                +

                (
                    lat2 - lat1
                )
                *
                fraction
            )


            dense.append(
                (
                    float(lon),
                    float(lat)
                )
            )


    return dense


# =========================================================
# DEM SMOOTHING
# =========================================================

def smooth_elevations(values):

    if len(values) < 5:

        return [
            float(v)
            for v in values
        ]


    smoothed = []


    for i in range(
        len(values)
    ):

        start = max(
            0,
            i - 2
        )


        end = min(
            len(values),
            i + 3
        )


        window = values[
            start:end
        ]


        smoothed.append(

            sum(
                float(v)
                for v
                in window
            )

            /

            len(window)
        )


    return smoothed


# =========================================================
# DEM UNIT
# =========================================================

def dem_value_to_meters(
    src,
    value
):

    unit = ""


    try:

        if src.units:

            unit = (
                src.units[0]
                or
                ""
            )

    except Exception:

        unit = ""


    unit = str(
        unit
    ).lower()


    if (
        "foot" in unit
        or
        "feet" in unit
        or
        unit == "ft"
    ):

        return (

            float(value)

            /

            FEET_PER_METER
        )


    return float(
        value
    )


# =========================================================
# DEM SAMPLING
# =========================================================

def sample_dem_points(points):

    if not os.path.exists(
        DEM_PATH
    ):

        raise HTTPException(

            status_code=500,

            detail=(
                "DEM file not found: "
                +
                DEM_PATH
            )
        )


    unique = {}


    for lon, lat in points:

        key = (

            round(
                float(lat),
                7
            ),

            round(
                float(lon),
                7
            )
        )


        unique[key] = (

            float(lon),

            float(lat)
        )


    keys = list(
        unique.keys()
    )


    lons = [

        unique[key][0]

        for key
        in keys
    ]


    lats = [

        unique[key][1]

        for key
        in keys
    ]


    with rasterio.open(
        DEM_PATH
    ) as src:


        if src.crs is None:

            raise HTTPException(

                status_code=500,

                detail=(
                    "DEM has no CRS information."
                )
            )


        xs, ys = rio_transform(

            "EPSG:4326",

            src.crs,

            lons,

            lats
        )


        outside = []


        for (
            key,
            x,
            y
        ) in zip(
            keys,
            xs,
            ys
        ):

            if not (

                src.bounds.left
                <=
                x
                <=
                src.bounds.right

                and

                src.bounds.bottom
                <=
                y
                <=
                src.bounds.top
            ):

                outside.append(
                    key
                )


        if outside:

            first_lat, first_lon = (
                outside[0]
            )


            raise HTTPException(

                status_code=400,

                detail=(

                    "DEM does not cover the entire "
                    "requested trail graph. First "
                    "uncovered point: "

                    +

                    str(first_lat)

                    +

                    ", "

                    +

                    str(first_lon)
                )
            )


        samples = list(

            src.sample(

                zip(
                    xs,
                    ys
                ),

                indexes=1,

                masked=True
            )
        )


        lookup = {}


        for (
            key,
            sample
        ) in zip(
            keys,
            samples
        ):

            value = (
                sample[0]
            )


            if np.ma.is_masked(
                value
            ):

                raise HTTPException(

                    status_code=400,

                    detail=(
                        "DEM contains NoData at "
                        +
                        str(key)
                    )
                )


            value = float(
                value
            )


            if not math.isfinite(
                value
            ):

                raise HTTPException(

                    status_code=400,

                    detail=(
                        "DEM returned invalid elevation."
                    )
                )


            if (
                src.nodata
                is not None

                and

                math.isclose(

                    value,

                    float(
                        src.nodata
                    ),

                    rel_tol=0.0,

                    abs_tol=1e-6
                )
            ):

                raise HTTPException(

                    status_code=400,

                    detail=(
                        "DEM contains NoData."
                    )
                )


            lookup[key] = (
                dem_value_to_meters(
                    src,
                    value
                )
            )


    return lookup


# =========================================================
# ADD ELEVATION TO EDGES
# =========================================================

def add_local_dem_edge_elevations(G):

    edge_samples = {}

    all_points = []


    for (
        u,
        v,
        key,
        data
    ) in G.edges(
        keys=True,
        data=True
    ):

        coords = (
            oriented_edge_coords(
                G,
                u,
                v,
                data
            )
        )


        samples = (
            densify_polyline(

                coords,

                ELEVATION_SAMPLE_SPACING_M
            )
        )


        if len(samples) < 2:

            samples = [

                (
                    float(
                        G.nodes[u]["x"]
                    ),

                    float(
                        G.nodes[u]["y"]
                    )
                ),

                (
                    float(
                        G.nodes[v]["x"]
                    ),

                    float(
                        G.nodes[v]["y"]
                    )
                )
            ]


        edge_samples[
            (
                u,
                v,
                key
            )
        ] = (
            samples
        )


        all_points.extend(
            samples
        )


    elevation_lookup = (
        sample_dem_points(
            all_points
        )
    )


    # Node elevations
    node_points = [

        (
            float(
                G.nodes[node]["x"]
            ),

            float(
                G.nodes[node]["y"]
            )
        )

        for node
        in G.nodes
    ]


    node_lookup = (
        sample_dem_points(
            node_points
        )
    )


    for node in G.nodes:

        lat = float(
            G.nodes[node]["y"]
        )

        lon = float(
            G.nodes[node]["x"]
        )


        node_key = (

            round(
                lat,
                7
            ),

            round(
                lon,
                7
            )
        )


        G.nodes[node][
            "elevation"
        ] = float(
            node_lookup[
                node_key
            ]
        )


    # Directed ascent/descent.
    for (
        (
            u,
            v,
            key
        ),
        samples
    ) in edge_samples.items():


        elevations = []


        for lon, lat in samples:

            sample_key = (

                round(
                    float(lat),
                    7
                ),

                round(
                    float(lon),
                    7
                )
            )


            elevations.append(

                float(
                    elevation_lookup[
                        sample_key
                    ]
                )
            )


        elevations = (
            smooth_elevations(
                elevations
            )
        )


        ascent = 0.0
        descent = 0.0


        for i in range(
            len(elevations) - 1
        ):

            delta = (

                elevations[
                    i + 1
                ]

                -

                elevations[i]
            )


            if delta > 0:

                ascent += delta


            elif delta < 0:

                descent += (
                    -delta
                )


        G[u][v][key][
            "ascent_m"
        ] = float(
            ascent
        )


        G[u][v][key][
            "descent_m"
        ] = float(
            descent
        )


        G[u][v][key][
            "elevation_sample_count"
        ] = len(
            samples
        )


    return (
        G,
        len(
            elevation_lookup
        )
    )


# =========================================================
# GRAPH DOWNLOAD
# =========================================================

def download_trail_graph(
    lat,
    lon,
    radius_meters
):

    cache_key = (

        round(
            float(lat),
            5
        ),

        round(
            float(lon),
            5
        ),

        int(
            radius_meters
        ),

        ELEVATION_SAMPLE_SPACING_M
    )


    if cache_key in GRAPH_CACHE:

        cached = (
            GRAPH_CACHE[
                cache_key
            ]
        )


        return (

            cached["graph"],

            cached[
                "filtered_edges_removed"
            ],

            True,

            cached[
                "unique_elevation_samples"
            ]
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

        "trail_visibility"
    ]


    useful_tags = list(
        ox.settings.useful_tags_way
    )


    for tag in extra_tags:

        if tag not in useful_tags:

            useful_tags.append(
                tag
            )


    ox.settings.useful_tags_way = (
        useful_tags
    )


    trail_filter = (
        '["highway"~"path|track|steps"]'
    )


    G = ox.graph.graph_from_point(

        (
            lat,
            lon
        ),

        dist=
            radius_meters,

        network_type=
            "walk",

        custom_filter=
            trail_filter,

        simplify=
            True,

        retain_all=
            True
    )


    original_edges = (
        G.number_of_edges()
    )


    remove_edges = []


    for (
        u,
        v,
        key,
        data
    ) in G.edges(
        keys=True,
        data=True
    ):

        if not edge_is_allowed_trail(
            data
        ):

            remove_edges.append(
                (
                    u,
                    v,
                    key
                )
            )


    G.remove_edges_from(
        remove_edges
    )


    G.remove_nodes_from(
        list(
            nx.isolates(
                G
            )
        )
    )


    if (
        G.number_of_nodes() == 0
        or
        G.number_of_edges() == 0
    ):

        raise HTTPException(

            status_code=400,

            detail=(
                "No usable trail network found."
            )
        )


    nearest = (
        ox.distance.nearest_nodes(

            G,

            X=lon,

            Y=lat
        )
    )


    component = (
        nx.node_connected_component(

            G.to_undirected(
                as_view=True
            ),

            nearest
        )
    )


    G = (
        G.subgraph(
            component
        )
        .copy()
    )


    filtered_edges_removed = (

        original_edges

        -

        G.number_of_edges()
    )


    (
        G,
        unique_elevation_samples
    ) = add_local_dem_edge_elevations(
        G
    )


    if (
        len(
            GRAPH_CACHE
        )
        >=
        MAX_CACHED_GRAPHS
    ):

        oldest = next(
            iter(
                GRAPH_CACHE
            )
        )

        GRAPH_CACHE.pop(
            oldest
        )


    GRAPH_CACHE[
        cache_key
    ] = {

        "graph":
            G,

        "filtered_edges_removed":
            filtered_edges_removed,

        "unique_elevation_samples":
            unique_elevation_samples
    }


    return (

        G,

        filtered_edges_removed,

        False,

        unique_elevation_samples
    )


# =========================================================
# SIMPLE ROUTING GRAPH
# =========================================================

def make_simple_routing_graph(G):

    S = nx.DiGraph()


    S.add_nodes_from(
        G.nodes(
            data=True
        )
    )


    for (
        u,
        v,
        data
    ) in G.edges(
        data=True
    ):

        length = float(

            data.get(
                "length",
                0
            )

            or

            0
        )


        ascent = float(

            data.get(
                "ascent_m",
                0
            )

            or

            0
        )


        if length <= 0:
            continue


        if (
            not S.has_edge(
                u,
                v
            )

            or

            length
            <
            S[u][v]["length"]
        ):

            S.add_edge(

                u,
                v,

                length=
                    length,

                ascent_m=
                    ascent
            )


    return S


# =========================================================
# REPEAT METRICS
# =========================================================

def repeated_edge_stats(
    G,
    route_nodes
):

    counts = {}
    lengths = {}


    for i in range(
        len(route_nodes) - 1
    ):

        u = route_nodes[i]

        v = route_nodes[
            i + 1
        ]


        edge_key = (
            undirected_edge_key(
                u,
                v
            )
        )


        edge = get_shortest_edge(
            G,
            u,
            v
        )


        length = (

            float(
                edge.get(
                    "length",
                    0
                )

                or

                0
            )

            if edge

            else

            0.0
        )


        counts[edge_key] = (

            counts.get(
                edge_key,
                0
            )

            +

            1
        )


        lengths[edge_key] = (
            length
        )


    repeat_edges = 0
    repeat_distance = 0.0


    for (
        edge_key,
        count
    ) in counts.items():

        if count > 1:

            repeat_edges += (
                count - 1
            )


            repeat_distance += (

                lengths.get(
                    edge_key,
                    0
                )

                *

                (
                    count - 1
                )
            )


    return (
        repeat_edges,
        repeat_distance
    )


def repeated_node_occurrences(
    route_nodes
):

    counts = {}


    for node in route_nodes[
        1:-1
    ]:

        counts[node] = (

            counts.get(
                node,
                0
            )

            +

            1
        )


    return sum(

        count - 1

        for count
        in counts.values()

        if count > 1
    )


def count_immediate_reversals(
    route_nodes
):

    count = 0


    for i in range(
        len(route_nodes) - 2
    ):

        if (
            route_nodes[i]
            ==
            route_nodes[
                i + 2
            ]
        ):

            count += 1


    return count


# =========================================================
# ROUTE SCORE
# =========================================================

def route_score(
    G,
    route_nodes,
    target_distance_meters,
    target_gain_meters
):

    total_distance = (
        path_distance_meters(
            G,
            route_nodes
        )
    )


    actual_gain = (
        path_gain_meters(
            G,
            route_nodes
        )
    )


    if total_distance <= 0:

        return (
            float("inf"),
            {}
        )


    distance_error = abs(

        total_distance

        -

        target_distance_meters
    )


    distance_ratio = (

        distance_error

        /

        target_distance_meters
    )


    gain_error = abs(

        actual_gain

        -

        target_gain_meters
    )


    if target_gain_meters > 0:

        gain_ratio = (

            gain_error

            /

            target_gain_meters
        )


    else:

        gain_ratio = (

            actual_gain

            /

            30.48
        )


    (
        repeated_edges,
        repeated_distance
    ) = repeated_edge_stats(
        G,
        route_nodes
    )


    repeat_ratio = (

        repeated_distance

        /

        total_distance
    )


    repeated_nodes = (
        repeated_node_occurrences(
            route_nodes
        )
    )


    immediate_reversals = (
        count_immediate_reversals(
            route_nodes
        )
    )


    # Short routes must be allowed to share
    # a short access stem.
    if (
        target_distance_meters
        <
        4
        *
        METERS_PER_MILE
    ):

        repeat_weight = 100.0
        node_weight = 20.0


    else:

        repeat_weight = 320.0
        node_weight = 60.0


    score = (

        distance_ratio
        *
        180.0

        +

        gain_ratio
        *
        220.0

        +

        repeat_ratio
        *
        repeat_weight

        +

        repeated_nodes
        *
        node_weight

        +

        immediate_reversals
        *
        30.0
    )


    return (

        score,

        {

            "total_distance_meters":
                total_distance,

            "actual_gain_meters":
                actual_gain,

            "distance_error_meters":
                distance_error,

            "gain_error_meters":
                gain_error,

            "repeated_edges":
                repeated_edges,

            "repeated_distance_meters":
                repeated_distance,

            "repeat_ratio":
                repeat_ratio,

            "repeated_nodes":
                repeated_nodes,

            "immediate_reversals":
                immediate_reversals,

            "score":
                score
        }
    )


# =========================================================
# SHORT ROUTE BEAM SEARCH
# =========================================================

def beam_search_short_loop(
    G,
    start_node,
    target_distance_meters,
    target_gain_meters,
    target_distance_miles,
    target_gain_ft,
    limits,
    profile
):
    """
    Search trail SEGMENTS directly.

    This is very different from the waypoint method.

    Each state represents an actual partial walk through
    the trail network.

    We keep the best partial possibilities, expand one
    trail segment at a time, and constantly test whether
    they can return to the start near the requested
    distance/elevation.
    """

    S = (
        make_simple_routing_graph(
            G
        )
    )


    reverse_S = (
        S.reverse(
            copy=False
        )
    )


    try:

        (
            return_distances,
            reverse_paths
        ) = nx.single_source_dijkstra(

            reverse_S,

            start_node,

            weight="length"
        )


    except Exception as exc:

        raise HTTPException(

            status_code=500,

            detail=(
                "Could not prepare short-route "
                "return paths: "
                +
                str(exc)
            )
        )


    # Convert paths in reversed graph:
    #
    # reversed graph:
    # START -> X -> CURRENT
    #
    # original graph:
    # CURRENT -> X -> START
    return_paths = {}


    for (
        node,
        reversed_path
    ) in reverse_paths.items():

        return_paths[node] = (
            list(
                reversed(
                    reversed_path
                )
            )
        )


    distance_limit_m = (

        limits[
            "distance_error_limit_miles"
        ]

        *

        METERS_PER_MILE
    )


    gain_limit_m = (

        limits[
            "gain_error_limit_ft"
        ]

        /

        FEET_PER_METER
    )


    max_route_distance = (

        target_distance_meters

        +

        distance_limit_m
    )


    max_route_gain = (

        target_gain_meters

        +

        gain_limit_m
    )


    beam_width = (
        profile[
            "beam_width"
        ]
    )


    max_steps = (
        profile[
            "beam_max_steps"
        ]
    )


    # State fields:
    #
    # route
    # node
    # distance
    # gain
    # used_edges
    # visited_nodes
    # repeated_distance
    # reversals

    initial_state = {

        "route":
            (
                start_node,
            ),

        "node":
            start_node,

        "distance":
            0.0,

        "gain":
            0.0,

        "used_edges":
            frozenset(),

        "visited_nodes":
            frozenset(
                {
                    start_node
                }
            ),

        "repeated_distance":
            0.0,

        "reversals":
            0
    }


    beam = [
        initial_state
    ]


    best_route = None
    best_metrics = None
    best_score = float(
        "inf"
    )


    best_any_route = None
    best_any_metrics = None
    best_any_score = float(
        "inf"
    )


    target_gain_density = (

        target_gain_meters

        /

        target_distance_meters
    )


    for depth in range(
        max_steps
    ):

        expanded = []


        for state in beam:

            current = (
                state["node"]
            )


            for neighbor in S.successors(
                current
            ):

                edge_data = (
                    S[current][neighbor]
                )


                edge_length = float(

                    edge_data.get(
                        "length",
                        0
                    )
                )


                edge_gain = float(

                    edge_data.get(
                        "ascent_m",
                        0
                    )
                )


                if edge_length <= 0:
                    continue


                new_distance = (

                    state["distance"]

                    +

                    edge_length
                )


                new_gain = (

                    state["gain"]

                    +

                    edge_gain
                )


                # If we already exceeded the maximum
                # acceptable gain, no continuation can
                # fix it because ascent never decreases.
                if (
                    new_gain
                    >
                    max_route_gain
                ):

                    continue


                # Distance can't shrink either.
                if (
                    new_distance
                    >
                    max_route_distance
                ):

                    continue


                edge_key = (
                    undirected_edge_key(
                        current,
                        neighbor
                    )
                )


                already_used = (

                    edge_key

                    in

                    state[
                        "used_edges"
                    ]
                )


                repeated_distance = (

                    state[
                        "repeated_distance"
                    ]

                    +

                    (
                        edge_length

                        if already_used

                        else

                        0
                    )
                )


                reversal = 0


                route = (
                    state["route"]
                )


                if (
                    len(route) >= 2

                    and

                    neighbor
                    ==
                    route[-2]
                ):

                    reversal = 1


                new_reversals = (

                    state["reversals"]

                    +

                    reversal
                )


                new_route = (

                    route

                    +

                    (
                        neighbor,
                    )
                )


                used_edges = set(
                    state[
                        "used_edges"
                    ]
                )


                used_edges.add(
                    edge_key
                )


                visited_nodes = set(
                    state[
                        "visited_nodes"
                    ]
                )


                node_revisited = (

                    neighbor

                    in

                    visited_nodes
                )


                visited_nodes.add(
                    neighbor
                )


                # -----------------------------------------
                # CAN WE STILL GET HOME?
                # -----------------------------------------

                if neighbor not in return_distances:

                    continue


                return_distance = float(
                    return_distances[
                        neighbor
                    ]
                )


                minimum_final_distance = (

                    new_distance

                    +

                    return_distance
                )


                # Even the shortest route home would
                # make us too long.
                if (
                    minimum_final_distance

                    >

                    max_route_distance
                ):

                    continue


                # -----------------------------------------
                # TEST CLOSING THIS ROUTE
                # -----------------------------------------

                if neighbor == start_node:

                    candidate_route = list(
                        new_route
                    )


                else:

                    return_route = (
                        return_paths[
                            neighbor
                        ]
                    )


                    candidate_route = (

                        list(
                            new_route
                        )

                        +

                        return_route[
                            1:
                        ]
                    )


                candidate_distance = (
                    path_distance_meters(
                        G,
                        candidate_route
                    )
                )


                # Don't spend CPU scoring candidates
                # nowhere near the desired mileage.
                if (

                    candidate_distance

                    >=

                    target_distance_meters
                    *
                    0.72

                    and

                    candidate_distance

                    <=

                    max_route_distance
                ):

                    (
                        score,
                        metrics
                    ) = route_score(

                        G,

                        candidate_route,

                        target_distance_meters,

                        target_gain_meters
                    )


                    if score < best_any_score:

                        best_any_score = (
                            score
                        )

                        best_any_route = (
                            candidate_route
                        )

                        best_any_metrics = (
                            metrics
                        )


                    distance_error_miles = (

                        metrics[
                            "distance_error_meters"
                        ]

                        /

                        METERS_PER_MILE
                    )


                    gain_error_ft = (

                        metrics[
                            "gain_error_meters"
                        ]

                        *

                        FEET_PER_METER
                    )


                    acceptable = (

                        distance_error_miles

                        <=

                        limits[
                            "distance_error_limit_miles"
                        ]

                        and

                        gain_error_ft

                        <=

                        limits[
                            "gain_error_limit_ft"
                        ]
                    )


                    if acceptable:

                        if score < best_score:

                            best_score = (
                                score
                            )

                            best_route = (
                                candidate_route
                            )

                            best_metrics = (
                                metrics
                            )


                        excellent_distance = (

                            distance_error_miles

                            <=

                            limits[
                                "excellent_distance_error_miles"
                            ]
                        )


                        excellent_gain = (

                            gain_error_ft

                            <=

                            limits[
                                "excellent_gain_error_ft"
                            ]
                        )


                        if (

                            excellent_distance

                            and

                            excellent_gain

                            and

                            metrics[
                                "immediate_reversals"
                            ]
                            <=
                            1
                        ):

                            return (

                                best_route,

                                best_metrics,

                                depth + 1,

                                "beam"
                            )


                # -----------------------------------------
                # PARTIAL STATE PRIORITY
                # -----------------------------------------

                # Estimated final mileage using the
                # shortest possible path back to start.
                estimated_final_distance = (
                    minimum_final_distance
                )


                distance_priority = abs(

                    estimated_final_distance

                    -

                    target_distance_meters
                ) / target_distance_meters


                if new_distance > 0:

                    current_gain_density = (

                        new_gain

                        /

                        new_distance
                    )


                else:

                    current_gain_density = 0.0


                # This is key for requests such as
                # 2.5 mi / 200 ft.
                #
                # Flat partial routes remain near the
                # front of the beam instead of getting
                # discarded in favor of steep mountain
                # routes.
                density_denominator = max(

                    target_gain_density,

                    0.005
                )


                gain_density_priority = abs(

                    current_gain_density

                    -

                    target_gain_density

                ) / density_denominator


                repeat_priority = (

                    repeated_distance

                    /

                    max(
                        new_distance,
                        1
                    )
                )


                revisit_penalty = (

                    0.04

                    if node_revisited

                    else

                    0.0
                )


                reversal_penalty = (

                    new_reversals

                    *
                    0.08
                )


                # Overshooting gain is especially bad.
                gain_overshoot = max(

                    0.0,

                    new_gain
                    -
                    target_gain_meters
                )


                gain_overshoot_priority = (

                    gain_overshoot

                    /

                    max(
                        target_gain_meters,
                        20
                    )
                )


                priority = (

                    distance_priority
                    *
                    3.0

                    +

                    gain_density_priority
                    *
                    1.8

                    +

                    gain_overshoot_priority
                    *
                    4.0

                    +

                    repeat_priority
                    *
                    0.8

                    +

                    revisit_penalty

                    +

                    reversal_penalty
                )


                expanded.append(

                    (
                        priority,

                        {

                            "route":
                                new_route,

                            "node":
                                neighbor,

                            "distance":
                                new_distance,

                            "gain":
                                new_gain,

                            "used_edges":
                                frozenset(
                                    used_edges
                                ),

                            "visited_nodes":
                                frozenset(
                                    visited_nodes
                                ),

                            "repeated_distance":
                                repeated_distance,

                            "reversals":
                                new_reversals
                        }
                    )
                )


        if not expanded:
            break


        # ---------------------------------------------
        # DEDUPLICATE SIMILAR PARTIAL ROUTES
        # ---------------------------------------------

        expanded.sort(
            key=lambda item:
                item[0]
        )


        next_beam = []

        seen_buckets = set()


        for (
            priority,
            state
        ) in expanded:

            distance_bucket = int(

                state["distance"]

                /
                75.0
            )


            gain_bucket = int(

                state["gain"]

                /
                5.0
            )


            bucket = (

                state["node"],

                distance_bucket,

                gain_bucket
            )


            if bucket in seen_buckets:
                continue


            seen_buckets.add(
                bucket
            )


            next_beam.append(
                state
            )


            if (
                len(next_beam)
                >=
                beam_width
            ):

                break


        beam = (
            next_beam
        )


    # Return only an acceptable route.
    if best_route is not None:

        return (

            best_route,

            best_metrics,

            max_steps,

            "beam"
        )


    # Give useful information if search found
    # something but couldn't meet limits.
    if best_any_route is not None:

        best_distance = (

            best_any_metrics[
                "total_distance_meters"
            ]

            /

            METERS_PER_MILE
        )


        best_gain = (

            best_any_metrics[
                "actual_gain_meters"
            ]

            *

            FEET_PER_METER
        )


        raise HTTPException(

            status_code=400,

            detail=(

                "Beam search could not find a "
                "trail-only route within the quality "
                "limits. Best candidate was "

                +

                str(
                    round(
                        best_distance,
                        2
                    )
                )

                +

                " mi / "

                +

                str(
                    round(
                        best_gain
                    )
                )

                +

                " ft gain."
            )
        )


    raise HTTPException(

        status_code=400,

        detail=(
            "Beam search could not find a "
            "suitable loop."
        )
    )


# =========================================================
# LONG ROUTE PATH SEARCH
# =========================================================

def waypoint_path(
    S,
    source,
    target,
    used_edges
):

    def weight(
        u,
        v,
        data
    ):

        cost = float(
            data.get(
                "length",
                1
            )
        )


        if (
            undirected_edge_key(
                u,
                v
            )

            in

            used_edges
        ):

            cost *= 40.0


        return cost


    return nx.shortest_path(

        S,

        source,

        target,

        weight=
            weight
    )


# =========================================================
# LONG/MEDIUM WAYPOINT SEARCH
# =========================================================

def generate_waypoint_loop(
    G,
    start_node,
    target_distance_meters,
    target_gain_meters,
    target_distance_miles,
    target_gain_ft,
    profile,
    limits
):

    S = (
        make_simple_routing_graph(
            G
        )
    )


    start_lat = float(
        G.nodes[start_node]["y"]
    )

    start_lon = float(
        G.nodes[start_node]["x"]
    )


    candidates = []


    max_radial_distance = min(

        profile[
            "search_radius_m"
        ]
        *
        0.90,

        target_distance_meters
        *
        0.32
    )


    for node in S.nodes:

        if node == start_node:
            continue


        radial = (
            haversine_meters(

                start_lat,

                start_lon,

                float(
                    G.nodes[node]["y"]
                ),

                float(
                    G.nodes[node]["x"]
                )
            )
        )


        if (

            radial

            >=

            profile[
                "min_anchor_distance_m"
            ]

            and

            radial

            <=

            max_radial_distance
        ):

            candidates.append(
                node
            )


    if (
        len(candidates)
        <
        max(
            profile[
                "anchor_counts"
            ]
        )
    ):

        raise HTTPException(

            status_code=400,

            detail=(
                "Not enough trail junctions "
                "were found for this route."
            )
        )


    best_route = None
    best_metrics = None
    best_score = float(
        "inf"
    )


    best_any_route = None
    best_any_metrics = None
    best_any_score = float(
        "inf"
    )


    for _ in range(
        profile[
            "attempts"
        ]
    ):

        anchor_count = random.choice(
            profile[
                "anchor_counts"
            ]
        )


        anchors = random.sample(
            candidates,
            anchor_count
        )


        # Reject anchors too close together.
        spacing_bad = False


        for i in range(
            len(anchors)
        ):

            for j in range(
                i + 1,
                len(anchors)
            ):

                a = anchors[i]
                b = anchors[j]


                separation = (
                    haversine_meters(

                        float(
                            G.nodes[a]["y"]
                        ),

                        float(
                            G.nodes[a]["x"]
                        ),

                        float(
                            G.nodes[b]["y"]
                        ),

                        float(
                            G.nodes[b]["x"]
                        )
                    )
                )


                if (

                    separation

                    <

                    profile[
                        "min_anchor_separation_m"
                    ]
                ):

                    spacing_bad = True
                    break


            if spacing_bad:
                break


        if spacing_bad:
            continue


        anchors.sort(

            key=lambda node:

                node_angle_from_start(

                    G,

                    start_node,

                    node
                )
        )


        if random.random() < 0.5:

            anchors.reverse()


        route = [
            start_node
        ]


        used_edges = set()

        current = start_node

        failed = False


        for destination in (
            anchors
            +
            [start_node]
        ):

            try:

                leg = (
                    waypoint_path(

                        S,

                        current,

                        destination,

                        used_edges
                    )
                )


            except nx.NetworkXNoPath:

                failed = True
                break


            for i in range(
                len(leg) - 1
            ):

                used_edges.add(

                    undirected_edge_key(

                        leg[i],

                        leg[
                            i + 1
                        ]
                    )
                )


            route.extend(
                leg[1:]
            )


            current = destination


        if failed:
            continue


        (
            score,
            metrics
        ) = route_score(

            G,

            route,

            target_distance_meters,

            target_gain_meters
        )


        distance = (
            metrics[
                "total_distance_meters"
            ]
        )


        if (

            distance

            <

            target_distance_meters
            *
            0.72

            or

            distance

            >

            target_distance_meters
            *
            1.25
        ):

            continue


        if score < best_any_score:

            best_any_score = (
                score
            )

            best_any_route = (
                route
            )

            best_any_metrics = (
                metrics
            )


        distance_error_miles = (

            metrics[
                "distance_error_meters"
            ]

            /

            METERS_PER_MILE
        )


        gain_error_ft = (

            metrics[
                "gain_error_meters"
            ]

            *

            FEET_PER_METER
        )


        if (

            distance_error_miles

            <=

            limits[
                "distance_error_limit_miles"
            ]

            and

            gain_error_ft

            <=

            limits[
                "gain_error_limit_ft"
            ]
        ):

            if score < best_score:

                best_score = (
                    score
                )

                best_route = (
                    route
                )

                best_metrics = (
                    metrics
                )


    if best_route is not None:

        return (

            best_route,

            best_metrics,

            profile[
                "attempts"
            ],

            "waypoint"
        )


    if best_any_route is not None:

        best_distance = (

            best_any_metrics[
                "total_distance_meters"
            ]

            /

            METERS_PER_MILE
        )


        best_gain = (

            best_any_metrics[
                "actual_gain_meters"
            ]

            *

            FEET_PER_METER
        )


        raise HTTPException(

            status_code=400,

            detail=(

                "No route met the requested quality "
                "limits. Best candidate was "

                +

                str(
                    round(
                        best_distance,
                        2
                    )
                )

                +

                " mi / "

                +

                str(
                    round(
                        best_gain
                    )
                )

                +

                " ft gain."
            )
        )


    raise HTTPException(

        status_code=400,

        detail=(
            "No suitable waypoint route found."
        )
    )


# =========================================================
# DEM INFO
# =========================================================

@app.get(
    "/dem-info"
)
def dem_info():

    if not os.path.exists(
        DEM_PATH
    ):

        raise HTTPException(

            status_code=404,

            detail=(
                "DEM file not found."
            )
        )


    with rasterio.open(
        DEM_PATH
    ) as src:

        return {

            "file":
                os.path.basename(
                    DEM_PATH
                ),

            "crs":
                str(
                    src.crs
                ),

            "width":
                src.width,

            "height":
                src.height,

            "bounds": {

                "left":
                    src.bounds.left,

                "bottom":
                    src.bounds.bottom,

                "right":
                    src.bounds.right,

                "top":
                    src.bounds.top
            },

            "pixel_size": {

                "x":
                    abs(
                        src.transform.a
                    ),

                "y":
                    abs(
                        src.transform.e
                    )
            },

            "elevation_sample_spacing_m":
                ELEVATION_SAMPLE_SPACING_M
        }


# =========================================================
# GENERATE ROUTE
# =========================================================

@app.post(
    "/generate-route"
)
def generate_route(
    request: RouteRequest
):

    try:

        if (
            request.target_distance_miles
            <=
            0
        ):

            raise HTTPException(

                status_code=400,

                detail=(
                    "Distance must be greater than 0."
                )
            )


        if request.target_gain_ft < 0:

            raise HTTPException(

                status_code=400,

                detail=(
                    "Elevation gain cannot be negative."
                )
            )


        target_distance_meters = (

            request.target_distance_miles

            *

            METERS_PER_MILE
        )


        target_gain_meters = (

            request.target_gain_ft

            /

            FEET_PER_METER
        )


        profile = (
            get_route_profile(
                request.target_distance_miles
            )
        )


        limits = (
            get_route_quality_limits(

                request.target_distance_miles,

                request.target_gain_ft
            )
        )


        (
            G,
            filtered_edges_removed,
            graph_from_cache,
            unique_elevation_samples
        ) = download_trail_graph(

            request.start_lat,

            request.start_lon,

            profile[
                "search_radius_m"
            ]
        )


        start_node = (
            ox.distance.nearest_nodes(

                G,

                X=request.start_lon,

                Y=request.start_lat
            )
        )


        end_node = (
            ox.distance.nearest_nodes(

                G,

                X=request.end_lon,

                Y=request.end_lat
            )
        )


        same_point = (

            abs(
                request.start_lat
                -
                request.end_lat
            )
            <
            0.0001

            and

            abs(
                request.start_lon
                -
                request.end_lon
            )
            <
            0.0001
        )


        if same_point:

            # -----------------------------------------
            # SHORT ROUTES:
            # SEGMENT-BY-SEGMENT BEAM SEARCH
            # -----------------------------------------

            if (
                request.target_distance_miles
                <
                4.0
            ):

                (
                    route_nodes,
                    metrics,
                    search_steps,
                    search_method
                ) = beam_search_short_loop(

                    G,

                    start_node,

                    target_distance_meters,

                    target_gain_meters,

                    request.target_distance_miles,

                    request.target_gain_ft,

                    limits,

                    profile
                )


                route_type = (
                    "short trail-segment beam loop"
                )


            # -----------------------------------------
            # LONGER ROUTES:
            # WAYPOINT SEARCH
            # -----------------------------------------

            else:

                (
                    route_nodes,
                    metrics,
                    search_steps,
                    search_method
                ) = generate_waypoint_loop(

                    G,

                    start_node,

                    target_distance_meters,

                    target_gain_meters,

                    request.target_distance_miles,

                    request.target_gain_ft,

                    profile,

                    limits
                )


                route_type = (
                    "adaptive waypoint trail loop"
                )


        # ---------------------------------------------
        # POINT TO POINT
        # ---------------------------------------------

        else:

            S = (
                make_simple_routing_graph(
                    G
                )
            )


            try:

                route_nodes = (
                    nx.shortest_path(

                        S,

                        start_node,

                        end_node,

                        weight="length"
                    )
                )


            except nx.NetworkXNoPath:

                raise HTTPException(

                    status_code=400,

                    detail=(
                        "No connected trail route found "
                        "between start and finish."
                    )
                )


            (
                _,
                metrics
            ) = route_score(

                G,

                route_nodes,

                target_distance_meters,

                target_gain_meters
            )


            search_steps = 1

            search_method = (
                "point-to-point"
            )

            route_type = (
                "trail point-to-point"
            )


        route_distance_miles = (

            metrics[
                "total_distance_meters"
            ]

            /

            METERS_PER_MILE
        )


        actual_gain_ft = (

            metrics[
                "actual_gain_meters"
            ]

            *

            FEET_PER_METER
        )


        distance_error_miles = abs(

            route_distance_miles

            -

            request.target_distance_miles
        )


        elevation_error_ft = abs(

            actual_gain_ft

            -

            request.target_gain_ft
        )


        repeated_distance_miles = (

            metrics[
                "repeated_distance_meters"
            ]

            /

            METERS_PER_MILE
        )


        coords = (
            route_coordinates(

                G,

                route_nodes
            )
        )


        return {

            "requested_distance_miles":
                request.target_distance_miles,

            "actual_distance_miles":
                round(
                    route_distance_miles,
                    2
                ),

            "distance_error_miles":
                round(
                    distance_error_miles,
                    2
                ),

            "requested_gain_ft":
                request.target_gain_ft,

            "actual_gain_ft":
                round(
                    actual_gain_ft
                ),

            "elevation_error_ft":
                round(
                    elevation_error_ft
                ),

            "route_type":
                route_type,

            "search_method":
                search_method,

            "route_profile":
                profile[
                    "name"
                ],

            "route":
                coords,

            "route_nodes":
                len(
                    route_nodes
                ),

            "route_geometry_points":
                len(
                    coords
                ),

            "repeated_edges":
                metrics[
                    "repeated_edges"
                ],

            "repeated_distance_miles":
                round(
                    repeated_distance_miles,
                    2
                ),

            "repeated_nodes":
                metrics[
                    "repeated_nodes"
                ],

            "immediate_reversals":
                metrics[
                    "immediate_reversals"
                ],

            "route_score":
                round(
                    metrics[
                        "score"
                    ],
                    2
                ),

            "max_allowed_distance_error_miles":
                round(
                    limits[
                        "distance_error_limit_miles"
                    ],
                    2
                ),

            "max_allowed_gain_error_ft":
                round(
                    limits[
                        "gain_error_limit_ft"
                    ]
                ),

            "search_steps":
                search_steps,

            "search_radius_m":
                profile[
                    "search_radius_m"
                ],

            "network_nodes":
                G.number_of_nodes(),

            "network_edges":
                G.number_of_edges(),

            "filtered_edges_removed":
                filtered_edges_removed,

            "graph_from_cache":
                graph_from_cache,

            "unique_elevation_samples":
                unique_elevation_samples,

            "elevation_sample_spacing_m":
                ELEVATION_SAMPLE_SPACING_M,

            "elevation_source":
                os.path.basename(
                    DEM_PATH
                ),

            "status":
                "Route generated"
        }


    except HTTPException:

        raise


    except Exception as exc:

        raise HTTPException(

            status_code=500,

            detail=str(
                exc
            )
        )


# =========================================================
# MAP PAGE
# =========================================================

@app.get(
    "/map",
    response_class=HTMLResponse
)
def route_map():

    return r"""
<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1.0"
>

<title>
Trail Running Creator
</title>

<link
    rel="stylesheet"
    href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
>

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

input {
    width: 170px;
    padding: 8px;
    border: 1px solid #aaa;
    border-radius: 4px;
}

button {
    padding: 10px 18px;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    background: #222;
    color: white;
    font-size: 15px;
}

button:disabled {
    opacity: 0.5;
    cursor: wait;
}

#results {
    margin-top: 12px;
    line-height: 1.55;
}

#map {
    height: calc(100vh - 510px);
    min-height: 500px;
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

</style>

</head>

<body>

<div id="controls">

<h2>
Trail Running Creator
</h2>

<div class="input-row">

<div class="input-group">

<label for="start_lat">
Start latitude
</label>

<input
    id="start_lat"
    type="number"
    step="any"
    value="33.589281"
>

</div>

<div class="input-group">

<label for="start_lon">
Start longitude
</label>

<input
    id="start_lon"
    type="number"
    step="any"
    value="-112.091148"
>

</div>

<div class="input-group">

<label for="end_lat">
End latitude
</label>

<input
    id="end_lat"
    type="number"
    step="any"
    value="33.589281"
>

</div>

<div class="input-group">

<label for="end_lon">
End longitude
</label>

<input
    id="end_lon"
    type="number"
    step="any"
    value="-112.091148"
>

</div>

</div>

<div class="input-row">

<div class="input-group">

<label for="distance">
Target distance (miles)
</label>

<input
    id="distance"
    type="number"
    step="0.1"
    min="0.1"
    value="2.5"
>

</div>

<div class="input-group">

<label for="gain">
Target elevation gain (ft)
</label>

<input
    id="gain"
    type="number"
    step="50"
    min="0"
    value="200"
>

</div>

</div>

<button id="generateButton">
Generate Trail Route
</button>

<div id="results">
Ready.
</div>

</div>

<div id="map"></div>

<script
    src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js">
</script>

<script>

const map =
    L.map(
        "map"
    )
    .setView(
        [
            33.589281,
            -112.091148
        ],
        15
    );


L.tileLayer(
    "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    {
        maxZoom: 19,

        attribution:
            "&copy; OpenStreetMap contributors"
    }
).addTo(map);


let routeLine = null;
let startMarker = null;
let finishMarker = null;


const button =
    document.getElementById(
        "generateButton"
    );


button.addEventListener(
    "click",
    generateRoute
);


async function generateRoute() {

    const results =
        document.getElementById(
            "results"
        );


    const data = {

        start_lat:
            parseFloat(
                document.getElementById(
                    "start_lat"
                ).value
            ),

        start_lon:
            parseFloat(
                document.getElementById(
                    "start_lon"
                ).value
            ),

        end_lat:
            parseFloat(
                document.getElementById(
                    "end_lat"
                ).value
            ),

        end_lon:
            parseFloat(
                document.getElementById(
                    "end_lon"
                ).value
            ),

        target_distance_miles:
            parseFloat(
                document.getElementById(
                    "distance"
                ).value
            ),

        target_gain_ft:
            parseFloat(
                document.getElementById(
                    "gain"
                ).value
            )
    };


    results.innerHTML =
        '<span class="warning">' +
        "Loading trails and elevation, then searching " +
        "trail-segment combinations..." +
        "</span>";


    button.disabled = true;


    try {

        const response =
            await fetch(
                "/generate-route",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body:
                        JSON.stringify(
                            data
                        )
                }
            );


        const text =
            await response.text();


        if (!text) {

            throw new Error(
                "Server returned an empty response."
            );
        }


        let result;


        try {

            result = JSON.parse(
                text
            );

        }

        catch {

            throw new Error(
                "Invalid server response: "
                +
                text.substring(
                    0,
                    500
                )
            );
        }


        if (!response.ok) {

            throw new Error(
                result.detail
                ||
                "Server error."
            );
        }


        if (
            !result.route
            ||
            result.route.length < 2
        ) {

            throw new Error(
                "Server returned an empty route."
            );
        }


        const coordinates =
            result.route.map(

                point => [

                    point.lat,

                    point.lon
                ]
            );


        if (routeLine) {

            map.removeLayer(
                routeLine
            );
        }


        if (startMarker) {

            map.removeLayer(
                startMarker
            );
        }


        if (finishMarker) {

            map.removeLayer(
                finishMarker
            );
        }


        routeLine =
            L.polyline(
                coordinates,
                {
                    weight: 5,
                    opacity: 0.9
                }
            )
            .addTo(map);


        startMarker =
            L.marker(
                coordinates[0]
            )
            .addTo(map)
            .bindPopup(
                "Start"
            );


        finishMarker =
            L.marker(
                coordinates[
                    coordinates.length - 1
                ]
            )
            .addTo(map)
            .bindPopup(
                "Finish"
            );


        map.fitBounds(
            routeLine.getBounds(),
            {
                padding: [
                    30,
                    30
                ]
            }
        );


        results.innerHTML =

            '<span class="success">' +
            "<b>Route generated</b>" +
            "</span><br>" +

            "<b>Distance target:</b> " +
            result.requested_distance_miles +
            " mi<br>" +

            "<b>Actual distance:</b> " +
            result.actual_distance_miles +
            " mi<br>" +

            "<b>Distance error:</b> " +
            result.distance_error_miles +
            " mi<br><br>" +

            "<b>Elevation target:</b> " +
            result.requested_gain_ft +
            " ft<br>" +

            "<b>Actual elevation gain:</b> " +
            result.actual_gain_ft +
            " ft<br>" +

            "<b>Elevation error:</b> " +
            result.elevation_error_ft +
            " ft<br><br>" +

            "<b>Search method:</b> " +
            result.search_method +
            "<br>" +

            "<b>Route profile:</b> " +
            result.route_profile +
            "<br>" +

            "<b>Repeated trail distance:</b> " +
            result.repeated_distance_miles +
            " mi<br>" +

            "<b>Repeated edges:</b> " +
            result.repeated_edges +
            "<br>" +

            "<b>Repeated junctions:</b> " +
            result.repeated_nodes +
            "<br>" +

            "<b>Immediate reversals:</b> " +
            result.immediate_reversals +
            "<br>" +

            "<b>Route score:</b> " +
            result.route_score +
            "<br><br>" +

            "<b>Search radius:</b> " +
            result.search_radius_m +
            " m<br>" +

            "<b>Search steps:</b> " +
            result.search_steps +
            "<br>" +

            "<b>Graph cached:</b> " +
            result.graph_from_cache +
            "<br>" +

            "<b>Elevation samples:</b> " +
            result.unique_elevation_samples +
            "<br>" +

            "<b>Elevation sample spacing:</b> ~" +
            result.elevation_sample_spacing_m +
            " m<br>" +

            '<span class="small">' +
            "Elevation source: " +
            result.elevation_source +
            "</span>";

    }

    catch (error) {

        results.innerHTML =

            '<span class="error">' +

            "<b>Error:</b> " +

            error.message +

            "</span>";
    }

    finally {

        button.disabled = false;
    }
}

</script>

</body>

</html>
"""
