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

# Your TIFF is in the same folder as main.py.
DEM_PATH = "output_USGS30m.tif"

METERS_PER_MILE = 1609.344
FEET_PER_METER = 3.28084

SEARCH_RADIUS_METERS = 3000

# Since your DEM is approximately 30 m resolution,
# sample the trail roughly every 30 m.
ELEVATION_SAMPLE_SPACING_M = 30.0

# Number of route candidates to test.
ROUTE_ATTEMPTS = 500

MAX_CACHED_GRAPHS = 4

GRAPH_CACHE = {}


# =========================================================
# REQUEST MODEL
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
        "status": "Trail Running Creator API is running",
        "map": "/map",
        "docs": "/docs",
        "dem_info": "/dem-info"
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

            part = part.strip().lower()

            if part:
                result.add(part)

    return result


def undirected_edge_key(u, v):

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

    p1 = math.radians(lat1)
    p2 = math.radians(lat2)

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
# EDGE HELPERS
# =========================================================

def get_shortest_edge(
    G,
    u,
    v
):

    edge_data = G.get_edge_data(
        u,
        v
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
            route_nodes[i + 1]
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
            route_nodes[i + 1]
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
            float(
                S[u][v].get(
                    "length",
                    float("inf")
                )
            )
        ):

            S.add_edge(
                u,
                v,
                length=length
            )

    return S


def penalized_shortest_path(
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

        base = float(
            data.get(
                "length",
                1.0
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

            # Strong penalty for already-used trails.
            return (
                base
                *
                40.0
            )

        return base


    return nx.shortest_path(
        S,
        source,
        target,
        weight=weight
    )


# =========================================================
# FULL OSM TRAIL GEOMETRY
# =========================================================

def oriented_edge_coords(
    G,
    u,
    v,
    edge
):

    geometry = edge.get(
        "geometry"
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


    # Reverse the geometry if it is stored
    # opposite the direction we're traveling.
    if (
        last_distance
        <
        first_distance
    ):

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
        v = route_nodes[i + 1]


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


    if (
        not coordinates
        and
        route_nodes
    ):

        node = (
            route_nodes[0]
        )


        coordinates.append(

            {

                "lat":
                    float(
                        G.nodes[node]["y"]
                    ),

                "lon":
                    float(
                        G.nodes[node]["x"]
                    )
            }
        )


    return coordinates


# =========================================================
# TRAIL FILTERING
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


    # Only trail-like ways.
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
                    lon2
                    -
                    lon1
                )
                *
                fraction
            )


            lat = (
                lat1
                +
                (
                    lat2
                    -
                    lat1
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
# SMOOTH DEM ELEVATIONS
# =========================================================

def smooth_elevations(values):

    if len(values) < 3:

        return [
            float(v)
            for v in values
        ]


    smoothed = [
        float(
            values[0]
        )
    ]


    for i in range(
        1,
        len(values) - 1
    ):

        smoothed.append(

            (

                float(
                    values[
                        i - 1
                    ]
                )

                +

                float(
                    values[i]
                )

                +

                float(
                    values[
                        i + 1
                    ]
                )

            )

            /

            3.0
        )


    smoothed.append(
        float(
            values[-1]
        )
    )


    return smoothed


# =========================================================
# LOCAL TIFF ELEVATION SAMPLING
# =========================================================

def sample_dem_points(points):

    """
    Input:
        [(longitude, latitude), ...]

    Output:
        {
            (rounded_lat, rounded_lon): elevation_meters
        }
    """


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

        for key in keys
    ]


    lats = [

        unique[key][1]

        for key in keys
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


        # Convert standard lon/lat into the
        # coordinate system used by the TIFF.
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
                    "trail graph. "

                    +

                    str(
                        len(outside)
                    )

                    +

                    " elevation sample points are "
                    "outside the TIFF. "

                    +

                    "First uncovered point: "

                    +

                    str(
                        first_lat
                    )

                    +

                    ", "

                    +

                    str(
                        first_lon
                    )
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

                        str(
                            key[0]
                        )

                        +

                        ", "

                        +

                        str(
                            key[1]
                        )
                    )
                )


            elevation = float(
                value
            )


            if not math.isfinite(
                elevation
            ):

                raise HTTPException(

                    status_code=400,

                    detail=(

                        "DEM returned invalid elevation at "

                        +

                        str(
                            key[0]
                        )

                        +

                        ", "

                        +

                        str(
                            key[1]
                        )
                    )
                )


            if (
                src.nodata
                is not None

                and

                math.isclose(

                    elevation,

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

                        "DEM contains NoData at "

                        +

                        str(
                            key[0]
                        )

                        +

                        ", "

                        +

                        str(
                            key[1]
                        )
                    )
                )


            lookup[key] = (
                elevation
            )


    return lookup


# =========================================================
# PRE-CALCULATE ASCENT/DESCENT FOR EACH TRAIL
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


    # Read all geometry sample points from
    # the local GeoTIFF.
    elevation_lookup = (
        sample_dem_points(
            all_points
        )
    )


    # Also sample graph junction nodes.
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


        key = (

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
            node_lookup[key]
        )


    # Calculate ascent/descent separately
    # for each directed trail edge.
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


        # Slight smoothing prevents one DEM pixel
        # from artificially inflating accumulated gain.
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

            difference = (

                elevations[
                    i + 1
                ]

                -

                elevations[i]
            )


            if difference > 0:

                ascent += (
                    difference
                )


            elif difference < 0:

                descent += (
                    -difference
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
# DOWNLOAD / PREPARE TRAIL GRAPH
# =========================================================

def download_trail_graph(
    lat,
    lon,
    radius_meters=
        SEARCH_RADIUS_METERS
):


    cache_key = (

        round(
            float(lat),
            4
        ),

        round(
            float(lon),
            4
        ),

        int(
            radius_meters
        )
    )


    if cache_key in GRAPH_CACHE:

        cached = (
            GRAPH_CACHE[
                cache_key
            ]
        )


        return (

            cached[
                "graph"
            ],

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


    edges_to_remove = []


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

            edges_to_remove.append(
                (
                    u,
                    v,
                    key
                )
            )


    G.remove_edges_from(
        edges_to_remove
    )


    G.remove_nodes_from(

        list(
            nx.isolates(
                G
            )
        )
    )


    if (

        G.number_of_nodes()
        ==
        0

        or

        G.number_of_edges()
        ==
        0
    ):

        raise HTTPException(

            status_code=400,

            detail=(
                "No usable trail network remained "
                "after filtering."
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


    G = G.subgraph(
        component
    ).copy()


    filtered_edges_removed = (

        original_edges

        -

        G.number_of_edges()
    )


    # No internet elevation API.
    # Read everything from the local USGS TIFF.
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

        oldest_key = (
            next(
                iter(
                    GRAPH_CACHE
                )
            )
        )

        GRAPH_CACHE.pop(
            oldest_key
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
# REPEATED TRAIL METRICS
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
        v = route_nodes[i + 1]


        key = (
            undirected_edge_key(
                u,
                v
            )
        )


        edge = (
            get_shortest_edge(
                G,
                u,
                v
            )
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


        counts[key] = (

            counts.get(
                key,
                0
            )

            +

            1
        )


        lengths[key] = (
            length
        )


    repeated_edges = 0
    repeated_distance = 0.0


    for (
        key,
        count
    ) in counts.items():


        if count > 1:

            repeated_edges += (
                count - 1
            )


            repeated_distance += (

                lengths.get(
                    key,
                    0.0
                )

                *

                (
                    count - 1
                )
            )


    return (

        repeated_edges,

        repeated_distance
    )


def repeated_node_occurrences(
    route_nodes
):

    if len(route_nodes) <= 2:
        return 0


    counts = {}


    # Ignore normal start/end return.
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

    reversals = 0


    for i in range(
        len(route_nodes) - 2
    ):


        # A -> B -> A
        if (
            route_nodes[i]
            ==
            route_nodes[
                i + 2
            ]
        ):

            reversals += 1


    return reversals


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


    distance_error_ratio = (

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

        gain_error_ratio = (

            gain_error

            /

            target_gain_meters
        )


    else:

        gain_error_ratio = (
            0.0
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


    repeated_node_ratio = (

        repeated_nodes

        /

        max(
            1,
            len(route_nodes) - 2
        )
    )


    immediate_reversals = (
        count_immediate_reversals(
            route_nodes
        )
    )


    # Lower score = better route.
    score = (

        distance_error_ratio
        *
        120.0

        +

        gain_error_ratio
        *
        150.0

        +

        repeat_ratio
        *
        450.0

        +

        repeated_node_ratio
        *
        120.0

        +

        immediate_reversals
        *
        40.0
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
# LOOP GENERATOR
# =========================================================

def generate_clean_loop(
    G,
    start_node,
    target_distance_meters,
    target_gain_meters,
    attempts=
        ROUTE_ATTEMPTS
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


    min_anchor_distance = min(

        1300.0,

        max(
            450.0,
            target_distance_meters
            *
            0.06
        )
    )


    anchor_candidates = []


    for node in S.nodes:


        if node == start_node:
            continue


        distance = (
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
            distance
            >=
            min_anchor_distance
        ):

            anchor_candidates.append(
                node
            )


    # Four waypoints works better for longer
    # trail-running loops.
    anchor_count = (

        4

        if target_distance_meters
        >=
        12000

        else

        3
    )


    if (
        len(
            anchor_candidates
        )
        <
        anchor_count
    ):

        raise HTTPException(

            status_code=400,

            detail=(
                "Not enough trail junctions "
                "were found to build a loop."
            )
        )


    best_route = None
    best_metrics = None
    best_score = float("inf")


    for _ in range(
        attempts
    ):


        anchors = random.sample(

            anchor_candidates,

            anchor_count
        )


        too_close = False


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


                if separation < 300.0:

                    too_close = True

                    break


            if too_close:
                break


        if too_close:
            continue


        # Geographic ordering reduces criss-crossing.
        anchors.sort(

            key=lambda node:

                node_angle_from_start(

                    G,

                    start_node,

                    node
                )
        )


        # Random clockwise/counter-clockwise.
        if random.random() < 0.5:

            anchors.reverse()


        route_nodes = [
            start_node
        ]


        used_edges = set()

        current = start_node

        failed = False


        # Start -> waypoint -> waypoint -> ... -> Start
        for destination in (

            anchors

            +

            [start_node]
        ):


            try:

                leg = (
                    penalized_shortest_path(

                        S,

                        current,

                        destination,

                        used_edges
                    )
                )


            except nx.NetworkXNoPath:

                failed = True

                break


            if len(leg) < 2:

                current = destination

                continue


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


            route_nodes.extend(
                leg[1:]
            )


            current = destination


        if (
            failed

            or

            len(route_nodes) < 4
        ):

            continue


        (
            score,
            metrics
        ) = route_score(

            G,

            route_nodes,

            target_distance_meters,

            target_gain_meters
        )


        total_distance = (
            metrics[
                "total_distance_meters"
            ]
        )


        if (
            total_distance
            <
            target_distance_meters
            *
            0.65
        ):

            continue


        if (
            total_distance
            >
            target_distance_meters
            *
            1.35
        ):

            continue


        if score < best_score:

            best_score = score

            best_route = (
                route_nodes
            )

            best_metrics = (
                metrics
            )


        distance_good = (

            metrics[
                "distance_error_meters"
            ]

            <=

            160.9344
        )


        gain_good = (

            target_gain_meters
            <=
            0

            or

            metrics[
                "gain_error_meters"
            ]
            <=
            30.48
        )


        repetition_good = (

            metrics[
                "repeat_ratio"
            ]

            <=

            0.02
        )


        # Excellent result:
        # <0.1 mi distance error
        # <100 ft elevation error
        # almost no repetition.
        if (

            distance_good

            and

            gain_good

            and

            repetition_good

            and

            metrics[
                "immediate_reversals"
            ]
            ==
            0
        ):

            break


    if best_route is None:

        raise HTTPException(

            status_code=400,

            detail=(
                "Could not generate a clean route "
                "near the requested distance "
                "and elevation."
            )
        )


    return (

        best_route,

        best_metrics
    )


# =========================================================
# DEM INFO ENDPOINT
# =========================================================

@app.get("/dem-info")
def dem_info():


    if not os.path.exists(
        DEM_PATH
    ):

        raise HTTPException(

            status_code=404,

            detail=(
                "DEM file not found: "
                +
                DEM_PATH
            )
        )


    with rasterio.open(
        DEM_PATH
    ) as src:


        return {

            "file":
                DEM_PATH,

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

            "nodata":
                src.nodata
        }


# =========================================================
# GENERATE ROUTE API
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
                    "Target distance must "
                    "be greater than 0."
                )
            )


        if (
            request.target_gain_ft
            <
            0
        ):

            raise HTTPException(

                status_code=400,

                detail=(
                    "Elevation gain cannot "
                    "be negative."
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


        (
            G,
            filtered_edges_removed,
            graph_from_cache,
            unique_elevation_samples
        ) = download_trail_graph(

            request.start_lat,

            request.start_lon,

            SEARCH_RADIUS_METERS
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


        # =================================================
        # LOOP
        # =================================================

        if same_point:


            (
                route_nodes,
                metrics
            ) = generate_clean_loop(

                G,

                start_node,

                target_distance_meters,

                target_gain_meters,

                attempts=
                    ROUTE_ATTEMPTS
            )


            route_type = (
                "distance + local-DEM trail loop"
            )


        # =================================================
        # POINT TO POINT
        # =================================================

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

                        weight=
                            "length"
                    )
                )


            except nx.NetworkXNoPath:

                raise HTTPException(

                    status_code=400,

                    detail=(
                        "No connected trail route "
                        "was found."
                    )
                )


            (
                _,
                metrics
            ) = route_score(

                G,

                route_nodes,

                max(

                    path_distance_meters(
                        G,
                        route_nodes
                    ),

                    1.0
                ),

                target_gain_meters
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


        coords = (
            route_coordinates(

                G,

                route_nodes
            )
        )


        repeated_distance_miles = (

            metrics[
                "repeated_distance_meters"
            ]

            /

            METERS_PER_MILE
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
                DEM_PATH,


            "status":
                "Local USGS DEM route generated"
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

    font-family:
        Arial,
        sans-serif;

    background:
        #f5f5f5;
}


#controls {

    padding:
        16px;

    background:
        white;

    border-bottom:
        1px solid #ccc;
}


h2 {

    margin:
        0 0 14px 0;
}


.input-row {

    display:
        flex;

    flex-wrap:
        wrap;

    gap:
        12px;

    margin-bottom:
        12px;
}


.input-group {

    display:
        flex;

    flex-direction:
        column;
}


label {

    font-size:
        13px;

    margin-bottom:
        4px;

    font-weight:
        bold;
}


input {

    width:
        165px;

    padding:
        8px;

    border:
        1px solid #aaa;

    border-radius:
        4px;
}


button {

    padding:
        10px 18px;

    border:
        none;

    border-radius:
        5px;

    cursor:
        pointer;

    background:
        #222;

    color:
        white;

    font-size:
        15px;
}


button:disabled {

    opacity:
        0.5;

    cursor:
        wait;
}


#results {

    margin-top:
        12px;

    line-height:
        1.55;
}


#map {

    height:
        calc(
            100vh - 430px
        );

    min-height:
        500px;

    width:
        100%;
}


.error {

    color:
        #b00020;
}


.success {

    color:
        #166534;
}


.warning {

    color:
        #9a6700;
}


.small {

    font-size:
        12px;

    color:
        #666;
}


@media (
    max-width: 700px
) {

    input {

        width:
            145px;
    }


    #map {

        height:
            65vh;
    }
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
    value="33.5777"
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
    value="-112.0822"
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
    value="33.5777"
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
    value="-112.0822"
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
    value="10"
>

</div>


<div class="input-group">

<label for="gain">
Target elevation gain (ft)
</label>

<input
    id="gain"
    type="number"
    step="100"
    min="0"
    value="2000"
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
            33.586,
            -112.085
        ],
        14
    );


L.tileLayer(

    "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",

    {

        maxZoom:
            19,

        attribution:
            "&copy; OpenStreetMap contributors"
    }

).addTo(
    map
);


let routeLine =
    null;


let startMarker =
    null;


let finishMarker =
    null;


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

        "Building trail graph, reading local USGS elevation, " +
        "and evaluating routes..." +

        "</span>";


    button.disabled =
        true;


    try {


        const response =
            await fetch(

                "/generate-route",

                {

                    method:
                        "POST",

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


        const responseText =
            await response.text();


        if (!responseText) {


            throw new Error(

                "Server returned an empty response. " +
                "Check Render logs."
            );
        }


        let result;


        try {


            result =
                JSON.parse(
                    responseText
                );


        }

        catch {


            throw new Error(

                "Invalid server response: "

                +

                responseText.substring(
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

                    weight:
                        5,

                    opacity:
                        0.9
                }

            ).addTo(
                map
            );


        startMarker =
            L.marker(

                coordinates[0]

            )
            .addTo(
                map
            )
            .bindPopup(
                "Start"
            );


        finishMarker =
            L.marker(

                coordinates[
                    coordinates.length
                    -
                    1
                ]

            )
            .addTo(
                map
            )
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
            "<br>" +


            "<b>Graph cached:</b> " +
            result.graph_from_cache +
            "<br>" +


            "<b>Elevation samples:</b> " +
            result.unique_elevation_samples +
            "<br>" +


            "<b>Elevation sample spacing:</b> ~" +
            result.elevation_sample_spacing_m +
            " m<br><br>" +


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


        button.disabled =
            false;
    }
}


</script>


</body>

</html>
"""
