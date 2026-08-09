from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import math
import random
import requests
import networkx as nx
import osmnx as ox


app = FastAPI()


# =========================================================
# SETTINGS / CACHE
# =========================================================

GRAPH_CACHE = {}
MAX_CACHED_GRAPHS = 4

# Cache individual elevation samples too.
ELEVATION_POINT_CACHE = {}

METERS_PER_MILE = 1609.344
FEET_PER_METER = 3.28084

# We sample the actual trail geometry at approximately
# this spacing before calculating climb/descent.
ELEVATION_SAMPLE_SPACING_M = 60.0


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
        "docs": "/docs"
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
            dphi / 2
        ) ** 2

        +

        math.cos(p1)
        *
        math.cos(p2)
        *
        math.sin(
            dlambda / 2
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
    """
    Sum the pre-calculated ascent of every
    actual trail edge in the route.
    """

    total_gain = 0.0

    for i in range(
        len(route_nodes) - 1
    ):

        edge = get_shortest_edge(
            G,
            route_nodes[i],
            route_nodes[i + 1]
        )

        if edge is not None:

            total_gain += float(
                edge.get(
                    "ascent_m",
                    0
                )
                or
                0
            )

    return total_gain


# =========================================================
# ROUTING GRAPH
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
                1
            )
        )

        edge_key = (
            undirected_edge_key(
                u,
                v
            )
        )

        # Make previously-used trail sections
        # very expensive.
        if edge_key in used_edges:

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
# FULL OSM EDGE GEOMETRY
# =========================================================

def oriented_edge_coords(
    G,
    u,
    v,
    edge
):
    """
    Get the actual OSM trail geometry and orient
    it in the direction u -> v.

    Returns:
        [(longitude, latitude), ...]
    """

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

    # Reverse the LineString if OSM stored
    # its coordinates in the other direction.
    if last_distance < first_distance:

        coords.reverse()

    return coords


def route_coordinates(
    G,
    route_nodes
):
    """
    Return full curved trail geometry for Leaflet.
    """

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
                "lat": float(lat),
                "lon": float(lon)
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


    # Trail-ish OSM highway values only.
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
    """
    Add intermediate points along the real trail.

    This means a long OSM edge does not only get
    elevation values at its beginning and end.
    """

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
# ELEVATION API
# =========================================================

def elevation_cache_key(
    lat,
    lon
):

    return (
        round(
            float(lat),
            6
        ),

        round(
            float(lon),
            6
        )
    )


def fetch_elevations_for_points(
    points
):
    """
    points:
        [(lon, lat), ...]

    Returns:
        {
            (lat, lon): elevation_meters
        }
    """

    missing = []
    already_added = set()

    for lon, lat in points:

        key = (
            elevation_cache_key(
                lat,
                lon
            )
        )

        if (
            key
            not in
            ELEVATION_POINT_CACHE

            and

            key
            not in
            already_added
        ):

            missing.append(
                (
                    lon,
                    lat,
                    key
                )
            )

            already_added.add(
                key
            )


    # Open-Meteo currently supports
    # up to 100 coordinates per request.
    batch_size = 100


    for start in range(
        0,
        len(missing),
        batch_size
    ):

        batch = missing[
            start:
            start + batch_size
        ]

        latitudes = ",".join(
            str(lat)

            for _,
            lat,
            _
            in batch
        )

        longitudes = ",".join(
            str(lon)

            for lon,
            _,
            _
            in batch
        )

        try:

            response = requests.get(

                "https://api.open-meteo.com/v1/elevation",

                params={
                    "latitude":
                        latitudes,

                    "longitude":
                        longitudes
                },

                timeout=30
            )

        except requests.RequestException as exc:

            raise HTTPException(
                status_code=502,
                detail=(
                    "Elevation service "
                    "connection failed: "
                    +
                    str(exc)
                )
            )


        if not response.ok:

            raise HTTPException(
                status_code=502,
                detail=(
                    "Elevation service returned HTTP "
                    +
                    str(
                        response.status_code
                    )
                )
            )


        try:

            payload = (
                response.json()
            )

        except ValueError:

            raise HTTPException(
                status_code=502,
                detail=(
                    "Elevation service "
                    "returned invalid JSON."
                )
            )


        elevations = (
            payload.get(
                "elevation"
            )
        )


        if (
            not isinstance(
                elevations,
                list
            )

            or

            len(elevations)
            !=
            len(batch)
        ):

            raise HTTPException(
                status_code=502,
                detail=(
                    "Elevation service returned "
                    "unexpected data."
                )
            )


        for (
            (
                _,
                _,
                key
            ),
            elevation
        ) in zip(
            batch,
            elevations
        ):

            if elevation is None:

                raise HTTPException(
                    status_code=502,
                    detail=(
                        "Elevation service returned "
                        "a missing elevation value."
                    )
                )

            ELEVATION_POINT_CACHE[
                key
            ] = float(
                elevation
            )


    result = {}

    for lon, lat in points:

        key = (
            elevation_cache_key(
                lat,
                lon
            )
        )

        result[key] = (
            ELEVATION_POINT_CACHE[
                key
            ]
        )

    return result


# =========================================================
# CALCULATE ELEVATION FOR EVERY TRAIL EDGE
# =========================================================

def add_dense_edge_elevations(G):
    """
    For every trail edge:

    1. get its real OSM geometry
    2. sample points every ~60 m
    3. fetch terrain elevation
    4. calculate ascent
    5. calculate descent

    ascent_m/descent_m are then stored directly
    on the graph edge.
    """

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
        ] = samples


        all_points.extend(
            samples
        )


    elevation_lookup = (
        fetch_elevations_for_points(
            all_points
        )
    )


    # Add elevation to graph nodes too.
    for node in G.nodes:

        lon = float(
            G.nodes[node]["x"]
        )

        lat = float(
            G.nodes[node]["y"]
        )

        cache_key = (
            elevation_cache_key(
                lat,
                lon
            )
        )


        if (
            cache_key
            not in
            ELEVATION_POINT_CACHE
        ):

            fetch_elevations_for_points(
                [
                    (
                        lon,
                        lat
                    )
                ]
            )


        G.nodes[node][
            "elevation"
        ] = float(
            ELEVATION_POINT_CACHE[
                cache_key
            ]
        )


    # Calculate ascent/descent
    # along every individual trail edge.
    for (
        (
            u,
            v,
            key
        ),
        samples
    ) in edge_samples.items():


        elevations = [

            elevation_lookup[
                elevation_cache_key(
                    lat,
                    lon
                )
            ]

            for lon, lat
            in samples
        ]


        ascent = 0.0
        descent = 0.0


        for i in range(
            len(elevations) - 1
        ):

            difference = (
                elevations[i + 1]
                -
                elevations[i]
            )


            if difference > 0:

                ascent += difference


            elif difference < 0:

                descent += (
                    -difference
                )


        edge = (
            G[u][v][key]
        )


        edge[
            "ascent_m"
        ] = float(
            ascent
        )


        edge[
            "descent_m"
        ] = float(
            descent
        )


        edge[
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
# DOWNLOAD TRAIL GRAPH
# =========================================================

def download_trail_graph(
    lat,
    lon,
    radius_meters=3000
):

    cache_key = (

        round(
            lat,
            4
        ),

        round(
            lon,
            4
        ),

        radius_meters
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

        simplify=True,

        retain_all=True
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
                "No usable trail network "
                "remained after filtering."
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


    # THIS IS THE IMPORTANT NEW STEP.
    G, unique_elevation_samples = (
        add_dense_edge_elevations(
            G
        )
    )


    # Limit graph memory usage.
    if (
        len(
            GRAPH_CACHE
        )
        >=
        MAX_CACHED_GRAPHS
    ):

        oldest_key = next(
            iter(
                GRAPH_CACHE
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


        if edge:

            length = float(
                edge.get(
                    "length",
                    0
                )
                or
                0
            )

        else:

            length = 0.0


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
                    0
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
            route_nodes[i + 2]
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


    # Lower score is better.
    #
    # Distance:          120
    # Elevation:         150
    # Repeated mileage:  450
    # Repeated nodes:    120
    # U-turns:            40

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
    attempts=400
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


        if distance >= min_anchor_distance:

            anchor_candidates.append(
                node
            )


    # More waypoints for longer routes.
    if target_distance_meters >= 12000:

        anchor_count = 4

    else:

        anchor_count = 3


    if (
        len(anchor_candidates)
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


        # Reject heavily clustered waypoints.
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


        # Put waypoints in geographic order
        # around the trailhead.
        anchors.sort(

            key=lambda node:

                node_angle_from_start(

                    G,

                    start_node,

                    node
                )
        )


        # Try clockwise and counterclockwise.
        if random.random() < 0.5:

            anchors.reverse()


        route_nodes = [
            start_node
        ]


        used_edges = set()

        current = start_node

        failed = False


        # Start -> waypoints -> Start.
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

                        leg[i + 1]
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


        # Excellent candidate:
        # within 0.10 mi and 100 ft.
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
                "Could not generate a clean "
                "route near the requested "
                "distance and elevation."
            )
        )


    return (
        best_route,
        best_metrics
    )


# =========================================================
# API
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


        if request.target_gain_ft < 0:

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


        search_radius_meters = (
            3000
        )


        (
            G,
            filtered_edges_removed,
            graph_from_cache,
            unique_elevation_samples
        ) = download_trail_graph(

            request.start_lat,

            request.start_lon,

            search_radius_meters
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


        # -------------------------------------------------
        # LOOP
        # -------------------------------------------------

        if same_point:


            (
                route_nodes,
                metrics
            ) = generate_clean_loop(

                G,

                start_node,

                target_distance_meters,

                target_gain_meters,

                attempts=400
            )


            route_type = (
                "distance + dense-elevation trail loop"
            )


        # -------------------------------------------------
        # POINT TO POINT
        # -------------------------------------------------

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
                        "No connected trail "
                        "route was found."
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
                (
                    "Open-Meteo / "
                    "Copernicus DEM GLO-90"
                ),


            "elevation_resolution":
                "90 meters",


            "status":
                (
                    "Dense-geometry "
                    "elevation route generated"
                )
        }


    except HTTPException:

        raise


    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )


# =========================================================
# MAP PAGE
# =========================================================

@app.get(
    "/map",
    response_class=HTMLResponse
)
def route_map():

    return """
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


const map = L.map(
    "map"
).setView(
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

        "Building trail graph, sampling elevation " +
        "along trail geometry, and evaluating routes..." +

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


        } catch {


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


            "<b>Unique elevation samples:</b> " +
            result.unique_elevation_samples +
            "<br>" +


            "<b>Elevation sampling:</b> ~" +
            result.elevation_sample_spacing_m +
            " m along trail geometry" +
            "<br><br>" +


            '<span class="small">' +

            "Elevation source: " +
            result.elevation_source +
            " (" +
            result.elevation_resolution +
            "). Terrain data © Copernicus; " +
            "elevation API by Open-Meteo." +

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
