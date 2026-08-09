from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import osmnx as ox
import networkx as nx
import random
import math


app = FastAPI()


class RouteRequest(BaseModel):
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float
    target_distance_miles: float
    target_gain_ft: float


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
    """
    Convert an OSM tag value/list into
    a lowercase set of strings.
    """

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

        # Some OSM tags can contain
        # semicolon-separated values.
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

    # Used only to geographically order
    # waypoints around the trailhead.
    x = (
        (lon - start_lon)
        *
        math.cos(
            math.radians(
                start_lat
            )
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
# GRAPH HELPERS
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
            route_nodes[i + 1]
        )

        if edge is not None:

            total += float(
                edge.get(
                    "length",
                    0
                )
                or 0
            )

    return total


def make_simple_routing_graph(G):
    """
    Turn the OSMnx MultiDiGraph into
    a simpler DiGraph for routing.

    If multiple OSM edges connect the
    same nodes, keep the shortest one.
    """

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
            or 0
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
    """
    Shortest path but make already-used
    trail segments very expensive.

    This encourages a loop instead of
    simply going out and back.
    """

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

        key = (
            undirected_edge_key(
                u,
                v
            )
        )

        if key in used_edges:

            return (
                base
                *
                35.0
            )

        return base


    return nx.shortest_path(
        S,
        source,
        target,
        weight=weight
    )


# =========================================================
# FULL TRAIL GEOMETRY
# =========================================================

def route_coordinates(
    G,
    route_nodes
):
    """
    Return actual OSM trail geometry,
    including curves/switchbacks.
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

        geometry = edge.get(
            "geometry"
        )

        if geometry is not None:

            edge_coords = list(
                geometry.coords
            )

            u_lon = float(
                G.nodes[u]["x"]
            )

            u_lat = float(
                G.nodes[u]["y"]
            )

            first_lon, first_lat = (
                edge_coords[0]
            )

            last_lon, last_lat = (
                edge_coords[-1]
            )

            first_distance = (
                abs(
                    first_lon
                    -
                    u_lon
                )
                +
                abs(
                    first_lat
                    -
                    u_lat
                )
            )

            last_distance = (
                abs(
                    last_lon
                    -
                    u_lon
                )
                +
                abs(
                    last_lat
                    -
                    u_lat
                )
            )

            # Geometry might be stored
            # backwards compared with our
            # route direction.
            if (
                last_distance
                <
                first_distance
            ):

                edge_coords.reverse()


            for (
                lon,
                lat
            ) in edge_coords:

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

        else:

            u_point = {
                "lat": float(
                    G.nodes[u]["y"]
                ),
                "lon": float(
                    G.nodes[u]["x"]
                )
            }

            v_point = {
                "lat": float(
                    G.nodes[v]["y"]
                ),
                "lon": float(
                    G.nodes[v]["x"]
                )
            }

            if not coordinates:

                coordinates.append(
                    u_point
                )

            else:

                previous = (
                    coordinates[-1]
                )

                if not (
                    abs(
                        previous["lat"]
                        -
                        u_point["lat"]
                    )
                    <
                    0.0000001
                    and
                    abs(
                        previous["lon"]
                        -
                        u_point["lon"]
                    )
                    <
                    0.0000001
                ):

                    coordinates.append(
                        u_point
                    )

            coordinates.append(
                v_point
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
                "lat": float(
                    G.nodes[node]["y"]
                ),
                "lon": float(
                    G.nodes[node]["x"]
                )
            }
        )

    return coordinates


# =========================================================
# STRICT TRAIL FILTERING
# =========================================================

def edge_is_allowed_trail(data):
    """
    Keep trail-like paths/tracks/steps.

    Reject clearly paved, private,
    indoor or urban pedestrian ways.
    """

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


    allowed_highways = {
        "path",
        "track",
        "steps"
    }


    if not highways.intersection(
        allowed_highways
    ):

        return False


    # Explicit sidewalk/crosswalk
    if footway.intersection(
        {
            "sidewalk",
            "crossing"
        }
    ):

        return False


    # Clearly hard urban surfaces
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


    # Do not route through mapped
    # pedestrian areas.
    if "yes" in area:

        return False


    # No indoor paths
    if "yes" in indoor:

        return False


    # Respect private/closed ways unless
    # pedestrian access is specifically allowed.
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


def download_trail_graph(
    lat,
    lon,
    radius_meters=3000
):

    # Ask OSMnx to preserve tags
    # that we need for trail filtering.

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


    # For now:
    #
    # path
    # track
    # steps
    #
    # We intentionally leave out
    # highway=footway.

    trail_filter = (
        '["highway"~"path|track|steps"]'
    )


    G = (
        ox.graph.graph_from_point(
            (
                lat,
                lon
            ),
            dist=radius_meters,

            # This makes pedestrian networks
            # bidirectional where appropriate.
            network_type="walk",

            custom_filter=
                trail_filter,

            simplify=True,

            retain_all=True
        )
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


    # Remove nodes that now have
    # no usable trail attached.

    isolated_nodes = list(
        nx.isolates(
            G
        )
    )

    G.remove_nodes_from(
        isolated_nodes
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
                "after filtering paved/private "
                "pedestrian ways."
            )
        )


    # Find trail component nearest
    # the requested trailhead.

    nearest = (
        ox.distance.nearest_nodes(
            G,
            X=lon,
            Y=lat
        )
    )


    weak_component = (
        nx.node_connected_component(
            G.to_undirected(
                as_view=True
            ),
            nearest
        )
    )


    G = G.subgraph(
        weak_component
    ).copy()


    removed_edges = (
        original_edges
        -
        G.number_of_edges()
    )


    return (
        G,
        removed_edges
    )


# =========================================================
# ROUTE QUALITY METRICS
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


        edge = get_shortest_edge(
            G,
            u,
            v
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
                    0.0
                )
                *
                (count - 1)
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


    # Don't penalize the expected
    # return to the starting node.
    interior = (
        route_nodes[1:-1]
    )


    counts = {}


    for node in interior:

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


def route_score(
    G,
    route_nodes,
    target_distance_meters
):

    total_distance = (
        path_distance_meters(
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


    # Lower is better.
    #
    # Repetition is intentionally
    # expensive.

    score = (

        distance_error_ratio
        *
        100.0

        +

        repeat_ratio
        *
        350.0

        +

        repeated_node_ratio
        *
        100.0

        +

        immediate_reversals
        *
        30.0
    )


    metrics = {

        "total_distance_meters":
            total_distance,

        "distance_error_meters":
            distance_error,

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


    return (
        score,
        metrics
    )


# =========================================================
# CLEAN LOOP GENERATOR
# =========================================================

def generate_clean_loop(
    G,
    start_node,
    target_distance_meters,
    attempts=180
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


    # Do not choose loop waypoints
    # directly beside the trailhead.

    min_anchor_distance = min(

        1200.0,

        max(
            450.0,
            target_distance_meters
            *
            0.07
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


    if (
        len(anchor_candidates)
        <
        3
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Not enough connected trail "
                "junctions were found to build "
                "a clean loop."
            )
        )


    best_route = None

    best_metrics = None

    best_score = float(
        "inf"
    )


    for _ in range(
        attempts
    ):


        # Pick three geographic waypoints.

        anchors = random.sample(
            anchor_candidates,
            3
        )


        # Don't use three waypoints
        # sitting right beside one another.

        pair_distances = []


        for i in range(3):

            for j in range(
                i + 1,
                3
            ):

                a = anchors[i]
                b = anchors[j]


                pair_distances.append(

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


        if min(
            pair_distances
        ) < 350.0:

            continue


        # Sort geographically around the
        # starting point.
        #
        # This greatly reduces crisscrossing
        # and weird star-shaped routes.

        anchors.sort(

            key=lambda node:

                node_angle_from_start(

                    G,

                    start_node,

                    node
                )
        )


        # Randomly choose clockwise
        # or counterclockwise.

        if random.random() < 0.5:

            anchors.reverse()


        route_nodes = [
            start_node
        ]


        used_edges = set()


        current = (
            start_node
        )


        failed = False


        # Start -> A -> B -> C -> Start

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


            # Mark these segments used
            # before creating the next leg.

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


            current = (
                destination
            )


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

            target_distance_meters
        )


        total_distance = (
            metrics[
                "total_distance_meters"
            ]
        )


        # Candidate is simply
        # too short.

        if (
            total_distance
            <
            target_distance_meters
            *
            0.65
        ):

            continue


        # Candidate is simply
        # too long.

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
        #
        # within ~0.15 mile
        # less than 2% repeated mileage
        # no immediate U-turns

        if (

            metrics[
                "distance_error_meters"
            ]
            <=
            241.4

            and

            metrics[
                "repeat_ratio"
            ]
            <=
            0.02

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
                "Could not build a clean trail "
                "loop near the requested distance. "
                "Try a slightly shorter distance "
                "or a larger search area."
            )
        )


    return (
        best_route,
        best_metrics
    )


# =========================================================
# ROUTE API
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


        target_distance_meters = (

            request.target_distance_miles
            *
            1609.344
        )


        # Keep fixed while we're developing.

        search_radius_meters = (
            3000
        )


        (
            G,
            filtered_edges_removed
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

                attempts=180
            )


            route_distance_meters = (
                metrics[
                    "total_distance_meters"
                ]
            )


            route_type = (
                "clean trail loop"
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
                        "No connected trail-only "
                        "route was found between "
                        "those locations."
                    )
                )


            route_distance_meters = (
                path_distance_meters(

                    G,

                    route_nodes
                )
            )


            (
                _,
                metrics
            ) = route_score(

                G,

                route_nodes,

                max(
                    route_distance_meters,
                    1.0
                )
            )


            route_type = (
                "trail point-to-point"
            )


        route_distance_miles = (

            route_distance_meters
            /
            1609.344
        )


        distance_error_miles = abs(

            route_distance_miles
            -
            request.target_distance_miles
        )


        coords = (
            route_coordinates(

                G,

                route_nodes
            )
        )


        repeated_distance_miles = (

            metrics.get(
                "repeated_distance_meters",
                0.0
            )

            /

            1609.344
        )


        return {

            "requested_distance_miles":
                request.target_distance_miles,

            "requested_gain_ft":
                request.target_gain_ft,

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

            # ELEVATION COMES NEXT
            "actual_gain_ft":
                None,

            "elevation_status":
                (
                    "Elevation matching "
                    "is the next step."
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
                metrics.get(
                    "repeated_edges",
                    0
                ),

            "repeated_distance_miles":
                round(
                    repeated_distance_miles,
                    2
                ),

            "repeated_nodes":
                metrics.get(
                    "repeated_nodes",
                    0
                ),

            "immediate_reversals":
                metrics.get(
                    "immediate_reversals",
                    0
                ),

            "route_score":
                round(
                    metrics.get(
                        "score",
                        0.0
                    ),
                    2
                ),

            "network_nodes":
                G.number_of_nodes(),

            "network_edges":
                G.number_of_edges(),

            "filtered_edges_removed":
                filtered_edges_removed,

            "search_radius_meters":
                search_radius_meters,

            "trail_policy":
                (
                    "OSM highway=path|track|steps; "
                    "excludes clearly paved, "
                    "private/closed, indoor, area, "
                    "sidewalk/crossing segments"
                ),

            "status":
                (
                    "Clean trail route generated"
                )
        }


    except HTTPException:

        raise


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
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

    padding: 16px;

    background:
        white;

    border-bottom:
        1px solid #ccc;
}

h2 {

    margin:
        0
        0
        14px
        0;
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
            100vh - 350px
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

        "Filtering to trails and searching " +
        "for a clean loop..." +

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

                "Server returned invalid JSON: "
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


        let elevationText =
            "Not calculated yet";


        if (
            result.actual_gain_ft
            !==
            null
        ) {


            elevationText =
                result.actual_gain_ft
                +
                " ft";
        }


        results.innerHTML =


            '<span class="success">' +

            "<b>Trail route generated</b>" +

            "</span><br>" +


            "<b>Route type:</b> " +
            result.route_type +
            "<br>" +


            "<b>Requested distance:</b> " +
            result.requested_distance_miles +
            " mi<br>" +


            "<b>Actual distance:</b> " +
            result.actual_distance_miles +
            " mi<br>" +


            "<b>Distance error:</b> " +
            result.distance_error_miles +
            " mi<br>" +


            "<b>Requested gain:</b> " +
            result.requested_gain_ft +
            " ft<br>" +


            "<b>Actual gain:</b> " +
            elevationText +
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


            "<b>Filtered edges removed:</b> " +
            result.filtered_edges_removed +
            "<br>" +


            "<b>Route score:</b> " +
            result.route_score;


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
