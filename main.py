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


def path_distance_meters(G, route_nodes):
    total = 0.0

    for i in range(len(route_nodes) - 1):
        u = route_nodes[i]
        v = route_nodes[i + 1]

        edge_data = G.get_edge_data(u, v)

        if not edge_data:
            continue

        shortest_edge = min(
            edge_data.values(),
            key=lambda edge: edge.get("length", float("inf"))
        )

        total += shortest_edge.get("length", 0)

    return total


def route_coordinates(G, route_nodes):
    return [
        {
            "lat": float(G.nodes[node]["y"]),
            "lon": float(G.nodes[node]["x"])
        }
        for node in route_nodes
    ]


def generate_distance_loop(
    G,
    start_node,
    target_distance_meters,
    attempts=500
):
    best_route = None
    best_error = float("inf")
    best_distance = 0.0

    # We want the outward section to use roughly half
    # the total distance, then find a path back.
    outward_target = target_distance_meters * 0.5

    nodes = list(G.nodes)

    for _ in range(attempts):

        current = start_node
        outward_route = [start_node]
        outward_distance = 0.0

        visited_edges = set()

        # Random outward exploration
        for _step in range(250):

            neighbors = list(G.successors(current))

            if not neighbors:
                break

            candidates = []

            for neighbor in neighbors:

                edge_data = G.get_edge_data(
                    current,
                    neighbor
                )

                if not edge_data:
                    continue

                edge = min(
                    edge_data.values(),
                    key=lambda e: e.get(
                        "length",
                        float("inf")
                    )
                )

                length = edge.get("length", 0)

                edge_key = (
                    min(current, neighbor),
                    max(current, neighbor)
                )

                # Penalize immediately repeating edges
                penalty = (
                    0.2
                    if edge_key in visited_edges
                    else 1.0
                )

                candidates.append(
                    (
                        neighbor,
                        length,
                        edge_key,
                        penalty
                    )
                )

            if not candidates:
                break

            weights = [
                item[3]
                for item in candidates
            ]

            chosen = random.choices(
                candidates,
                weights=weights,
                k=1
            )[0]

            neighbor, length, edge_key, _ = chosen

            outward_route.append(neighbor)
            outward_distance += length
            visited_edges.add(edge_key)

            current = neighbor

            # Once reasonably far out,
            # try connecting back to start.
            if outward_distance >= outward_target * 0.7:

                try:
                    return_route = nx.shortest_path(
                        G,
                        current,
                        start_node,
                        weight="length"
                    )
                except nx.NetworkXNoPath:
                    continue

                return_distance = path_distance_meters(
                    G,
                    return_route
                )

                total_distance = (
                    outward_distance +
                    return_distance
                )

                error = abs(
                    total_distance -
                    target_distance_meters
                )

                if error < best_error:

                    full_route = (
                        outward_route +
                        return_route[1:]
                    )

                    best_route = full_route
                    best_distance = total_distance
                    best_error = error

                # Close enough: stop early
                if error <= 160.9344:
                    return (
                        best_route,
                        best_distance
                    )

            # Don't wander ridiculously far.
            if outward_distance > (
                target_distance_meters * 0.9
            ):
                break

    return best_route, best_distance


@app.post("/generate-route")
def generate_route(request: RouteRequest):

    try:

        target_distance_meters = (
            request.target_distance_miles
            * 1609.344
        )

        # Make search area scale somewhat
        # with requested mileage.
        search_radius_meters = max(
            3000,
            min(
                8000,
                int(
                    target_distance_meters
                    * 0.45
                )
            )
        )

        G = ox.graph.graph_from_point(
            (
                request.start_lat,
                request.start_lon
            ),
            dist=search_radius_meters,
            network_type="walk",
            simplify=True
        )

        start_node = ox.distance.nearest_nodes(
            G,
            X=request.start_lon,
            Y=request.start_lat
        )

        end_node = ox.distance.nearest_nodes(
            G,
            X=request.end_lon,
            Y=request.end_lat
        )

        same_point = (
            abs(
                request.start_lat -
                request.end_lat
            ) < 0.0001
            and
            abs(
                request.start_lon -
                request.end_lon
            ) < 0.0001
        )

        if same_point:

            route_nodes, route_distance_meters = (
                generate_distance_loop(
                    G,
                    start_node,
                    target_distance_meters,
                    attempts=500
                )
            )

            if not route_nodes:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Could not generate a loop "
                        "near the requested distance."
                    )
                )

            route_type = "loop"

        else:

            route_nodes = nx.shortest_path(
                G,
                start_node,
                end_node,
                weight="length"
            )

            route_distance_meters = (
                path_distance_meters(
                    G,
                    route_nodes
                )
            )

            route_type = "point-to-point"

        route_distance_miles = (
            route_distance_meters /
            1609.344
        )

        distance_error_miles = abs(
            route_distance_miles -
            request.target_distance_miles
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

            "route_type":
                route_type,

            "route":
                route_coordinates(
                    G,
                    route_nodes
                ),

            "route_nodes":
                len(route_nodes),

            "network_nodes":
                G.number_of_nodes(),

            "network_edges":
                G.number_of_edges(),

            "search_radius_meters":
                search_radius_meters,

            "status":
                "Route generated"
        }

    except nx.NetworkXNoPath:
        raise HTTPException(
            status_code=400,
            detail=(
                "No connected route was found."
            )
        )

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.get("/map", response_class=HTMLResponse)
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

<title>Trail Running Creator</title>

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
    margin-top: 0;
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
    width: 165px;
    padding: 8px;
}

button {
    padding: 10px 18px;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    background: #222;
    color: white;
}

button:disabled {
    opacity: 0.5;
}

#results {
    margin-top: 12px;
    line-height: 1.5;
}

#map {
    height: calc(100vh - 290px);
    min-height: 500px;
    width: 100%;
}

.error {
    color: #b00020;
}

.success {
    color: #166534;
}

</style>

</head>

<body>

<div id="controls">

<h2>Trail Running Creator</h2>

<div class="input-row">

<div class="input-group">
<label>Start latitude</label>
<input
    id="start_lat"
    type="number"
    step="any"
    value="33.5777"
>
</div>

<div class="input-group">
<label>Start longitude</label>
<input
    id="start_lon"
    type="number"
    step="any"
    value="-112.0822"
>
</div>

<div class="input-group">
<label>End latitude</label>
<input
    id="end_lat"
    type="number"
    step="any"
    value="33.5777"
>
</div>

<div class="input-group">
<label>End longitude</label>
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
<label>Target distance (miles)</label>
<input
    id="distance"
    type="number"
    step="0.1"
    value="10"
>
</div>

<div class="input-group">
<label>Target gain (ft)</label>
<input
    id="gain"
    type="number"
    step="100"
    value="2000"
>
</div>

</div>

<button id="generateButton">
Generate Route
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

const map = L.map("map").setView(
    [33.586, -112.085],
    14
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
        "Generating candidate loops...";

    button.disabled = true;

    try {

        const response = await fetch(
            "/generate-route",
            {
                method: "POST",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body:
                    JSON.stringify(data)
            }
        );

        const result =
            await response.json();

        if (!response.ok) {

            throw new Error(
                result.detail ||
                "Server error"
            );
        }

        const coordinates =
            result.route.map(
                p => [
                    p.lat,
                    p.lon
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

        routeLine = L.polyline(
            coordinates,
            {
                weight: 5,
                opacity: 0.9
            }
        ).addTo(map);

        startMarker = L.marker(
            coordinates[0]
        )
        .addTo(map)
        .bindPopup("Start");

        finishMarker = L.marker(
            coordinates[
                coordinates.length - 1
            ]
        )
        .addTo(map)
        .bindPopup("Finish");

        map.fitBounds(
            routeLine.getBounds(),
            {
                padding: [30, 30]
            }
        );

        results.innerHTML =

            '<span class="success">' +
            "<b>Route generated</b>" +
            "</span><br>" +

            "<b>Type:</b> " +
            result.route_type +
            "<br>" +

            "<b>Target:</b> " +
            result.requested_distance_miles +
            " mi<br>" +

            "<b>Actual:</b> " +
            result.actual_distance_miles +
            " mi<br>" +

            "<b>Difference:</b> " +
            result.distance_error_miles +
            " mi<br>" +

            "<b>Target elevation:</b> " +
            result.requested_gain_ft +
            " ft" +
            " (not active yet)";

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
