from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import osmnx as ox
import networkx as nx
import random

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


def get_shortest_edge(G, u, v):
    edge_data = G.get_edge_data(u, v)

    if not edge_data:
        return None

    return min(
        edge_data.values(),
        key=lambda edge: edge.get("length", float("inf"))
    )


def path_distance_meters(G, route_nodes):
    total = 0.0

    for i in range(len(route_nodes) - 1):
        u = route_nodes[i]
        v = route_nodes[i + 1]

        edge = get_shortest_edge(G, u, v)

        if edge:
            total += float(edge.get("length", 0))

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
    attempts=50
):
    best_route = None
    best_distance = 0.0
    best_error = float("inf")

    outward_target = target_distance_meters * 0.50

    for _ in range(attempts):

        current = start_node
        outward_route = [start_node]
        outward_distance = 0.0

        visited_edges = set()
        recently_visited_nodes = {start_node}

        for _step in range(180):

            neighbors = list(G.successors(current))

            if not neighbors:
                break

            candidates = []

            for neighbor in neighbors:

                edge = get_shortest_edge(
                    G,
                    current,
                    neighbor
                )

                if edge is None:
                    continue

                length = float(
                    edge.get("length", 0)
                )

                if length <= 0:
                    continue

                edge_key = tuple(
                    sorted(
                        (
                            int(current),
                            int(neighbor)
                        )
                    )
                )

                weight = 1.0

                # Strongly discourage reusing the same trail edge.
                if edge_key in visited_edges:
                    weight *= 0.08

                # Discourage bouncing between nearby/recent nodes.
                if neighbor in recently_visited_nodes:
                    weight *= 0.20

                candidates.append(
                    {
                        "neighbor": neighbor,
                        "length": length,
                        "edge_key": edge_key,
                        "weight": weight
                    }
                )

            if not candidates:
                break

            weights = [
                candidate["weight"]
                for candidate in candidates
            ]

            chosen = random.choices(
                candidates,
                weights=weights,
                k=1
            )[0]

            neighbor = chosen["neighbor"]
            length = chosen["length"]
            edge_key = chosen["edge_key"]

            outward_route.append(neighbor)

            outward_distance += length

            visited_edges.add(edge_key)

            recently_visited_nodes.add(neighbor)

            if len(recently_visited_nodes) > 30:
                old_node = outward_route[
                    max(
                        0,
                        len(outward_route) - 31
                    )
                ]

                recently_visited_nodes.discard(
                    old_node
                )

            current = neighbor

            # Once we have traveled a reasonable amount,
            # try finding the shortest way back to start.
            if outward_distance >= outward_target * 0.65:

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

                # Within about 0.15 mile.
                if error <= 241.4:
                    return (
                        best_route,
                        best_distance
                    )

            # Stop wandering if the outward portion
            # is already almost the full requested route.
            if outward_distance > (
                target_distance_meters * 0.90
            ):
                break

    return best_route, best_distance


@app.post("/generate-route")
def generate_route(request: RouteRequest):

    try:

        if request.target_distance_miles <= 0:
            raise HTTPException(
                status_code=400,
                detail="Target distance must be greater than 0."
            )

        target_distance_meters = (
            request.target_distance_miles
            * 1609.344
        )

        # Keep this small for now so Render/Overpass
        # does not have to download a massive network.
        search_radius_meters = 3000

        G = ox.graph.graph_from_point(
            (
                request.start_lat,
                request.start_lon
            ),
            dist=search_radius_meters,
            network_type="walk",
            simplify=True
        )

        if G.number_of_nodes() == 0:
            raise HTTPException(
                status_code=400,
                detail="No walkable network was found near the start point."
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

            (
                route_nodes,
                route_distance_meters
            ) = generate_distance_loop(
                G,
                start_node,
                target_distance_meters,
                attempts=50
            )

            if not route_nodes:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Could not generate a loop near "
                        "the requested distance."
                    )
                )

            route_type = "loop"

        else:

            try:
                route_nodes = nx.shortest_path(
                    G,
                    start_node,
                    end_node,
                    weight="length"
                )

            except nx.NetworkXNoPath:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "No connected route was found "
                        "between the selected points."
                    )
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
    width: 165px;
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
    line-height: 1.5;
}

#map {
    height: calc(100vh - 300px);
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

@media (max-width: 700px) {

    input {
        width: 145px;
    }

    #map {
        height: 65vh;
    }

}

</style>

</head>


<body>


<div id="controls">

<h2>Trail Running Creator</h2>


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
Target gain (ft)
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

const map = L.map(
    "map"
).setView(
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


    if (
        !Number.isFinite(
            data.start_lat
        ) ||
        !Number.isFinite(
            data.start_lon
        ) ||
        !Number.isFinite(
            data.end_lat
        ) ||
        !Number.isFinite(
            data.end_lon
        ) ||
        !Number.isFinite(
            data.target_distance_miles
        )
    ) {

        results.innerHTML =
            '<span class="error">' +
            "Please enter valid numbers." +
            "</span>";

        return;
    }


    results.innerHTML =
        '<span class="warning">' +
        "Downloading trail network and searching candidate loops..." +
        "</span>";


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
                    JSON.stringify(
                        data
                    )
            }
        );


        /*
        Read response as text first.

        This prevents:
        "Unexpected end of JSON input"

        if Render sends an empty response.
        */

        const responseText =
            await response.text();


        if (!responseText) {

            throw new Error(
                "Server returned an empty response. " +
                "Check the Render logs."
            );
        }


        let result;


        try {

            result =
                JSON.parse(
                    responseText
                );

        }

        catch (jsonError) {

            throw new Error(
                "Server returned invalid JSON: " +
                responseText.substring(
                    0,
                    500
                )
            );

        }


        if (!response.ok) {

            throw new Error(
                result.detail ||
                "Server error."
            );

        }


        if (
            !result.route ||
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
        .bindPopup(
            "Start"
        );


        finishMarker = L.marker(
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

            "<b>Type:</b> " +
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

            "<b>Requested elevation:</b> " +
            result.requested_gain_ft +
            " ft " +

            "<span class='warning'>" +
            "(not used yet)" +
            "</span><br>" +

            "<b>Route nodes:</b> " +
            result.route_nodes +
            "<br>" +

            "<b>Graph nodes:</b> " +
            result.network_nodes +
            "<br>" +

            "<b>Graph edges:</b> " +
            result.network_edges;


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
