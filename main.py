from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import osmnx as ox
import networkx as nx

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
    return {"status": "Trail Running Creator API is running"}


@app.post("/generate-route")
def generate_route(request: RouteRequest):

    try:
        search_radius_meters = 3000

        # Download nearby walkable/trail network
        G = ox.graph.graph_from_point(
            (request.start_lat, request.start_lon),
            dist=search_radius_meters,
            network_type="walk",
            simplify=True
        )

        # Find graph nodes closest to requested coordinates
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

        # Find shortest walkable route
        route_nodes = nx.shortest_path(
            G,
            start_node,
            end_node,
            weight="length"
        )

        # Build coordinate list
        route_coordinates = []

        for node in route_nodes:
            route_coordinates.append({
                "lat": G.nodes[node]["y"],
                "lon": G.nodes[node]["x"]
            })

        # Calculate route distance
        route_distance_meters = 0

        for i in range(len(route_nodes) - 1):

            u = route_nodes[i]
            v = route_nodes[i + 1]

            edge_data = G.get_edge_data(u, v)

            shortest_edge = min(
                edge_data.values(),
                key=lambda edge: edge.get("length", float("inf"))
            )

            route_distance_meters += shortest_edge.get("length", 0)

        route_distance_miles = route_distance_meters / 1609.344

        return {
            "requested_distance_miles": request.target_distance_miles,
            "requested_gain_ft": request.target_gain_ft,

            "actual_distance_miles": round(route_distance_miles, 2),

            "route": route_coordinates,

            "route_nodes": len(route_nodes),

            "network_nodes": G.number_of_nodes(),
            "network_edges": G.number_of_edges(),

            "status": "Route generated"
        }

    except nx.NetworkXNoPath:
        raise HTTPException(
            status_code=400,
            detail="No connected trail route was found."
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
