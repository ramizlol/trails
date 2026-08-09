from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import osmnx as ox

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
        "status": "Trail Running Creator API is running"
    }


@app.post("/generate-route")
def generate_route(request: RouteRequest):

    try:
        # Search radius around the starting point.
        # 8,000 meters = about 5 miles in every direction.
        search_radius_meters = 8000

        # Download the walkable OpenStreetMap network around the start.
        G = ox.graph.graph_from_point(
            (request.start_lat, request.start_lon),
            dist=search_radius_meters,
            network_type="walk",
            simplify=True
        )

        # Find nearest graph nodes to the requested start/end coordinates.
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

        return {
            "requested_distance_miles": request.target_distance_miles,
            "requested_gain_ft": request.target_gain_ft,

            "start": {
                "lat": request.start_lat,
                "lon": request.start_lon
            },

            "end": {
                "lat": request.end_lat,
                "lon": request.end_lon
            },

            "osm_start_node": int(start_node),
            "osm_end_node": int(end_node),

            "network_nodes": G.number_of_nodes(),
            "network_edges": G.number_of_edges(),

            "search_radius_meters": search_radius_meters,

            "status": "Trail network successfully downloaded"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
