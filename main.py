@app.post("/generate-route")
def generate_route(request: RouteRequest):
    try:
        search_radius_meters = 2000

        G = ox.graph.graph_from_point(
            (request.start_lat, request.start_lon),
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

        return {
            "requested_distance_miles": request.target_distance_miles,
            "requested_gain_ft": request.target_gain_ft,
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
