from fastapi import FastAPI
from pydantic import BaseModel

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
    # Temporary test response
    return {
        "requested_distance": request.target_distance_miles,
        "requested_gain": request.target_gain_ft,
        "route": []
    }
