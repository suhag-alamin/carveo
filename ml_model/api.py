
# FastAPI server for car price prediction
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from car_price_prediction import predict_price, train_model
import os

# Initialize FastAPI app
app = FastAPI(title="Car Price Prediction API")

# Define input data model
class CarData(BaseModel):
    make: str
    model: str
    year: str
    mileage: float
    fuelType: str
    transmission: str

# Define response model
class PredictionResponse(BaseModel):
    predictedPrice: int
    confidenceScore: float

# Train model on startup
@app.on_event("startup")
async def startup_event():
    if not os.path.exists('models/car_price_model.pkl'):
        print("Training model on startup...")
        train_model()

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Car Price Prediction API is running"}

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
def get_prediction(car_data: CarData):
    try:
        result = predict_price(
            make=car_data.make,
            model_name=car_data.model,
            year=car_data.year,
            mileage=car_data.mileage,
            fuel_type=car_data.fuelType,
            transmission=car_data.transmission
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# Run the API with uvicorn
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
