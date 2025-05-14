
# Vercel Python serverless function for car price prediction
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import sys
import pickle
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

# Add ml_model directory to path so we can import from it
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use a simpler model for Vercel due to size constraints
class SimplifiedModel:
    def __init__(self):
        # Premium makes command higher prices
        self.premium_makes = ["BMW", "Mercedes-Benz", "Audi", "Lexus", "Tesla"]
        self.mid_makes = ["Toyota", "Honda", "Mazda", "Subaru", "Volkswagen"]
        
    def predict(self, X):
        # Extract features from DataFrame
        results = []
        for _, row in X.iterrows():
            make = row['make']
            year = int(row['year'])
            mileage = float(row['mileage'])
            fuel_type = row['fuel_type']
            transmission = row['transmission']
            
            # Base price calculation
            base_price = 20000
            
            # Make adjustment
            if make in self.premium_makes:
                base_price *= 1.5
            elif make in self.mid_makes:
                base_price *= 1.2
                
            # Year adjustment
            current_year = 2025
            year_factor = 1 - ((current_year - year) * 0.05)
            base_price *= max(year_factor, 0.3)
            
            # Mileage adjustment
            mileage_factor = 1 - (mileage / 300000)
            base_price *= max(mileage_factor, 0.4)
            
            # Fuel type adjustment
            if fuel_type == "Electric":
                base_price *= 1.25
            elif fuel_type in ["Hybrid", "Plug-in Hybrid"]:
                base_price *= 1.15
                
            # Transmission adjustment
            if transmission == "Automatic":
                base_price *= 1.05
                
            # Add some consistent variation
            noise = 1 + (hash(f"{make}{year}{mileage}") % 100) / 500  # ±10% deterministic variation
            base_price *= noise
            
            results.append(round(base_price))
            
        return np.array(results)

# Initialize the model
model = SimplifiedModel()

# Initialize FastAPI app
app = FastAPI(title="Car Price Prediction API")

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this with your frontend domain for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Car Price Prediction API is running"}

# Prediction endpoint
@app.post("/predict")
async def get_prediction(car_data: CarData):
    try:
        input_data = pd.DataFrame({
            'make': [car_data.make],
            'year': [car_data.year],
            'mileage': [car_data.mileage],
            'fuel_type': [car_data.fuelType],
            'transmission': [car_data.transmission]
        })
        
        # Make prediction with simplified model
        predicted_price = model.predict(input_data)[0]
        
        # Generate confidence score (in a real scenario, this would be derived from model stats)
        # Here we're simulating it based on the car data
        hash_val = hash(f"{car_data.make}{car_data.model}{car_data.year}")
        confidence_score = 70 + (hash_val % 1000) / 40  # Range between 70-95
        confidence_score = min(max(confidence_score, 70), 95)  # Clamp between 70-95
        
        return {
            "predictedPrice": round(predicted_price),
            "confidenceScore": round(confidence_score, 1)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# This is for Vercel serverless function
def handler(request: Request):
    return app
