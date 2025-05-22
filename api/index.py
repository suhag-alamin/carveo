
# FastAPI server for car price prediction
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

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
    featureImportance: list

# Generate synthetic data for model training
def generate_synthetic_data(n_samples=1000):
    np.random.seed(42)
    
    # Generate synthetic data that mimics real car data
    makes = ["Audi", "BMW", "Chevrolet", "Ford", "Honda", "Hyundai", "Kia", 
             "Lexus", "Mazda", "Mercedes-Benz", "Nissan", "Subaru", "Tesla", "Toyota", "Volkswagen"]
    
    # Define price ranges for different makes (premium vs standard)
    premium_makes = ["BMW", "Mercedes-Benz", "Audi", "Lexus", "Tesla"]
    
    # Generate data
    data = {
        'make': np.random.choice(makes, n_samples),
        'year': np.random.randint(2000, 2024, n_samples),
        'mileage': np.random.randint(0, 300000, n_samples),
        'fuel_type': np.random.choice(["Gasoline", "Diesel", "Electric", "Hybrid", "Plug-in Hybrid"], n_samples),
        'transmission': np.random.choice(["Automatic", "Manual", "Semi-Automatic", "CVT"], n_samples),
    }
    
    # Calculate synthetic prices based on factors
    prices = []
    for i in range(n_samples):
        base_price = 20000
        
        # Make adjustment
        if data['make'][i] in premium_makes:
            base_price *= 1.5
        
        # Year adjustment (newer cars cost more)
        year_factor = 1 - ((2023 - data['year'][i]) * 0.05)
        base_price *= max(year_factor, 0.3)
        
        # Mileage adjustment
        mileage_factor = 1 - (data['mileage'][i] / 300000)
        base_price *= max(mileage_factor, 0.4)
        
        # Fuel type adjustment
        if data['fuel_type'][i] == "Electric":
            base_price *= 1.25
        elif data['fuel_type'][i] in ["Hybrid", "Plug-in Hybrid"]:
            base_price *= 1.15
            
        # Transmission adjustment
        if data['transmission'][i] == "Automatic":
            base_price *= 1.05
            
        # Add noise
        noise = np.random.normal(1, 0.1)
        base_price *= noise
        
        prices.append(round(base_price))
    
    data['price'] = prices
    return pd.DataFrame(data)

# Train the model
def train_model():
    # Generate or load real data
    df = generate_synthetic_data(5000)
    
    # Define features and target
    X = df.drop('price', axis=1)
    y = df['price']
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define categorical features for one-hot encoding
    categorical_features = ['make', 'fuel_type', 'transmission']
    numeric_features = ['year', 'mileage']
    
    # Create preprocessing pipelines
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features),
            ('num', numeric_transformer, numeric_features)
        ])
    
    # Create pipeline with preprocessor and model
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Evaluation:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared: {r2:.2f}")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/car_price_model.pkl')
    
    return model, r2

# Function to calculate feature importance
def get_feature_importance(model, make, fuel_type, transmission, year, mileage):
    # This function calculates dynamic feature importance based on the input values
    
    # Start with base importance values (these will be adjusted based on inputs)
    importances = {}
    
    # Convert year to int for calculations
    year_int = int(year)
    current_year = 2023
    age = current_year - year_int
    
    # Premium makes have different importance profiles
    premium_makes = ["BMW", "Mercedes-Benz", "Audi", "Lexus", "Tesla"]
    mid_tier_makes = ["Toyota", "Honda", "Volkswagen", "Subaru", "Mazda"]
    
    # Dynamic calculation of feature importance:
    
    # 1. Year importance - higher for newer cars, lower for older ones
    year_importance = 30 - min(age * 0.7, 15)  # Decreases with age, but has a floor
    
    # 2. Mileage importance - increases with higher mileage
    mileage_ratio = min(mileage / 150000, 1)  # Cap at 150k miles for calculation
    mileage_importance = 15 + (mileage_ratio * 15)  # 15-30% importance based on mileage
    
    # 3. Make importance - premium brands have higher make importance
    if make in premium_makes:
        make_importance = 30  # Premium brands have high make importance
    elif make in mid_tier_makes:
        make_importance = 22  # Mid-tier brands have medium make importance
    else:
        make_importance = 15  # Budget brands have lower make importance
    
    # 4. Fuel type importance - higher for alternative fuels
    if fuel_type == "Electric":
        fuel_importance = 25  # Electric vehicles have high fuel type importance
    elif fuel_type in ["Hybrid", "Plug-in Hybrid"]:
        fuel_importance = 18  # Hybrids have medium-high fuel type importance
    else:
        fuel_importance = 10  # Conventional fuels have lower importance
    
    # 5. Transmission importance - varies by type
    if transmission == "Automatic":
        transmission_importance = 12
    elif transmission == "CVT":
        transmission_importance = 15
    else:
        transmission_importance = 8
    
    # Collect all importances
    raw_importances = [
        {"name": "Year", "importance": year_importance},
        {"name": "Mileage", "importance": mileage_importance},
        {"name": "Make", "importance": make_importance},
        {"name": "Fuel Type", "importance": fuel_importance},
        {"name": "Transmission", "importance": transmission_importance}
    ]
    
    # Make sure no feature has less than 5% importance
    for item in raw_importances:
        if item["importance"] < 5:
            item["importance"] = 5
            
    # Normalize to ensure percentages add up to 100%
    total = sum(item["importance"] for item in raw_importances)
    
    normalized_importances = []
    for item in raw_importances:
        normalized_importance = round((item["importance"] / total) * 100)
        normalized_importances.append({
            "name": item["name"],
            "importance": normalized_importance
        })
    
    # Sort by importance (descending)
    normalized_importances.sort(key=lambda x: x["importance"], reverse=True)
    
    # Ensure the sum is exactly 100% (adjust the smallest value if needed)
    total = sum(item["importance"] for item in normalized_importances)
    if total != 100:
        # Find the smallest importance
        normalized_importances.sort(key=lambda x: x["importance"])
        normalized_importances[0]["importance"] += (100 - total)
        # Sort back by importance (descending)
        normalized_importances.sort(key=lambda x: x["importance"], reverse=True)
    
    return normalized_importances

# Global variable to hold the ML model
ml_model = None

# Load or train the model on startup
@app.on_event("startup")
async def startup_event():
    global ml_model
    model_path = 'models/car_price_model.pkl'
    
    if os.path.exists(model_path):
        print("Loading existing model...")
        ml_model = joblib.load(model_path)
    else:
        print("Training new model...")
        ml_model, _ = train_model()

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Car Price Prediction API is running"}

# Prediction endpoint
@app.post("/predict")
async def get_prediction(car_data: CarData):
    global ml_model
    
    try:
        # If model isn't loaded yet (cold start), train it
        if ml_model is None:
            ml_model, _ = train_model()
            
        # Create a DataFrame with the input data
        input_data = pd.DataFrame({
            'make': [car_data.make],
            'year': [int(car_data.year)],
            'mileage': [car_data.mileage],
            'fuel_type': [car_data.fuelType],
            'transmission': [car_data.transmission]
        })
        
        # Make prediction
        predicted_price = ml_model.predict(input_data)[0]
        
        # Generate confidence score 
        # In a real application, this would be based on prediction intervals
        # Here we're simulating it based on the input data
        hash_val = hash(f"{car_data.make}{car_data.model}{car_data.year}")
        confidence_score = 70 + (hash_val % 1000) / 40  # Range between 70-95
        confidence_score = min(max(confidence_score, 70), 95)  # Clamp between 70-95
        
        # Get feature importance - use our dynamic calculation function
        feature_importance = get_feature_importance(
            ml_model,
            car_data.make,
            car_data.fuelType,
            car_data.transmission,
            car_data.year,
            car_data.mileage
        )
        
        print(f"Feature importance calculated: {feature_importance}")
        
        return {
            "predictedPrice": round(predicted_price),
            "confidenceScore": round(confidence_score, 1),
            "featureImportance": feature_importance
        }
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# Run the API with uvicorn when executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("index:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=True)
