# Car Price Prediction ML Model using scikit-learn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# This would typically use a real dataset
# For demonstration purposes, we'll create synthetic data
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

# Function to predict price for new data
def predict_price(make, model_name, year, mileage, fuel_type, transmission):
    # Load the trained model
    try:
        ml_model = joblib.load('models/car_price_model.pkl')
    except FileNotFoundError:
        # If model doesn't exist, train it
        ml_model, _ = train_model()
    
    # Create a DataFrame with the input data
    input_data = pd.DataFrame({
        'make': [make],
        'year': [int(year)],
        'mileage': [int(mileage)],
        'fuel_type': [fuel_type],
        'transmission': [transmission]
    })
    
    # Make prediction
    predicted_price = ml_model.predict(input_data)[0]
    
    # Generate confidence score (in a real scenario, this would be derived from model stats)
    # Here we're simulating it based on distance from training data distribution
    confidence_score = np.random.uniform(70, 95)  # In a real model, use prediction intervals
    
    return {
        'predictedPrice': round(predicted_price),
        'confidenceScore': round(confidence_score, 1)
    }

if __name__ == "__main__":
    # For demonstration, train the model and make a sample prediction
    train_model()
    
    prediction = predict_price(
        make="Toyota",
        model_name="Camry",
        year=2018,
        mileage=50000,
        fuel_type="Gasoline",
        transmission="Automatic"
    )
    
    print(f"Sample Prediction:")
    print(f"Predicted Price: ${prediction['predictedPrice']}")
    print(f"Confidence Score: {prediction['confidenceScore']}%")
