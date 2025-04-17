
# Car Price Prediction ML Model

This directory contains a Python machine learning model for car price prediction.

## Setup

1. Install Python 3.9+ if not already installed
2. Create a virtual environment:
   ```
   python -m venv venv
   ```
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Unix/MacOS: `source venv/bin/activate`
4. Install requirements:
   ```
   pip install -r requirements.txt
   ```

## Running the API

Start the FastAPI server:
```
python api.py
```

The API will be available at http://localhost:8000

## Endpoints

- GET `/`: Health check
- POST `/predict`: Predict car price
- GET `/health`: API health status

## Example Request

```
POST /predict
Content-Type: application/json

{
  "make": "Toyota",
  "model": "Camry",
  "year": "2018",
  "mileage": 50000,
  "fuelType": "Gasoline",
  "transmission": "Automatic"
}
```

## Example Response

```
{
  "predictedPrice": 18500,
  "confidenceScore": 85.7
}
```
