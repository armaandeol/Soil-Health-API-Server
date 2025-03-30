from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from typing import List, Dict
import os

# Initialize FastAPI app
app = FastAPI(
    title="Soil Health API",
    description="API for predicting soil health based on NPK values, moisture, location, and crop type",
    version="1.0.0"
)

# Global variables for models
model = None
crop_encoder = None
soil_health_encoder = None

# Define ideal NPK values for different crops
IDEAL_NPK = {
    "Wheat": {"N": 50, "P": 30, "K": 40},
    "Rice": {"N": 60, "P": 35, "K": 45},
    "Maize": {"N": 55, "P": 25, "K": 35},
    "Tomato": {"N": 40, "P": 20, "K": 30},
    "Potato": {"N": 45, "P": 25, "K": 35},
    "Soybean": {"N": 50, "P": 30, "K": 40},
    "Cotton": {"N": 65, "P": 40, "K": 50},
    "Barley": {"N": 50, "P": 30, "K": 40}
}


# Lazy loading for models
def load_models():
    global model, crop_encoder, soil_health_encoder
    try:
        if model is None or crop_encoder is None or soil_health_encoder is None:
            model = joblib.load("soil_health_model.pkl")
            crop_encoder = joblib.load("crop_encoder.pkl")
            soil_health_encoder = joblib.load("soil_health_encoder.pkl")
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        return False


# Request model
class SoilDataInput(BaseModel):
    N: float
    P: float
    K: float
    Moisture: float
    Latitude: float
    Longitude: float
    Crop: str


# Response model
class SoilHealthResponse(BaseModel):
    soil_health: str
    deficiency: str
    deficiency_details: Dict[str, float]


# Function to check NPK deficiency
def check_deficiency(N, P, K, crop):
    if crop in IDEAL_NPK:
        ideal = IDEAL_NPK[crop]
        deficiency = {
            "N": max(0, ideal["N"] - N),
            "P": max(0, ideal["P"] - P),
            "K": max(0, ideal["K"] - K)
        }
        return deficiency
    else:
        return {"N": 0, "P": 0, "K": 0}


# Function to predict soil health
def predict_soil_health(N, P, K, Moisture, Latitude, Longitude, Crop):
    # Load models if not loaded
    if not load_models():
        return "Model Loading Error", None

    # Check if crop is valid
    if Crop not in crop_encoder.classes_:
        return "Invalid Crop Type", None

    try:
        # Encode crop type
        Crop_encoded = crop_encoder.transform([Crop])[0]

        # Create input list
        input_data = [[N, P, K, Moisture, Latitude, Longitude, Crop_encoded]]

        # Predict soil health
        prediction = model.predict(input_data)

        # Decode predicted class
        soil_health = soil_health_encoder.inverse_transform(prediction)[0]

        # Check for nutrient deficiencies
        deficiency = check_deficiency(N, P, K, Crop)
        return soil_health, deficiency
    except Exception as e:
        print(f"Error in prediction: {e}")
        return f"Prediction Error: {str(e)}", None


# API Endpoints
@app.get("/")
def read_root():
    return {"message": "Welcome to the Soil Health API"}


@app.get("/health")
def health_check():
    models_loaded = load_models()
    return {
        "status": "healthy" if models_loaded else "unhealthy",
        "models_loaded": models_loaded,
        "environment": {
            "python_path": os.environ.get('PYTHONPATH', 'Not set'),
            "environment": os.environ.get('VERCEL_ENV', 'Not on Vercel')
        }
    }


@app.post("/predict", response_model=SoilHealthResponse)
def predict(data: SoilDataInput):
    try:
        # Extract data
        N, P, K, Moisture = data.N, data.P, data.K, data.Moisture
        Latitude, Longitude, Crop = data.Latitude, data.Longitude, data.Crop

        # Predict soil health
        prediction, deficiency = predict_soil_health(N, P, K, Moisture, Latitude, Longitude, Crop)

        # Check for model errors
        if prediction.startswith("Model Loading Error") or prediction.startswith("Prediction Error"):
            raise HTTPException(status_code=500, detail=prediction)

        # Create deficiency message
        deficiency_str = ", ".join(
            [f"{nutrient} deficient by {deficit}" for nutrient, deficit in deficiency.items() if deficit > 0])
        deficiency_str = deficiency_str if deficiency_str else "No Deficiency"

        # Return results
        return {
            "soil_health": prediction,
            "deficiency": deficiency_str,
            "deficiency_details": deficiency
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-predict")
def batch_predict(data_list: List[SoilDataInput]):
    results = []
    for item in data_list:
        try:
            # Extract data
            N, P, K, Moisture = item.N, item.P, item.K, item.Moisture
            Latitude, Longitude, Crop = item.Latitude, item.Longitude, item.Crop

            # Predict soil health
            prediction, deficiency = predict_soil_health(N, P, K, Moisture, Latitude, Longitude, Crop)

            # Check for model errors
            if prediction.startswith("Model Loading Error") or prediction.startswith("Prediction Error"):
                results.append({"error": prediction})
                continue

            # Create deficiency message
            deficiency_str = ", ".join(
                [f"{nutrient} deficient by {deficit}" for nutrient, deficit in deficiency.items() if deficit > 0])
            deficiency_str = deficiency_str if deficiency_str else "No Deficiency"

            # Append to results
            results.append({
                "soil_health": prediction,
                "deficiency": deficiency_str,
                "deficiency_details": deficiency,
                "input": {
                    "N": N, "P": P, "K": K, "Moisture": Moisture,
                    "Latitude": Latitude, "Longitude": Longitude, "Crop": Crop
                }
            })
        except Exception as e:
            results.append({"error": str(e)})

    return {"results": results}