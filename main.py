from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from typing import List, Dict, Optional, Union
import logging
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Soil Health API",
    description="API for predicting soil health based on NPK values, moisture, location, and crop type",
    version="1.0.0"
)

# Load models
try:
    model = joblib.load("soil_health_model.pkl")
    crop_encoder = joblib.load("crop_encoder.pkl")
    soil_health_encoder = joblib.load("soil_health_encoder.pkl")
    logger.info("Models loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {e}")
    raise

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
    # Check if crop is valid
    if Crop not in crop_encoder.classes_:
        return "Invalid Crop Type", None
    
    # Encode crop type
    Crop_encoded = crop_encoder.transform([Crop])[0]
    
    # Create input DataFrame
    input_data = pd.DataFrame([[N, P, K, Moisture, Latitude, Longitude, Crop_encoded]],
                              columns=["N", "P", "K", "Moisture", "Latitude", "Longitude", "Crop Type"])
    
    # Predict soil health
    prediction = model.predict(input_data)
    
    # Decode predicted class
    soil_health = soil_health_encoder.inverse_transform(prediction)[0]
    
    # Check for nutrient deficiencies
    deficiency = check_deficiency(N, P, K, Crop)
    return soil_health, deficiency

# API Endpoints
@app.get("/")
def read_root():
    return {"message": "Welcome to the Soil Health API"}

@app.post("/predict", response_model=SoilHealthResponse)
def predict(data: SoilDataInput):
    try:
        # Extract data
        N, P, K, Moisture = data.N, data.P, data.K, data.Moisture
        Latitude, Longitude, Crop = data.Latitude, data.Longitude, data.Crop
        
        # Log input data
        logger.info(f"Received data: N={N}, P={P}, K={K}, Moisture={Moisture}, "
                    f"Latitude={Latitude}, Longitude={Longitude}, Crop={Crop}")
        
        # Predict soil health
        prediction, deficiency = predict_soil_health(N, P, K, Moisture, Latitude, Longitude, Crop)
        
        # Create deficiency message
        deficiency_str = ", ".join([f"{nutrient} deficient by {deficit}" for nutrient, deficit in deficiency.items() if deficit > 0])
        deficiency_str = deficiency_str if deficiency_str else "No Deficiency"
        
        # Log results
        logger.info(f"Prediction: {prediction}, Deficiency: {deficiency_str}")
        
        # Return results
        return {
            "soil_health": prediction,
            "deficiency": deficiency_str,
            "deficiency_details": deficiency
        }
    except Exception as e:
        logger.error(f"Error processing request: {e}")
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
            
            # Create deficiency message
            deficiency_str = ", ".join([f"{nutrient} deficient by {deficit}" for nutrient, deficit in deficiency.items() if deficit > 0])
            deficiency_str = deficiency_str if deficiency_str else "No Deficiency"
            
            # Log results
            logger.info(f"Prediction for {Crop} at ({Latitude}, {Longitude}): {prediction}, Deficiency: {deficiency_str}")
            
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
            logger.error(f"Error processing batch item: {e}")
            results.append({"error": str(e)})
    
    return {"results": results}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)