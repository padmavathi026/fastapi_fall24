from typing import Union

from fastapi import FastAPI

import joblib
from pydantic import BaseModel
import pandas as pd
import numpy as np



# Path to the saved low BMI model
low_bmi_model_path = "/code/app/tuned_xgb_low_bmi.pkl"

# Load the low BMI model
try:
    low_bmi_model = joblib.load(low_bmi_model_path)
    print("Low BMI model loaded successfully.")
except Exception as e:
    low_bmi_model = None
    print(f"Error loading Low BMI model: {e}")




app = FastAPI()

with open('/code/app/tuned_xgb_low_bmi.pkl', 'rb') as f: 
    reloaded_model = joblib.load(f) 

class Payload(BaseModel):
    HighBP: float
    HighChol: float
    CholCheck: float
    BMI: float
    Smoker: float
    Stroke: float
    HeartDiseaseorAttack: float
    PhysActivity: float
    Fruits: float
    Veggies: float
    HvyAlcoholConsump: float
    AnyHealthcare: float
    NoDocbcCost: float
    GenHlth: float
    MentHlth: float
    PhysHlth: float
    DiffWalk: float
    Sex: float
    Age: float
    Education: float
    Income: float

try:
    model = joblib.load(low_bmi_model_path)
    print("Model loaded successfully.")
except Exception as e:
    model = None
    print(f"Error loading model: {e}")


@app.get("/items/{item_id}")
def read_item(
    item_id: int,q: 
    Union[str, None] = None,
    x: Union[str, None] = None
):
    return {"item_id": item_id, "q": q, "x":x} 


@app.post("/")
async def handle_post(data: dict):
    return {"message": "Data received", "data": data}


@app.get("/")
def read_root():
    return {"message": "Welcome to the Diabetes Prediction API",
            "Name" : "Padmavathi Moorthy",
            "Model" : "XGBoost Classifier"}

@app.post("/predict")
def predict(payload: Payload):
    if not low_bmi_model:
        return {"error": "Low BMI model not loaded."}
    
    # Convert payload to DataFrame
    input_data = pd.DataFrame([payload.dict().values()], columns=payload.dict().keys())

    # Ensure 'Diabetes_012' is not included
    if "Diabetes_012" in input_data.columns:
        input_data = input_data.drop(columns=["Diabetes_012"])
    
    try:
        # Make prediction
        prediction = low_bmi_model.predict(input_data)[0]
        return {"prediction": int(prediction)}
    except Exception as e:
        return {"error": str(e)}

