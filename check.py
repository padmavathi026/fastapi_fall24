import joblib
import os

# File path
model_path = "/Users/padmavathimoorthy/Downloads/tuned_xgb_low_bmi.pkl"

# Check if the file exists
if os.path.exists(model_path):
    print(f"File exists at {model_path}")
else:
    print(f"File does not exist at {model_path}")

# Attempt to load the model
try:
    model = joblib.load(model_path)
    print("Model loaded successfully.")
    print("Model type:", type(model))
except Exception as e:
    print(f"Error loading model: {e}")
