import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import pandas as pd
df = pd.read_csv("diabetes_012_health_indicators_BRFSS2021.csv")

from sklearn.model_selection import train_test_split
# Step 2: Train/Test Split and Stratification
important_feature = 'BMI'  
median_value = df[important_feature].median()


# Split data into two sets based on the important feature
low_bmi_data = df[df[important_feature] <= median_value]
high_bmi_data = df[df[important_feature] > median_value]

# Ensure the important feature distributions are consistent
print("Low BMI Median:", low_bmi_data[important_feature].median())
print("High BMI Median:", high_bmi_data[important_feature].median())

# Train/Test Split for each subset
X_low = low_bmi_data.drop(columns=['Diabetes_012'])
y_low = low_bmi_data['Diabetes_012']
X_high = high_bmi_data.drop(columns=['Diabetes_012'])
y_high = high_bmi_data['Diabetes_012']

X_train_low = X_low.drop(columns=['body_mass_index'], errors='ignore')
X_train_high = X_high.drop(columns=['body_mass_index'], errors='ignore')
X_train_low, X_test_low, y_train_low, y_test_low = train_test_split(X_low, y_low, test_size=0.2, stratify=y_low, random_state=42)
X_train_high, X_test_high, y_train_high, y_test_high = train_test_split(X_high, y_high, test_size=0.2, stratify=y_high, random_state=42)


# After splitting the dataset
X_train_low = X_train_low.astype({col: 'float64' for col in X_train_low.select_dtypes(include='int').columns})
X_test_low = X_test_low.astype({col: 'float64' for col in X_test_low.select_dtypes(include='int').columns})

X_train_high = X_train_high.astype({col: 'float64' for col in X_train_high.select_dtypes(include='int').columns})
X_test_high = X_test_high.astype({col: 'float64' for col in X_test_high.select_dtypes(include='int').columns})

X_test_low = X_test_low.drop(columns=['diabetes_012'], errors='ignore')
X_test_high = X_test_high.drop(columns=['diabetes_012'], errors='ignore')

import joblib
import pandas as pd
from sklearn.metrics import f1_score

def main():
    # Paths to models and test data
    low_model_path = "/code/app/tuned_xgb_low_bmi.pkl"
    test_data_path_X = "X_test_low.csv"
    test_data_path_y = "y_test_low.csv"
    low_model_path = "/code/app/tuned_xgb_low_bmi.pkl"
    low_bmi_model = joblib.load(low_model_path)
    print(f"Model type after loading: {type(low_bmi_model)}")

    # Load the trained model
    try:
        low_bmi_model = joblib.load(low_model_path)
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model file not found at {low_model_path}")
        return

    # Load the test data
    try:
        X_test_low = pd.read_csv(test_data_path_X, encoding="utf-8")
        y_test_low = pd.read_csv(test_data_path_y, encoding="utf-8").squeeze()
        print("Test data loaded successfully.")
    except UnicodeDecodeError:
        print("UTF-8 decoding failed. Retrying with ISO-8859-1 encoding.")
        X_test_low = pd.read_csv(test_data_path_X, encoding="ISO-8859-1")
        y_test_low = pd.read_csv(test_data_path_y, encoding="ISO-8859-1").squeeze()
    except FileNotFoundError as e:
        print(f"Error loading test data: {e}")
        return

    # Validate the model and data
    if low_bmi_model is not None and not X_test_low.empty and not y_test_low.empty:
        # Make predictions
        y_pred = low_bmi_model.predict(X_test_low)

        # Evaluate performance
        f1 = f1_score(y_test_low, y_pred, average="weighted")
        print(f"F1 Score for Low BMI Model: {f1}")
    else:
        print("Model or test data is invalid.")
    


if __name__ == "__main__":
    main()
