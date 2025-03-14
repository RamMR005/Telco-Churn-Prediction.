## 📖 Table of Contents
- [Project Overview](#-telco-customer-churn-prediction)
- [Project Structure](#-project-structure)
- [Data Exploration](#-1-data-exploring)
- [Data Preprocessing](#-2-data-preprocessing)
- [Model Training](#-step-3-model-training)
- [Model Serialization](#-step-4-model-serialization-saving--loading)
- [Flask API Deployment](#-step-5-flask-api-for-prediction)
- [Installation](#-installation)
- [Usage](#usage)
# 📊 Telco Customer Churn Prediction

This project aims to predict customer churn using machine learning. It includes data preprocessing, model training, and an API for predictions.

---

## 📂 Project Structure  

```📂 Telco-Churn-Prediction
│── 📂 data # Raw & processed datasets
│── 📂 scripts # All python scripts
│── 📂 model # Saved machine learning models
│── 📂 app # Flask API for deployment
│── requirements.txt # Dependencies
│── README.md # Project Documentation
```


---

##  1️: Data Exploring

### 📜 **Step 1: Load & Explore Data**
```python
import pandas as pd  # Import pandas for data handling

df = pd.read_csv("../data/Telco-Customer-Churn.csv")  # Load dataset

# Show first 5 rows
print(df.head())  

# Show dataset information
print(df.info())  

# Check for missing values
print(df.isnull().sum()) 

```

📌 Explanation:
Load dataset using pd.read_csv()
Display first few rows using df.head()
Check data types & null values

##  2: Data Preprocessing

description: |
  This script loads the raw dataset, cleans it, and prepares it for model training.

steps:
  - Load the raw dataset from the `data` directory.
  - Convert the "TotalCharges" column to numeric (handling empty values).
  - Remove the `customerID` column as it is not needed for prediction.
  - Convert the target column `Churn` into numerical values (Yes → 1, No → 0).
  - Perform One-Hot Encoding on categorical features using `pd.get_dummies()`.
  - Save the cleaned dataset as `processed_telco.csv` in the `data` directory.

code: |
  ```python
  import pandas as pd

  def load_and_preprocess_data():
      df = pd.read_csv("../data/Telco-Customer-Churn.csv")

      # Convert "TotalCharges" to numeric (it has some empty values as " ")
      df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')

      # Drop customerID (not useful for predictions)
      df.drop(columns=["customerID"], inplace=True)

      # Convert target variable to 0 & 1
      df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

      # Convert categorical columns to numerical
      df = pd.get_dummies(df, drop_first=True)  # One-hot encoding

      return df

  # Run preprocessing and save cleaned data
  df_cleaned = load_and_preprocess_data()
  df_cleaned.to_csv("../data/processed_telco.csv", index=False)

  print("✅ Data preprocessed & saved!")
  print(df_cleaned.head())
  pd.set_option('display.max_columns', None)  # Show all columns
  pd.set_option('display.expand_frame_repr', False) # Prevent line wrapping
  print(df_cleaned.head()) 
  ```
## Step 3: Model Training

description: |
  This script trains a Random Forest model to predict customer churn based on the preprocessed dataset.

steps:
  - Load the cleaned dataset from `processed_telco.csv`.
  - Split the dataset into features (`X`) and target variable (`y`).
  - Perform a train-test split (80% training, 20% testing).
  - Train a `RandomForestClassifier` with 200 trees (`n_estimators=200`) and a maximum depth of 10.
  - Evaluate the model using `accuracy_score`.
  - Save the trained model (`model.pkl`) and feature names (`features.pkl`) in the `model` directory.

code: |
  ```python
  import pandas as pd
  import os
  from sklearn.model_selection import train_test_split
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.metrics import accuracy_score
  import joblib

  # ✅ Load cleaned dataset
  df = pd.read_csv("../data/processed_telco.csv")

  # ✅ Split features and target
  X = df.drop("Churn", axis=1)
  y = df["Churn"]

  # ✅ Train-test split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # ✅ Train model
  model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
  model.fit(X_train, y_train)

  # ✅ Evaluate model
  y_pred = model.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  print(f"✅ Model Accuracy: {accuracy:.4f}")

  # ✅ Ensure model directory exists
  model_dir = "../model"
  os.makedirs(model_dir, exist_ok=True)

  # ✅ Save model
  joblib.dump(model, os.path.join(model_dir, "model.pkl"))
  joblib.dump(list(X_train.columns), os.path.join(model_dir, "features.pkl"))  # Save feature names

  print("✅ Model & feature names saved successfully!")
```
## Step 4: Model Serialization (Saving & Loading)

description: |
  This script demonstrates how to save and load a machine learning model using Python's `pickle` module.

steps:
  - Create a sample model object (replace this with an actual trained ML model).
  - Save the model to a file (`model.pkl`) using `pickle.dump()`.
  - Load the saved model from `model.pkl` using `pickle.load()`.
  - Print the loaded model to verify successful serialization.
  - Print the Python version for debugging and compatibility checks.

code: |
  ```python
  import pickle

  # ✅ Create a sample model (replace with your actual trained ML model)
  model = {"example": "test model"}

  # ✅ Save the model to a file
  with open("model.pkl", "wb") as f:
      pickle.dump(model, f)

  print("✅ Model saved!")

  # ✅ Load the model from the file
  with open("model.pkl", "rb") as f:
      model = pickle.load(f)

  print("✅ Model loaded:", model)

  # ✅ Print Python version for debugging
  import sys
  print(sys.version)
```
## Step 5: Flask API for Prediction

description: |
  This script sets up a Flask API that allows users to send JSON data and receive churn predictions.
  It loads a trained model and expects specific input features.

steps:
  - Initialize a Flask web application.
  - Load the trained model (`model.pkl`) and feature names (`features.pkl`).
  - Define an API endpoint `/predict` to accept POST requests.
  - Convert input JSON to a Pandas DataFrame.
  - Apply one-hot encoding and ensure feature alignment.
  - Handle missing columns by setting their values to 0.
  - Ensure correct column order before making predictions.
  - Return predictions as JSON.
  - Handle potential errors and return HTTP 500 in case of failure.

code: |
  ```python
  from flask import Flask, request, jsonify
  import pandas as pd
  import joblib

  app = Flask(__name__)

  # ✅ Load trained model & feature names
  model = joblib.load("../model/model.pkl")  # Ensure correct path
  expected_features = joblib.load("../model/features.pkl")  # Load feature names

  @app.route('/predict', methods=['POST'])
  def predict():
      try:
          # ✅ Convert input JSON to DataFrame
          df = pd.DataFrame(request.json)

          # ✅ Apply One-Hot Encoding
          df = pd.get_dummies(df)

          # ✅ Add missing columns with default value 0
          for col in expected_features:
              if col not in df.columns:
                  df[col] = 0

          # ✅ Ensure correct column order
          df = df[expected_features]

          # ✅ Make prediction
          prediction = model.predict(df)

          return jsonify({"prediction": prediction.tolist()})

      except Exception as e:
          return jsonify({"error": str(e)}), 500

  if __name__ == '__main__':
      app.run(debug=True)
```
usage: |

Run the Flask API:
* python app.py
* Send a POST request with input JSON data using Postman or cURL:
```json
{
  "gender": ["Male"],
  "SeniorCitizen": [0],
  "Partner": ["Yes"],
  "tenure": [12],
  "InternetService": ["Fiber optic"],
  "Contract": ["Month-to-month"]
} 
```
Expected Response:
``` json
{
  "prediction": [1]  # 1 indicates churn, 0 means no churn
}
```

# 📦 Installation

## 1️⃣ Clone the Repository
```bash
git clone https://github.com/RamMR005/Telco-Churn-Prediction.git
cd Telco-Churn-Prediction
```

## 2️⃣ Set Up a Virtual Environment

```bash
python -m venv venv  # Create a virtual environment
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate  # Windows
```
## 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
