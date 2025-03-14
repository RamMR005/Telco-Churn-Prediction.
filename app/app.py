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
