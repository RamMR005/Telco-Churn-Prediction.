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


