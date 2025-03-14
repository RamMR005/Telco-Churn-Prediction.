import pickle

model = {"example": "test model"}  # Replace with your ML model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved!")

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

print("Model loaded:", model)
import sys
print(sys.version)
