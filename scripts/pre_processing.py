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
print (df_cleaned.head())
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.expand_frame_repr', False) # Prevent line wrapping
print(df_cleaned.head())
