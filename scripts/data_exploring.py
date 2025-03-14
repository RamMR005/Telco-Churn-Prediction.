import pandas as pd

"""try:"""
df = pd.read_csv("../data/Telco-Customer-Churn.csv")  # Relative path
#except FileNotFoundError:
#df = pd.read_csv("data/Telco-Customer-Churn.csv")  # Alternative path

# Show basic info
print(df.head())  # First few rows
print(df.info())  # Column types
print(df.isnull().sum())  # Check for missing values

pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.expand_frame_repr', False) # Prevent line wrapping
print(df.head())

