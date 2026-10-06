import os
import pandas as pd
import joblib

# Dynamic absolute pathing
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
test_df_path = os.path.join(BASE_DIR, "data", "x_test_sample.csv")
model_path = os.path.join(BASE_DIR, "model", "loan_default.pkl")

print(f"Looking for test data at: {test_df_path}")

if not os.path.exists(test_df_path):
    raise FileNotFoundError(f"Missing test data file at: {test_df_path}")

test_df = pd.read_csv(test_df_path)
print("Test sample loaded successfully!")

# Your remaining evaluation logic continues here...
