import os
import pandas as pd

# Dynamic path that works perfectly on Windows (locally) and Linux (GitHub Actions)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "credit_train.csv")

print(f"Loading data from: {data_path}")
credit_df = pd.read_csv(data_path, header=0, sep=',')

# Your remaining train.py code continues below...
# (Make sure to append your model training logic here if it gets overwritten)
