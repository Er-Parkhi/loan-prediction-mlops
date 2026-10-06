import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import traceback

app = FastAPI(
    title="Loan Prediction API",
    version="1.0"
)

try:
    model = joblib.load("./model/loan_default.pkl")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

class LoanInput(BaseModel):
    current_loan_amount: float = Field(alias="Current Loan Amount")
    term: float = Field(alias="Term") # If you passed text like "Short Term", change type to str
    credit_score: float = Field(alias="Credit Score")
    annual_income: float = Field(alias="Annual Income")
    years_in_current_job: float = Field(alias="Years in current job")
    home_ownership: float = Field(alias="Home Ownership")
    purpose: float = Field(alias="Purpose")
    monthly_debt: float = Field(alias="Monthly Debt")
    years_of_credit_history: float = Field(alias="Years of Credit History")
    months_since_last_delinquent: float = Field(alias="Months since last delinquent")
    number_of_open_accounts: float = Field(alias="Number of Open Accounts")
    number_of_credit_problems: float = Field(alias="Number of Credit Problems")
    current_credit_balance: float = Field(alias="Current Credit Balance")
    maximum_open_credit: float = Field(alias="Maximum Open Credit")
    bankruptcies: float = Field(alias="Bankruptcies")
    tax_liens: float = Field(alias="Tax Liens")

@app.get("/")
def home():
    return {"message": "Loan Prediction API"}

@app.post("/predict")
def predict(data: LoanInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Machine learning model is not loaded.")
        
    try:
        # 1. Convert input data to a base DataFrame row using Pydantic aliases
        input_dict = data.model_dump(by_alias=True)
        df_raw = pd.DataFrame([input_dict])
        
        # 2. Re-create the encoding steps used during training.
        # Example for Term: If you encoded string labels like "Short Term" or numeric representations
        # We simulate pd.get_dummies matching your exact training feature columns.
        
        # Start a DataFrame with all columns initialized to 0
        expected_features = list(model.feature_names_in_)
        input_data = pd.DataFrame(0.0, index=[0], columns=expected_features)
        
        # 3. Map numerical columns directly from raw input
        for col in expected_features:
            if col in df_raw.columns:
                input_data[col] = float(df_raw[col].iloc[0])
                
        # 4. Handle One-Hot Encoded columns manually (e.g., for Term)
        # Check what value came in for Term (e.g., 1.0 or a string)
        incoming_term = data.term 
        
        # If your model expects 'Term_Short Term', map it when conditions match:
        if incoming_term == 1.0 or str(incoming_term).lower() == "short term":
            if "Term_Short Term" in input_data.columns:
                input_data["Term_Short Term"] = 1.0
        else:
            if "Term_Long Term" in input_data.columns:
                input_data["Term_Long Term"] = 1.0
                
        # NOTE: If you have other encoded features like 'Home Ownership_Rent', 
        # you will need to map them here using the same logic.

        # 5. Final validation to confirm no columns are missing or misaligned
        input_data = input_data[expected_features]
       
        # 6. Run prediction
        prediction = model.predict(input_data)
     
        return {
            "prediction": float(prediction[0])
        }
        
    except Exception as e:
        print("--- PREDICTION ERROR ---")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction Failed: {str(e)}")
