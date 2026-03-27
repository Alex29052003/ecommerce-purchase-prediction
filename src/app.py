import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Ecommerce Purchase Prediction API")
with open("model/ecommerce_model.pkl", "rb") as f:
    artifacts = pickle.load(f)

model = artifacts["model"]
columns = artifacts["columns"]
threshold = artifacts["threshold"]
class SessionData(BaseModel):
    Administrative: int
    Administrative_Duration: float
    Informational: int
    Informational_Duration: float
    ProductRelated: int
    ProductRelated_Duration: float
    BounceRates: float
    ExitRates: float
    PageValues: float
    SpecialDay: float
    OperatingSystems: int
    Browser: int
    Region: int
    TrafficType: int
    Weekend: bool

@app.get("/")
def root():
    return {"message": "Ecommerce Purchase API is running"}
@app.post("/predict")
def predict(data: SessionData):
    input_dict = data.dict()
    df_input = pd.DataFrame([input_dict])

    for col in columns:
        if col not in df_input.columns:
            df_input[col] = 0

    df_input = df_input[columns]
    probability = model.predict_proba(df_input)[0][1]
    prediction = int(probability > threshold)

    return {
        "purchase_probability": round(float(probability), 4),
        "prediction": prediction,
        "label": "will_buy" if prediction == 1 else "no_buy"}