import pickle
import pandas as pd
from fastapi import FastAPI
import uvicorn
from typing import Dict, Any, Literal
from pydantic import BaseModel, Field, ConfigDict

# Request
class Employee(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    education: Literal["Bachelor's", "Master's & above", "Below Secondary"]
    gender: Literal["Male", "Female"]
    recruitment_channel: Literal["Other", "Sourcing", "Referred"]
    department: Literal[
        "Sales & Marketing", "Operations", "Technology", "Procurement",
        "Analytics", "Finance", "HR", "R&D", "Legal"
    ]
    kpis_met_80: int = Field(alias="kpis_met_>80%")
    awards_won: int = Field(alias="awards_won?")
    no_of_trainings: int = Field(..., ge=1, le=8)
    age: int = Field(..., ge=20, le=60)
    previous_year_rating: int = Field(..., ge=1, le=5)
    length_of_service: int = Field(..., ge=1, le=34)
    avg_training_score: int = Field(..., ge=40, le=99)
    region: int = Field(..., ge=1, le=34)

# Response
class PredictResponse(BaseModel):
    is_promoted_probability: float
    is_promoted: bool

app = FastAPI(title="employee-promotion-prediction")

with open('xgb_model.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

def predict_single(employee):
    data_point = pd.DataFrame([employee])
    result = pipeline.predict_proba(data_point)[0, 1]
    return float(result)


@app.post("/predict")
def predict(employee: Employee) -> PredictResponse:
    prob = predict_single(employee.model_dump(by_alias=True))

    return PredictResponse(
        is_promoted_probability=prob,
        is_promoted=bool(prob >= 0.5)
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9797)