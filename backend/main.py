import os
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from defaults import (
    DEFAULT_AREA_SQFT,
    DEFAULT_BATHROOM,
    DEFAULT_BHK,
    DEFAULT_FURNISHING,
    DEFAULT_PARKING,
    DEFAULT_STATUS,
    DEFAULT_TRANSACTION,
    DEFAULT_TYPE,
    FURNISHING_OPTIONS,
    STATUS_OPTIONS,
    TRANSACTION_OPTIONS,
    TYPE_OPTIONS,
    match_locality,
)
from llm import explain_prediction, extract_features

load_dotenv()

MODEL_PATH = Path(__file__).parent / "model" / "price_model.pkl"
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Delhi Housing Price Predictor")

allowed_origins = [
    origin.strip()
    for origin in os.environ.get("FRONTEND_ORIGIN", "").split(",")
    if origin.strip()
]
allowed_origins += ["http://localhost:5500", "http://127.0.0.1:5500", "null"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)


class PredictRequest(BaseModel):
    query: str


def resolve_features(extracted: dict) -> dict:
    """Fill in anything the LLM did not extract with a dataset default, and
    normalise categorical values to the exact strings the model was trained
    on (OneHotEncoder(handle_unknown='ignore') drops anything it doesn't
    recognise, so a mismatched casing silently loses that signal)."""

    def pick(value, options, default):
        if value and value in options:
            return value
        return default

    return {
        "Area": extracted.get("area_sqft") or DEFAULT_AREA_SQFT,
        "BHK": extracted.get("bhk") or DEFAULT_BHK,
        "Bathroom": extracted.get("bathroom") or DEFAULT_BATHROOM,
        "Parking": extracted.get("parking") if extracted.get("parking") is not None else DEFAULT_PARKING,
        "Locality": match_locality(extracted.get("locality")),
        "Furnishing": pick(extracted.get("furnishing"), FURNISHING_OPTIONS, DEFAULT_FURNISHING),
        "Type": pick(extracted.get("property_type"), TYPE_OPTIONS, DEFAULT_TYPE),
        "Status": pick(extracted.get("status"), STATUS_OPTIONS, DEFAULT_STATUS),
        "Transaction": pick(extracted.get("transaction"), TRANSACTION_OPTIONS, DEFAULT_TRANSACTION),
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictRequest):
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        extracted = extract_features(query)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not understand the query: {exc}")

    features = resolve_features(extracted)
    row = pd.DataFrame([features])
    predicted_price = float(model.predict(row)[0])

    try:
        explanation = explain_prediction(query, features, predicted_price)
    except Exception:
        explanation = "Estimate generated from the model; a written explanation could not be generated right now."

    return {
        "predicted_price": predicted_price,
        "features": features,
        "explanation": explanation,
    }
