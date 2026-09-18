"""
LLM layer: turns a free text query into structured housing features
(extraction), and turns a model prediction back into a plain English
explanation. Both calls go through Groq's OpenAI-compatible chat API.
"""

import json
import os

from groq import Groq

from defaults import (
    FURNISHING_OPTIONS,
    STATUS_OPTIONS,
    TRANSACTION_OPTIONS,
    TYPE_OPTIONS,
)

MODEL_NAME = "openai/gpt-oss-120b"

_client = None


def get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set")
        _client = Groq(api_key=api_key)
    return _client


EXTRACTION_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_housing_features",
        "description": (
            "Extract structured housing features mentioned in a query about "
            "a property in Delhi. Only fill a field if the user actually "
            "mentioned it; leave everything else null."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "bhk": {
                    "type": ["integer", "null"],
                    "description": "Number of bedrooms, e.g. 3 for '3BHK'.",
                },
                "area_sqft": {
                    "type": ["number", "null"],
                    "description": "Built up area in square feet.",
                },
                "locality": {
                    "type": ["string", "null"],
                    "description": "Delhi locality or area name, e.g. Dwarka, Rohini, Saket.",
                },
                "furnishing": {
                    "type": ["string", "null"],
                    "enum": FURNISHING_OPTIONS + [None],
                },
                "bathroom": {"type": ["integer", "null"]},
                "parking": {
                    "type": ["integer", "null"],
                    "description": "Number of car parking spots.",
                },
                "property_type": {
                    "type": ["string", "null"],
                    "enum": TYPE_OPTIONS + [None],
                    "description": "Apartment or an independent Builder_Floor.",
                },
                "transaction": {
                    "type": ["string", "null"],
                    "enum": TRANSACTION_OPTIONS + [None],
                    "description": "New_Property if newly built, Resale otherwise.",
                },
                "status": {
                    "type": ["string", "null"],
                    "enum": STATUS_OPTIONS + [None],
                },
            },
            "required": [],
        },
    },
}


def extract_features(query: str) -> dict:
    client = get_client()
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "You extract structured housing fields from a user's question. Never guess a value the user did not state.",
            },
            {"role": "user", "content": query},
        ],
        tools=[EXTRACTION_TOOL],
        tool_choice={"type": "function", "function": {"name": "extract_housing_features"}},
        temperature=0,
    )
    tool_call = response.choices[0].message.tool_calls[0]
    return json.loads(tool_call.function.arguments)


def explain_prediction(query: str, used_features: dict, predicted_price: float) -> str:
    client = get_client()
    price_text = f"₹{predicted_price:,.0f}"
    prompt = (
        f"The user asked: \"{query}\"\n\n"
        f"A regression model trained on Delhi housing listings predicted a price of {price_text} "
        f"using these features: {json.dumps(used_features)}.\n\n"
        "Write a short, plain English explanation (2 to 3 sentences) of this estimate for the user. "
        "Mention the locality and BHK naturally. If a feature was not given by the user and a typical "
        "default was used instead, say so briefly. Do not repeat the exact feature dictionary, write "
        "normal sentences. Do not use markdown."
    )
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )
    return response.choices[0].message.content.strip()
