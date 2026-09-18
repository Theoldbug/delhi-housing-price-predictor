# Delhi Housing Price Predictor

Ask for a home price the way you'd ask a person, not a form. Type something
like *"what would a 3BHK in Dwarka cost, semi furnished?"* and get back an
estimate with the reasoning behind it, instead of filling in a dropdown for
every field.

## Why

Price prediction tools usually make you fill in BHK, locality, area, and
furnishing as separate form fields. This project closes that gap: a language
model extracts the structured fields from a plain-English question, a
regression model trained on real Delhi listings predicts the price, and a
second language model call turns the number back into a written explanation.

## Architecture

```
User question
  -> LLM (tool calling) extracts {locality, bhk, area, furnishing, ...}
  -> missing fields filled with dataset defaults
  -> scikit-learn regression pipeline predicts price
  -> LLM turns the prediction into a plain-English explanation
  -> FastAPI returns {predicted_price, features, explanation}
  -> static frontend renders the result
```

- **Data**: `data/raw_listings.csv`, 1259 Delhi listings (locality, BHK, area,
  bathrooms, parking, furnishing, property type, transaction type, price).
  Cleaned in `notebooks/01_data_and_model.ipynb`: dropped rows with corrupted
  fields, bucketed rare localities, removed a target-leaking column, filtered
  price-per-sqft outliers.
- **Model**: `scikit-learn` `Pipeline` (`ColumnTransformer` +
  `RandomForestRegressor`), evaluated against a `LinearRegression` baseline
  and `GradientBoostingRegressor`, trained in the same notebook, saved to
  `backend/model/price_model.pkl`.
- **LLM layer**: `backend/llm.py`, Groq API (`openai/gpt-oss-120b`). Feature
  extraction uses forced tool calling with a strict schema; the model is
  instructed to leave a field `null` rather than guess. A second call
  generates the explanation from the extracted features and the prediction.
- **Backend**: `backend/main.py`, FastAPI. `/predict` runs the full pipeline
  above; `/health` for a liveness check.
- **Frontend**: `frontend/index.html`, plain HTML/CSS/JS, no build step.

## Running locally

**Backend**
```
cd backend
pip install -r requirements.txt
cp .env.example .env   # add your GROQ_API_KEY
uvicorn main:app --reload
```

**Frontend**

Open `frontend/index.html` directly in a browser (it targets
`http://localhost:8000` automatically when not deployed).

## Live demo

_Deployment in progress — link to follow._
