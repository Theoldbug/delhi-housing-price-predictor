"""
Fallback values used to fill in whatever the LLM extraction step does not
find in the user's query. Computed from data/clean_listings.csv (median for
numeric fields, mode for categorical fields).
"""

DEFAULT_AREA_SQFT = 1200
DEFAULT_BHK = 3
DEFAULT_BATHROOM = 2
DEFAULT_PARKING = 1
DEFAULT_FURNISHING = "Semi-Furnished"
DEFAULT_TYPE = "Builder_Floor"
DEFAULT_STATUS = "Ready_to_move"
DEFAULT_TRANSACTION = "Resale"
DEFAULT_LOCALITY = "Other"

FURNISHING_OPTIONS = ["Furnished", "Semi-Furnished", "Unfurnished"]
TYPE_OPTIONS = ["Apartment", "Builder_Floor"]
STATUS_OPTIONS = ["Ready_to_move", "Almost_ready"]
TRANSACTION_OPTIONS = ["New_Property", "Resale"]

KNOWN_LOCALITIES = [
    "Shahdara", "Alaknanda", "Rohini Sector 24", "Dwarka", "Lajpat Nagar 3",
    "Lajpat Nagar 2", "Laxmi Nagar", "Karol Bagh", "Greater Kailash",
    "Hauz Khas", "New Friends Colony", "Saket", "Dilshad Garden",
    "Kirti Nagar", "Mehrauli", "Vasundhara Enclave", "Okhla", "Malviya Nagar",
    "Kalkaji", "Narela", "Chittaranjan Park", "Safdarjung Enclave",
    "Commonwealth Games Village 2010", "Sultanpur", "Paschim Vihar",
    "Chhattarpur", "Vasant Kunj", "Uttam Nagar", "Mathura Road",
    "Sheikh Sarai Phase 1", "Patel Nagar", "Lajpat Nagar",
    "Mahavir Enclave Part 1", "Patel Nagar West", "Punjabi Bagh",
    "Dwarka Mor", "Sheikh Sarai", "Mahavir Enclave", "Punjabi Bagh West",
    "Dwarka Sector 12", "Chhattarpur Enclave Phase2", "Budh Vihar",
    "Dwarka Sector 24", "Rohini Sector 23", "Budh Vihar Phase 1",
    "Greater Kailash 1", "Geeta Colony", "Uttam Nagar West",
]


def match_locality(raw: str | None) -> str:
    """Map free-text locality mentions to a category the model was trained on.

    The model's OneHotEncoder was fit with handle_unknown='ignore', so any
    string not in KNOWN_LOCALITIES silently contributes nothing to the
    prediction. Matching to the closest known name (rather than passing the
    raw string through) keeps the locality signal instead of losing it.
    """
    if not raw:
        return DEFAULT_LOCALITY
    raw_lower = raw.lower().strip()
    for locality in KNOWN_LOCALITIES:
        if locality.lower() == raw_lower:
            return locality
    for locality in KNOWN_LOCALITIES:
        if locality.lower().startswith(raw_lower) or raw_lower in locality.lower():
            return locality
    return DEFAULT_LOCALITY
