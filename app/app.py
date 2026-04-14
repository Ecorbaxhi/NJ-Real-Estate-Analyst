# Let's import the libraries we need for the backend and machine learning
from pathlib import Path
import pandas as pd

# Let's import FastAPI to create our web application
from fastapi import FastAPI

# Let's allow frontend to talk with backend
from fastapi.middleware.cors import CORSMiddleware

# Let's import BaseModel to define the input structure
from pydantic import BaseModel

# Let's import tools for model training
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Let's create the FastAPI app
app = FastAPI(title="NJ Real Estate Analyst API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Let's define the structure of the input data coming from the user
class HouseInput(BaseModel):
    bedrooms: int
    bathrooms: float
    sqft_living: int
    floors: float
    yr_built: int
    zipcode: str
    address: str   # 👈 NEW
    listing_price: float
    days_on_market: int


# Let's create a function to estimate fair price using the model
def estimate_price(model, house):
    return model.predict(house)[0]


# Let's create a function to estimate price from comparable houses
def estimate_price_from_comps(comps, house):
    
    # Let's make sure we have comparable houses
    if len(comps) == 0:
        return None

    comps = comps.copy()
    comps["price_per_sqft"] = comps["price"] / comps["sqft_living"]
    avg_price_sqft = comps["price_per_sqft"].mean()
    estimated_price_comps = avg_price_sqft * house["sqft_living"].values[0]

    return estimated_price_comps


# Let's combine model price and comps price in a smarter way
def combine_prices(predicted_price, estimated_price_comps, comps_count):
    if comps_count >= 20:
        return (0.3 * predicted_price) + (0.7 * estimated_price_comps)
    elif comps_count >= 10:
        return (0.5 * predicted_price) + (0.5 * estimated_price_comps)
    else:
        return (0.7 * predicted_price) + (0.3 * estimated_price_comps)


# Let's estimate price drop risk using BOTH days on market and overpricing
def estimate_price_drop_risk(difference, days_on_market):

    # Step 1: Base risk from days on market
    if days_on_market <= 30:
        risk_score = 1   # LOW
    elif days_on_market <= 60:
        risk_score = 2   # MEDIUM
    else:
        risk_score = 3   # HIGH

    # Step 2: Adjust based on pricing
    if difference > 10:
        risk_score += 1   # significantly overpriced → increase risk
    elif difference < -5:
        risk_score -= 1   # underpriced → decrease risk

    # Step 3: Keep score between 1 and 3
    risk_score = max(1, min(risk_score, 3))

    # Step 4: Convert to label
    if risk_score == 1:
        return "LOW"
    elif risk_score == 2:
        return "MEDIUM"
    else:
        return "HIGH"


# Let's find comparable houses using more realistic filters
def find_comparables(df, house):

    # Let's start with the full dataset
    comps = df.copy()

    # Let's read the house features
    sqft = house["sqft_living"].values[0]
    bedrooms = house["bedrooms"].values[0]
    bathrooms = house["bathrooms"].values[0]
    floors = house["floors"].values[0]
    year = house["yr_built"].values[0]

    # Let's filter by square footage (±20%)
    comps = comps[
        (comps["sqft_living"] >= 0.8 * sqft) &
        (comps["sqft_living"] <= 1.2 * sqft)
    ]

    # Let's filter by bedrooms (±1)
    comps = comps[
        (comps["bedrooms"] >= bedrooms - 1) &
        (comps["bedrooms"] <= bedrooms + 1)
    ]

    # Let's filter by bathrooms (±1)
    comps = comps[
        (comps["bathrooms"] >= bathrooms - 1) &
        (comps["bathrooms"] <= bathrooms + 1)
    ]

    # Let's filter by floors (exact or very close)
    comps = comps[
        (comps["floors"] >= floors - 0.5) &
        (comps["floors"] <= floors + 0.5)
    ]

    # Let's filter by year built (±10 years)
    comps = comps[
        (comps["yr_built"] >= year - 10) &
        (comps["yr_built"] <= year + 10)
    ]

    # Let's filter by zipcode only if it exists in input
    if "zipcode" in house.columns:
        zipcode = str(house["zipcode"].values[0]).zfill(5)
        comps = comps[comps["zipcode_str"] == zipcode]

    # Let's calculate a similarity score
    comps["distance"] = (
        abs(comps["sqft_living"] - sqft) +
        abs(comps["bedrooms"] - bedrooms) * 200 +
        abs(comps["bathrooms"] - bathrooms) * 150 +
        abs(comps["floors"] - floors) * 100 +
        abs(comps["yr_built"] - year) * 10
    )

    # Let's sort by most similar houses
    comps = comps.sort_values(by="distance")

    # Let's return the top 20 most similar houses
    return comps.head(20)


# Let's load the dataset from the data folder
data_path = Path(__file__).resolve().parent.parent / "data" / "kc_house_data.csv"
df = pd.read_csv(data_path)

# Let's store zipcode as string for matching and as numeric for the model
df["zipcode_str"] = df["zipcode"].astype(str).str.zfill(5)
df["zipcode_num"] = df["zipcode"].astype(int)

# Let's store the valid zipcodes present in the dataset
valid_zipcodes = set(df["zipcode_str"].unique())


# Let's define features WITH zipcode (location-aware model)
features_with_zip = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "floors",
    "yr_built",
    "zipcode"
]

# Let's define features WITHOUT zipcode (general model)
features_no_zip = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "floors",
    "yr_built"
]

# Let's prepare datasets for both models
X_with_zip = df[features_with_zip]
X_no_zip = df[features_no_zip]
y = df["price"]


# Let's split data for both models
X_train_zip, X_test_zip, y_train_zip, y_test_zip = train_test_split(
    X_with_zip, y, test_size=0.2, random_state=42
)

X_train_no_zip, X_test_no_zip, y_train_no_zip, y_test_no_zip = train_test_split(
    X_no_zip, y, test_size=0.2, random_state=42
)


# Let's train models WITH zipcode
model_zip = LinearRegression()
model_zip.fit(X_train_zip, y_train_zip)

rf_model_zip = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model_zip.fit(X_train_zip, y_train_zip)


# Let's train models WITHOUT zipcode
model_no_zip = LinearRegression()
model_no_zip.fit(X_train_no_zip, y_train_no_zip)

rf_model_no_zip = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model_no_zip.fit(X_train_no_zip, y_train_no_zip)



# Let's create a simple route to check if the API is running
@app.api_route("/", methods=["GET", "HEAD"])
def home():
    return {"message": "NJ Real Estate Analyst API is running"}

# Let's generate a human explanation for the result including location quality
def generate_explanation(price_diff_pct, days_on_market, comps_count, location_score):

    explanation = ""

    # Price logic
    if price_diff_pct > 10:
        explanation += "The property is significantly overpriced compared to similar homes. "
    elif price_diff_pct > 0:
        explanation += "The property is slightly overpriced compared to similar homes. "
    else:
        explanation += "The property is priced below market value. "

    # Time on market
    if days_on_market > 60:
        explanation += "It has been on the market for a long time, increasing the likelihood of a price drop. "
    elif days_on_market > 30:
        explanation += "It has been on the market for a moderate period. "

    # Location logic (NEW)
    if location_score > 0.7:
        explanation += "The location is very strong, with many nearby amenities such as schools, parks, and transport. "
    elif location_score > 0.4:
        explanation += "The location is average, with some nearby amenities. "
    else:
        explanation += "The location has limited nearby amenities, which may affect the value. "

    # Reliability
    if comps_count < 5:
        explanation += "However, few comparable properties were found, so the estimate may be less reliable."

    return explanation.strip()


# Let's convert address into latitude and longitude using OpenStreetMap
import requests

# Let's convert address into latitude and longitude using OpenStreetMap
def get_coordinates(address):
    url = "https://nominatim.openstreetmap.org/search"

    params = {
        "q": address,
        "format": "json",
        "limit": 1
    }

    headers = {
        "User-Agent": "NJ-Real-Estate-Analyst/1.0"
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        data = response.json()

        if len(data) == 0:
            return None, None

        lat = float(data[0]["lat"])
        lon = float(data[0]["lon"])
        return lat, lon
    except:
        return None, None


# Let's find nearby places using OpenStreetMap (Overpass API)
def get_nearby_places(lat, lon):

    # Let's use a more reliable Overpass server
    url = "https://overpass.kumi.systems/api/interpreter"

    # Let's search nearby places using nodes, ways, and relations
    query = f"""
    [out:json][timeout:25];
    (
      node(around:1000,{lat},{lon})["amenity"="school"];
      way(around:1000,{lat},{lon})["amenity"="school"];
      relation(around:1000,{lat},{lon})["amenity"="school"];

      node(around:1000,{lat},{lon})["amenity"="hospital"];
      way(around:1000,{lat},{lon})["amenity"="hospital"];
      relation(around:1000,{lat},{lon})["amenity"="hospital"];

      node(around:1000,{lat},{lon})["shop"="supermarket"];
      way(around:1000,{lat},{lon})["shop"="supermarket"];
      relation(around:1000,{lat},{lon})["shop"="supermarket"];

      node(around:1000,{lat},{lon})["railway"="station"];
      way(around:1000,{lat},{lon})["railway"="station"];
      relation(around:1000,{lat},{lon})["railway"="station"];

      node(around:1000,{lat},{lon})["leisure"="park"];
      way(around:1000,{lat},{lon})["leisure"="park"];
      relation(around:1000,{lat},{lon})["leisure"="park"];
    );
    out center;
    """

    headers = {
        "User-Agent": "NJ-Real-Estate-Analyst/1.0"
    }

    try:
        # Let's increase timeout and avoid silent failures
        response = requests.get(url, params={"data": query}, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data
    except:
        return None


# Let's count nearby places by category
def summarize_nearby_places(nearby_data):

    summary = {
        "schools": 0,
        "hospitals": 0,
        "supermarkets": 0,
        "stations": 0,
        "parks": 0
    }

    if nearby_data is None or "elements" not in nearby_data:
        return summary

    for place in nearby_data["elements"]:
        tags = place.get("tags", {})

        if tags.get("amenity") == "school":
            summary["schools"] += 1
        elif tags.get("amenity") == "hospital":
            summary["hospitals"] += 1
        elif tags.get("shop") == "supermarket":
            summary["supermarkets"] += 1
        elif tags.get("railway") == "station":
            summary["stations"] += 1
        elif tags.get("leisure") == "park":
            summary["parks"] += 1

    return summary


# Let's convert nearby places into a normalized location score (0 to 1)
def calculate_location_score(nearby_summary):

    # Let's assign weights to each category (importance)
    weights = {
        "schools": 1.0,
        "stations": 1.5,
        "supermarkets": 0.8,
        "parks": 0.6,
        "hospitals": 1.2
    }

    score = 0
    max_score = 0

    for key in weights:
        count = nearby_summary.get(key, 0)

        # Let's cap counts to avoid extreme influence
        capped_count = min(count, 5)

        score += capped_count * weights[key]
        max_score += 5 * weights[key]

    # Normalize between 0 and 1
    if max_score == 0:
        return 0

    return score / max_score

# Let's create the main prediction route that receives user input
@app.post("/predict")
def predict_house(data: HouseInput):

    # Let's build a more complete address for better geocoding
    full_address = f"{data.address}, {data.zipcode}, USA"

    # Let's get coordinates
    lat, lon = get_coordinates(full_address)


    # Let's get nearby places
    nearby_places = get_nearby_places(lat, lon) if lat and lon else None
    nearby_summary = summarize_nearby_places(nearby_places)

    # Let's choose the right model depending on whether the zipcode is known
    if data.zipcode in valid_zipcodes:

        # Convert zipcode to int for the model
        zipcode_int = int(data.zipcode)

        # Let's build the input using zipcode
        house_example = pd.DataFrame({
            "bedrooms": [data.bedrooms],
            "bathrooms": [data.bathrooms],
            "sqft_living": [data.sqft_living],
            "floors": [data.floors],
            "yr_built": [data.yr_built],
            "zipcode": [zipcode_int]
        })

        # Let's predict using the models trained with zipcode
        lr_price = estimate_price(model_zip, house_example)
        rf_price = estimate_price(rf_model_zip, house_example)

        # Let's keep track of which mode was used
        zipcode_mode = "known_zipcode"

        # Let's find comparable houses
        comps = find_comparables(df, house_example)

    else:

        # Let's build the input without zipcode
        house_example = pd.DataFrame({
            "bedrooms": [data.bedrooms],
            "bathrooms": [data.bathrooms],
            "sqft_living": [data.sqft_living],
            "floors": [data.floors],
            "yr_built": [data.yr_built]
        })

        # Let's predict using the models trained without zipcode
        lr_price = estimate_price(model_no_zip, house_example)
        rf_price = estimate_price(rf_model_no_zip, house_example)

        # Let's keep track of which mode was used
        zipcode_mode = "unknown_zipcode_used_general_model"

        # Let's skip comparables when zipcode is unknown
        comps = pd.DataFrame()


    # Let's give more importance to Random Forest
    predicted_price = (0.2 * lr_price) + (0.8 * rf_price)

    # Let's estimate price from comparables
    estimated_price_comps = estimate_price_from_comps(comps, house_example)

    # Let's combine both approaches
    if estimated_price_comps is None:
        final_estimated_price = predicted_price
    else:
        final_estimated_price = combine_prices(
            predicted_price,
            estimated_price_comps,
            len(comps)
        )

    # Let's calculate a location score based on nearby amenities
    location_score = calculate_location_score(nearby_summary)

    # Let's adjust the price using a multiplier (max ±10%)
    location_multiplier = 1 + (location_score - 0.5) * 0.2

    final_estimated_price *= location_multiplier

    # Let's calculate the price difference
    difference = (data.listing_price - final_estimated_price) / final_estimated_price * 100

    # Let's determine price status
    if difference > 5:
        price_status = "Overpriced"
    elif difference < -5:
        price_status = "Underpriced"
    else:
        price_status = "Fairly priced"

    # Let's estimate price drop risk
    price_drop_risk = estimate_price_drop_risk(difference, data.days_on_market)

    explanation = generate_explanation(
        difference,
        data.days_on_market,
        len(comps),
        location_score
    )

    return {
        "estimated_fair_price": round(final_estimated_price, 2),
        "listing_price": round(data.listing_price, 2),
        "price_difference_percent": round(difference, 2),
        "price_status": price_status,
        "price_drop_risk": price_drop_risk,
        "comparable_houses_found": int(len(comps)),
        "zipcode_mode": zipcode_mode,
        "explanation": explanation,
        "nearby_places": nearby_summary,

        # Let's return the raw Overpass response for debugging
        "nearby_raw_data": nearby_places,

        # Let's return how many raw places Overpass found before summarizing
        "raw_places_found": len(nearby_places["elements"]) if nearby_places and "elements" in nearby_places else 0,

        "latitude": lat,
        "longitude": lon,
    }

