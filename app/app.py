# Let's import the libraries we need for the backend and machine learning
from pathlib import Path
import pandas as pd

# Let's import FastAPI to create our web application
from fastapi import FastAPI

# Let's import BaseModel to define the input structure
from pydantic import BaseModel

# Let's import tools for model training
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Let's create the FastAPI app
app = FastAPI(title="NJ Real Estate Analyst API")


# Let's define the structure of the input data coming from the user
class HouseInput(BaseModel):
    bedrooms: int
    bathrooms: float
    sqft_living: int
    floors: float
    yr_built: int
    zipcode: int
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


# Let's create a function to estimate price drop risk
def estimate_price_drop_risk(difference, days_on_market):
    if difference > 10 or days_on_market > 60:
        return "HIGH"
    elif difference > 5 or days_on_market > 30:
        return "MEDIUM"
    else:
        return "LOW"


# Let's find comparable houses based on similar features
def find_comparables(df, house):

    zipcode = house["zipcode"].values[0]
    sqft = house["sqft_living"].values[0]
    bedrooms = house["bedrooms"].values[0]
    bathrooms = house["bathrooms"].values[0]
    year = house["yr_built"].values[0]

    comps = df[
        (df["zipcode"] == zipcode) &
        (df["sqft_living"].between(sqft * 0.85, sqft * 1.15)) &
        (df["bedrooms"].between(bedrooms - 1, bedrooms + 1)) &
        (df["bathrooms"].between(bathrooms - 0.5, bathrooms + 0.5)) &
        (df["yr_built"].between(year - 10, year + 10))
    ]

    return comps.copy()


# Let's load the dataset from the data folder
data_path = Path(__file__).resolve().parent.parent / "data" / "kc_house_data.csv"
df = pd.read_csv(data_path)


# Let's select only the features the app will use
features_for_app = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "floors",
    "yr_built",
    "zipcode"
]

X_app = df[features_for_app]
y_app = df["price"]


# Let's split the dataset into training and testing
X_train_app, X_test_app, y_train_app, y_test_app = train_test_split(
    X_app, y_app, test_size=0.2, random_state=42
)


# Let's train the Linear Regression model
model_app = LinearRegression()
model_app.fit(X_train_app, y_train_app)


# Let's train the Random Forest model for better performance
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_app, y_train_app)


# Let's create a simple route to check if the API is running
@app.get("/")
def home():
    return {"message": "NJ Real Estate Analyst API is running"}


# Let's create the main prediction route that receives user input
@app.post("/predict")
def predict_house(data: HouseInput):

    # Let's convert user input into a dataframe
    house_example = pd.DataFrame({
        "bedrooms": [data.bedrooms],
        "bathrooms": [data.bathrooms],
        "sqft_living": [data.sqft_living],
        "floors": [data.floors],
        "yr_built": [data.yr_built],
        "zipcode": [data.zipcode]
    })

    # Let's predict price using both models
    lr_price = estimate_price(model_app, house_example)
    rf_price = estimate_price(rf_model, house_example)

    # Let's give more importance to Random Forest
    predicted_price = (0.2 * lr_price) + (0.8 * rf_price)

    # Let's find comparable houses
    comps = find_comparables(df, house_example)

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

    # Let's return the results to the user
    return {
        "estimated_fair_price": round(final_estimated_price, 2),
        "listing_price": round(data.listing_price, 2),
        "price_difference_percent": round(difference, 2),
        "price_status": price_status,
        "price_drop_risk": price_drop_risk,
        "comparable_houses_found": int(len(comps))
    }
    