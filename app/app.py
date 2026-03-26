# Let's import pandas to work with the dataset
import pandas as pd

# Let's import Path so the app can find files correctly
from pathlib import Path

# Let's import the tools to split data and train the model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Let's import Random Forest to test a stronger model
from sklearn.ensemble import RandomForestRegressor

# Let's import metrics to evaluate our models
from sklearn.metrics import mean_absolute_error, r2_score

# Let's create a function to estimate fair price using the model
def estimate_price(model, house):
    return model.predict(house)[0]

# Let's create a function to estimate price from comparable houses
def estimate_price_from_comps(comps, house):
    
    # Let's make sure we have comparable houses
    if len(comps) == 0:
        return None

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
def find_comparables(df, house, tolerance=0.15):

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

# Let's build the correct path to the dataset
data_path = Path(__file__).resolve().parent.parent / "data" / "kc_house_data.csv"

# Let's load the housing dataset
df = pd.read_csv(data_path)

# Let's keep only the variables the app will use
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

# Let's split the data into training and testing parts
X_train_app, X_test_app, y_train_app, y_test_app = train_test_split(
    X_app, y_app, test_size=0.2, random_state=42
)

# Let's train the price prediction model
model_app = LinearRegression()
model_app.fit(X_train_app, y_train_app)

# Let's train a Random Forest model for better predictions
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_app, y_train_app)

# Let's calculate feature importance from Random Forest
feature_importance = pd.Series(rf_model.feature_importances_, index=features_for_app)
feature_importance = feature_importance.sort_values(ascending=False)

print("\nTop factors influencing price:")
for feature, importance in feature_importance.items():
    print(f"{feature}: {round(importance * 100, 2)}%")

# Let's evaluate Linear Regression
lr_predictions = model_app.predict(X_test_app)
lr_mae = mean_absolute_error(y_test_app, lr_predictions)
lr_r2 = r2_score(y_test_app, lr_predictions)

# Let's evaluate Random Forest
rf_predictions = rf_model.predict(X_test_app)
rf_mae = mean_absolute_error(y_test_app, rf_predictions)
rf_r2 = r2_score(y_test_app, rf_predictions)

# Let's print model performance
print("Linear Regression MAE:", round(lr_mae, 2))
print("Linear Regression R2:", round(lr_r2, 3))

print("Random Forest MAE:", round(rf_mae, 2))
print("Random Forest R2:", round(rf_r2, 3))

# Let's ask the user to enter the house details
bedrooms = int(input("Enter number of bedrooms: "))
bathrooms = float(input("Enter number of bathrooms: "))
sqft_living = int(input("Enter square feet: "))
floors = float(input("Enter number of floors: "))
yr_built = int(input("Enter year built: "))
zipcode = int(input("Enter zipcode: "))
listing_price = float(input("Enter listing price: "))

# Let's turn the user input into a dataframe the model can read
house_example = pd.DataFrame({
    "bedrooms": [bedrooms],
    "bathrooms": [bathrooms],
    "sqft_living": [sqft_living],
    "floors": [floors],
    "yr_built": [yr_built],
    "zipcode": [zipcode]
})

# Let's predict price using Linear Regression
lr_price = estimate_price(model_app, house_example)

# Let's predict price using Random Forest
rf_price = estimate_price(rf_model, house_example)

# Let's give more importance to Random Forest based on better performance
# We weight models based on performance (Random Forest performs better)
predicted_price = (0.2 * lr_price) + (0.8 * rf_price)

print("Linear Regression price:", round(lr_price, 2))
print("Random Forest price:", round(rf_price, 2))

# Let's calculate how much the house is overpriced or underpriced
difference = (listing_price - predicted_price) / predicted_price * 100

# Let's show the result
print("Dataset loaded. Number of houses:", len(df))
print("App features prepared:", X_app.shape[1])
print("Model trained successfully")
print("Estimated fair price:", round(predicted_price, 2))
print("Listing price:", listing_price)
print("Price difference (%):", round(difference, 2))

if difference > 5:
    print("This house appears overpriced.")
elif difference < -5:
    print("This house appears underpriced.")
else:
    print("This house price looks fair.")


# Let's find comparable houses for our example
comps = find_comparables(df, house_example)

print("Comparable houses found:", len(comps))
print("Average comparable price:", round(comps["price"].mean(), 2))

# Let's estimate price using comparable houses
estimated_price_comps = estimate_price_from_comps(comps, house_example)

# Let's safely print comparable price
if estimated_price_comps is not None:
    print("Estimated price from comps:", round(estimated_price_comps, 2))
else:
    print("No comparable houses found. Using model only.")

# Let's calculate the final estimated price using smart weighting
# Let's handle the case where no comparables are found
if estimated_price_comps is None:
    final_estimated_price = predicted_price
else:
    final_estimated_price = combine_prices(predicted_price, estimated_price_comps, len(comps))

print("Final estimated price:", round(final_estimated_price, 2))

# Let's ask the user how long the house has been listed
days_on_market = int(input("Enter days on market: "))

# Let's estimate the final price drop risk
price_drop_risk = estimate_price_drop_risk(difference, days_on_market)

print("Price drop risk:", price_drop_risk)

