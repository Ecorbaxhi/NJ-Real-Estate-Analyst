# Let's import pandas to work with the dataset
import pandas as pd

# Let's import Path so the app can find files correctly
from pathlib import Path

# Let's import the tools to split data and train the model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

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

# Let's take one sample house from the test data
house_example = X_test_app.iloc[[0]]

# Let's predict the fair price of that house
predicted_price = model_app.predict(house_example)[0]

# Let's set an example listing price
listing_price = 560000

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


# Let's find comparable houses for our example
comps = find_comparables(df, house_example)

print("Comparable houses found:", len(comps))
print("Average comparable price:", round(comps["price"].mean(), 2))

# Let's calculate price per sqft for comparables
comps["price_per_sqft"] = comps["price"] / comps["sqft_living"]

# Let's estimate price using average price per sqft
avg_price_sqft = comps["price_per_sqft"].mean()
estimated_price_comps = avg_price_sqft * house_example["sqft_living"].values[0]

print("Estimated price from comps:", round(estimated_price_comps, 2))