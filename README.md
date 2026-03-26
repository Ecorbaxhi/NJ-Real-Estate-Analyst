# NJ-Real-Estate-Analyst
NJ Real Estate Analyst is an application that analyzes residential property listings in New Jersey to help users understand whether a house is fairly priced or overpriced.
The system compares a selected property with similar homes in the same area and estimates a fair market value based on key characteristics such as size, location, property type, and other listing details. It also analyzes signals like how long the property has been on the market and other market indicators to estimate the likelihood that the listing price may drop in the near future. The goal of this project is to apply data analysis and machine learning techniques to support better decision-making for home buyers, sellers, and investors.


## Project Overview

NJ Real Estate Analyst is a data-driven application designed to evaluate residential properties and support real estate decision-making using machine learning and comparative market analysis.

The application answers two key questions:
1. Is a house overpriced or underpriced?
2. How likely is the price to drop in the near future?

Users can input property details such as bedrooms, bathrooms, square footage, location, and listing price. The system then estimates a fair market value, compares it to the listing price, and provides actionable insights.

---

## Key Features

### 1. Fair Price Estimation
The app uses a machine learning model trained on historical housing data to estimate the fair value of a property based on its characteristics.

### 2. Comparable Homes Analysis
The system identifies similar properties (comparables) based on:
- Location (zipcode)
- Size (square footage range)
- Bedrooms and bathrooms
- Year built

It then calculates an alternative price estimate using average price per square foot.

### 3. Smart Price Evaluation
The final estimated price combines:
- Machine learning prediction
- Comparable homes analysis

The weighting is adaptive:
- More comparables → rely more on market data
- Fewer comparables → rely more on the model
- No comparables → rely fully on the model

### 4. Overpricing Detection
The app calculates the percentage difference between:
- Listing price
- Estimated fair price

It classifies the property as:
- Overpriced
- Underpriced
- Fairly priced

### 5. Price Drop Risk Prediction
The system evaluates the likelihood of a price drop using:
- Degree of overpricing
- Days on market

Risk levels:
- LOW
- MEDIUM
- HIGH

---

## Methodology

The project combines two approaches:

- Machine Learning (Linear Regression, with potential extension to Random Forest, XGBoost)
- Rule-based Comparative Market Analysis

This hybrid approach improves robustness and reflects real-world real estate practices.

---

## Dataset

The current version uses a public housing dataset (~21,000 observations) containing:
- Price
- Square footage
- Bedrooms / bathrooms
- Floors
- Year built
- Zipcode

The dataset is used for model training and prototyping.

---

## Future Improvements

- Integration with real-time data (Zillow, MLS APIs)
- AI-powered search for comparable properties online
- Advanced models (Random Forest, Gradient Boosting)
- Time-on-market prediction models
- Web interface (Streamlit or FastAPI)

---

## Scope

Although the project is named "NJ Real Estate Analyst", the current model is trained on a U.S. housing dataset and is designed as a general framework.

The system can be adapted to any U.S. region by:
- Using local datasets
- Integrating region-specific market data

---

## Conclusion

NJ Real Estate Analyst demonstrates how machine learning and data analysis can be applied to real estate valuation and investment decisions, providing a scalable and practical tool for identifying mispriced properties and anticipating market behavior.
