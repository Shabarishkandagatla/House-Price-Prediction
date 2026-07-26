import streamlit as st
import pandas as pd
import joblib

# -----------------------
# Page Configuration
# -----------------------
st.set_page_config(
    page_title="California House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# -----------------------
# Load Model
# -----------------------
model = joblib.load("models/house_price_model.pkl")

# -----------------------
# Title
# -----------------------
st.title("🏠 California House Price Prediction")
st.write("Predict California house prices using Machine Learning.")

st.divider()

# -----------------------
# Layout
# -----------------------
col1, col2 = st.columns(2)

# ==========================
# LEFT COLUMN
# ==========================
with col1:

    st.subheader("📍 Location")

    locations = {

        "🌉 Bay Area": {
            "San Francisco": (-122.4194, 37.7749),
            "San Jose": (-121.8863, 37.3382),
            "Oakland": (-122.2712, 37.8044),
            "Berkeley": (-122.2730, 37.8715),
            "Fremont": (-121.9886, 37.5483),
            "Palo Alto": (-122.1430, 37.4419),
            "Mountain View": (-122.0839, 37.3861),
            "Sunnyvale": (-122.0363, 37.3688),
            "Cupertino": (-122.0322, 37.3229)
        },

        "🌴 Southern California": {
            "Los Angeles": (-118.2437, 34.0522),
            "San Diego": (-117.1611, 32.7157),
            "Anaheim": (-117.9143, 33.8366),
            "Long Beach": (-118.1937, 33.7701),
            "Irvine": (-117.8265, 33.6846),
            "Santa Monica": (-118.4912, 34.0195),
            "Malibu": (-118.7798, 34.0259)
        },

        "🌲 Central California": {
            "Sacramento": (-121.4944, 38.5816),
            "Fresno": (-119.7871, 36.7378),
            "Bakersfield": (-119.0187, 35.3733),
            "Modesto": (-120.9969, 37.6391),
            "Stockton": (-121.2908, 37.9577)
        },

        "🌊 Coastal California": {
            "Santa Barbara": (-119.6982, 34.4208),
            "Monterey": (-121.8947, 36.6002),
            "Santa Cruz": (-122.0308, 36.9741)
        }
    }

    region = st.selectbox(
        "🌍 Select Region",
        list(locations.keys())
    )

    city = st.selectbox(
        "📍 Select City",
        list(locations[region].keys())
    )

    longitude, latitude = locations[region][city]

    st.info(f"📌 Selected: {city}")

    ocean_proximity = st.selectbox(
        "🌊 Ocean Proximity",
        [
            "<1H OCEAN",
            "INLAND",
            "ISLAND",
            "NEAR BAY",
            "NEAR OCEAN"
        ]
    )

# ==========================
# RIGHT COLUMN
# ==========================
with col2:

    st.subheader("🏡 Property Details")

    housing_median_age = st.slider(
        "Housing Median Age (Years)",
        1,
        52,
        25
    )

    total_rooms = st.slider(
        "Total Rooms",
        10,
        50,
        4
    )

    total_bedrooms = st.slider(
        "Total Bedrooms",
        0,
        30,
        3
    )

    population = st.slider(
        "Neighborhood Population",
        10,
        100,
        5
    )

    households = st.slider(
        "Number of Households",
        1,
        100,
        5
    )

    median_income = st.slider(
        "Median Household Income (× $10,000/year)",
        0.5,
        15.0,
        4.0,
        0.1
    )

st.divider()

# ==========================
# Prediction
# ==========================
if st.button("🔮 Predict House Price", use_container_width=True):

    input_df = pd.DataFrame({

        "longitude": [longitude],
        "latitude": [latitude],
        "housing_median_age": [housing_median_age],
        "total_rooms": [total_rooms],
        "total_bedrooms": [total_bedrooms],
        "population": [population],
        "households": [households],
        "median_income": [median_income],
        "ocean_proximity": [ocean_proximity]

    })

    prediction = model.predict(input_df)

    st.success("✅ Prediction Completed Successfully!")

    st.metric(
        "🏠 Estimated House Price",
        f"${prediction[0]:,.0f}"
    )