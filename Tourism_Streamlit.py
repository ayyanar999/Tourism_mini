from email.headerregistry import Address
import re
import attr
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load necessary files
model_classification = joblib.load("visit_mode_model.pkl")
model_regression = joblib.load("rating_model.pkl")
data = pd.read_csv("Tourism_Mini_data.csv")
mapping = pd.read_csv("merged_data.csv")
data_encoded = pd.read_csv("Tourism_Mini.csv") 

st.set_page_config(page_title="Tourism Experience Analytics", layout="wide")
st.title("\U0001F3D6️ Tourism Experience Analytics")

# Sidebar Navigation
menu = ["Dashboard", "Visit Mode Predictor", "Attraction Rating Predictor", "Recommendations"]
choice = st.sidebar.selectbox("Select Module", menu)

# Function to predict visit mode (classification)
def predict_visit_mode(model,input_data):
    prediction = model.predict(input_data)
    visit_modes = ["Business", "Family", "Couple", "Friends"]
    return visit_modes[prediction[0]]

# Function to predict attraction rating (regression)
def predict_rating(model, attraction, visit_mode, price, time_spent):
    # Assuming these are the features for the regression model
    input_features = np.array([[attraction, visit_mode, price, time_spent]])
    prediction = model.predict(input_features)
    return prediction[0]

# Function to get recommendations (simple content-based recommendation example)
def get_recommendations(user_id, data, top_n=5):
    # Filter data based on user preferences or behavior
    user_data = data[data['user_id'] == user_id]
    recommendations = user_data[['attraction_name', 'rating']].sort_values(by='rating', ascending=False).head(top_n)
    return recommendations['attraction_name'].tolist()

# Function to create a simple dashboard
def load_dashboard(data):
    st.write("### Tourism Data Overview")
    st.write(data.head())
    st.write(f"Total attractions: {len(data['Attraction'].unique())}")
    st.write(f"Total users: {data['UserId'].nunique()}")
    st.write(f"Average rating: {data['Rating'].mean():.2f}")

# Handling different pages based on sidebar selection
if choice == "Dashboard":
    st.subheader("\U0001F4CA Tourism Dashboard")
    load_dashboard(data)

elif choice == "Visit Mode Predictor":
    user_id = st.selectbox("Select User ID", data["UserId"].unique())
    attraction_id = st.selectbox("Enter Attraction ID",data["AttractionTypeId"].unique())
    # rating = st.slider("Enter Rating (1-5)",1, 5)
    attraction = st.selectbox("Attraction", data["Attraction"].unique())
    address = st.selectbox("Attraction Address", data["AttractionAddress"].unique())
    city_id = st.selectbox("City ID", data["CityId"].unique())
    city_name = st.selectbox("City Name", data["CityName"].unique())
    continent = st.selectbox("Continent", data["ContinentId_x"].unique())
    attraction_city = st.selectbox("Attraction City", data["AttractionCityId"].unique())
    region = st.selectbox("Region", data["RegionId_y"].unique())
    country = st.selectbox("Country", data["CountryId_y"].unique())
    continent_name = st.selectbox("Continent Name", data["Continent"].unique())
    country_name = st.selectbox("Country Name", data["Country"].unique())
    region_name = st.selectbox("Region Name", data["Region"].unique())
    attraction_type = st.selectbox("Attraction Type", data["AttractionType"].unique())
    type_id = st.selectbox("Attraction Type ID", data["AttractionTypeId"].unique())

    input_data = pd.DataFrame({
    'UserId': [user_id],
    'AttractionId': [attraction_id],
    # 'Rating': [rating],
    'Attraction': [attraction],
    'AttractionAddress': [address],
    'CityID': [city_id],
    'AttractionType': [attraction_type],
    'AttractionTypeID': [type_id],
    'CityName': [city_name],
    'Continent': [continent],
    'AttractionCityId': [attraction_city],
    'RegionId': [region],
    'CountryId': [country],
    'ContinentName': [continent_name],
    'CountryName': [country_name],
    'RegionName': [region_name]
    })
    
    attraction_dict = dict(zip(mapping["Attraction"], mapping["Attraction_Encoded"]))
    input_data["Attraction"] = input_data["Attraction"].map(attraction_dict)
    attraction_type_dict = dict(zip(mapping["AttractionType"], mapping["AttractionType_Encoded"]))
    input_data["AttractionType"] = input_data["AttractionType"].map(attraction_type_dict)
    attraction_address_dict = dict(zip(mapping["AttractionAddress"], mapping["AttractionAddress_Encoded"]))
    input_data["AttractionAddress"] = input_data["AttractionAddress"].map(attraction_address_dict)
    
    if st.button("Predict Visit Mode"):
        result = predict_visit_mode(model_classification,input_data)
        st.success(f"Predicted Visit Mode: {result}")

elif choice == "Attraction Rating Predictor":
    st.subheader("\U00002B50 Predict Attraction Rating")
    # Sample inputs
    attraction = st.selectbox("Attraction", data["Attraction"].unique())
    visit_mode = st.selectbox("Visit Mode", data["VisitMode"].unique())
    price = st.slider("Price Level", 1, 5, 3)
    time_spent = st.slider("Time Spent (hours)", 1, 10, 2)

    if st.button("Predict Rating"):
        predicted_rating = predict_rating(model_regression, attraction, visit_mode, price, time_spent)
        st.success(f"Predicted Rating: {predicted_rating:.2f} / 5")

elif choice == "Recommendations":
    st.subheader("\U0001F4A1 Personalized Recommendations")
    user_id = st.selectbox("Select User ID", data["UserId"].unique())
    top_n = st.slider("Number of Recommendations", 1, 10, 5)

    if st.button("Get Recommendations"):
        recommendations = get_recommendations(user_id, data, top_n=top_n)
        st.write("### Recommended Attractions:")
        for idx, rec in enumerate(recommendations, start=1):
            st.write(f"{idx}. {rec}")
