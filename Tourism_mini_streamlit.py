from math import e
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Streamlit app
# Load necessary files
model_classification = joblib.load("visit_mode_model.pkl")
model_regression = joblib.load("rating_model.pkl")
data = pd.read_csv("Tourism_Mini_data.csv")
mapping = pd.read_csv("merged_data.csv", encoding='utf-8-sig', on_bad_lines='skip')
mapping.columns = mapping.columns.str.strip() # Remove leading/trailing spaces from column names

st.set_page_config(page_title="Tourism Experience Analytics", layout="wide")
st.title("\U0001F3D6️ Tourism Experience Analytics")

# Sidebar Navigation
menu = ["Dashboard", "Visit Mode Predictor", "Attraction Rating Predictor", "Recommendations"]
choice = st.sidebar.selectbox("Select Module", menu)

# Function to predict visit mode (classification)
def predict_visit_mode(model,input_data):
    prediction = model.predict(input_data)
    visit_modes = {1:"Couple", 2:"Family", 3:"Friends", 4:"Solo"}
    prediction_index = int(round(prediction[0]))  
    if prediction_index < 1 or prediction_index > len(visit_modes):
        return "Unknown Visit Mode"
    return visit_modes[prediction_index]

# Function to predict attraction rating (regression)
def predict_rating(model, attraction, visit_mode):
    # Assuming these are the features for the regression model
    input_features = np.array([[attraction, visit_mode]])
    prediction = model.predict(input_features)
    return prediction[0]

# Function to get recommendations (simple content-based recommendation example)
def get_recommendations(user_id, data, top_n=5):
    # Filter data based on user preferences or behavior
    user_data = data[data['UserId'] == user_id]
    recommendations = user_data[['Attraction', 'Rating']].sort_values(by='Rating', ascending=False).head(top_n)
    return recommendations['Attraction'].tolist()

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
    attraction = st.selectbox("Attraction", data["Attraction"].unique())
    address = st.selectbox("Attraction Address", data["AttractionAddress"].unique())
    city_name = st.selectbox("City Name", data["CityName"].unique())
    continent_name = st.selectbox("Continent Name", data["Continent"].unique())
    country_name = st.selectbox("Country Name", data["Country"].unique())
    region_name = st.selectbox("Region Name", data["Region"].unique())

# # Filter for the user input to get the matching encoded row
    filtered_row = mapping[
        (mapping["UserId"] == user_id) &
        (mapping["Attraction"] == attraction) &
        (mapping["AttractionAddress"] == address) &
        (mapping["CityName"] == city_name) &
        (mapping["Continent"] == continent_name) &
        (mapping["Country"] == country_name) &
        (mapping["Region"] == region_name)
    ]  
      
    # if st.button("Predict Visit Mode"):  
    if filtered_row.empty:
            st.warning("No matching record found for selected inputs. Please check your selections.")
    else:
        # Select Only the Encoded Features Required by the Model 
        # Create a DataFrame from the first matching row
        matched_row = pd.DataFrame([filtered_row.iloc[0]]) 
        input_encoded = matched_row[[
                "UserId",  
                "Attraction_encoded",
                "AttractionAddress_encoded",
                "CityName_encoded",
                "Continent_encoded",
                "Country_encoded",
                "Region_encoded"
            ]].copy()  

        # Rename columns to match the training feature names
        input_encoded.columns = [
                'UserId', 
                'Attraction', 
                'AttractionAddress', 
                'Continent', 
                'Region',
                'Country', 
                'CityName',          
           ]

            # --- Predict Visit Mode ---
        result = predict_visit_mode(model_classification, input_encoded[["UserId", "Attraction", "AttractionAddress", "Continent", "Region", "Country", "CityName"]])
        st.success(f"Predicted Visit Mode: {result}")

elif choice == "Attraction Rating Predictor":
    st.subheader("\U00002B50 Predict Attraction Rating")
    # Sample inputs
    attraction = st.selectbox("Attraction", data["Attraction"].unique())
    visit_mode = st.selectbox("Visit Mode", data["VisitMode"].unique())

    filtered_row = mapping[
        (mapping["Attraction"] == attraction) &
        (mapping["VisitMode"] == visit_mode)
    ]
    
    if st.button("Predict Rating"):
        if filtered_row.empty:
            st.warning("No matching record found for selected inputs. Please check your selections.")
        else:
            # --- Select Only the Encoded Features Required by the Model ---
            matched_row = pd.DataFrame([filtered_row.iloc[0]])
            input_encoded = matched_row[[
                "Attraction_encoded",
                "VisitMode_encoded",
            ]].copy()

            input_encoded.columns = [
                'Attraction', 
                'VisitMode', 
            ]
            
        predicted_rating = predict_rating(model_regression, input_encoded["Attraction"].values[0], input_encoded["VisitMode"].values[0])
        st.success(f"Predicted Rating: {predicted_rating:.2f} / 5")

elif choice == "Recommendations":

    st.subheader("🔍 Personalized Recommendations")

    # --- Generate Recommendations ---
    def generate_recommendations(merged_df, target_index):
        merged_df['features'] = (
            merged_df['AttractionType'].astype(str) + ' ' +
            merged_df['CityName'].astype(str) + ' ' +
            merged_df['Region'].astype(str)
        )
    
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(merged_df['features'])
    
        nn = NearestNeighbors(n_neighbors=11, metric='cosine')
        nn.fit(tfidf_matrix)
    
        distances, indices = nn.kneighbors(tfidf_matrix[target_index])
        recommendations = merged_df.iloc[indices[0][1:]][['Attraction', 'CityName', 'Region']]
        return recommendations.reset_index(drop=True)

# Dropdown to select an attraction
    st.write("### User-Based Recommendations")
    selected_attraction = st.selectbox("Choose an attraction you liked:", data['Attraction'].unique())

# Recommend Button
    if st.button("Get Recommendations"):
       try:
         target_index = data[data['Attraction'] == selected_attraction].index[0]
         recommendations = generate_recommendations(data, target_index)

         st.subheader("🧭 Top 10 Similar Attractions:")
         st.table(recommendations)
       except Exception as e:
         st.error(f"Error: {e}")

