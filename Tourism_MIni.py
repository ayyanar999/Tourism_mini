from re import S
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np
import joblib

# File paths
files = {
       "City": "City.xlsx",
       "Country": "Country.xlsx",
       "Continent": "Continent.xlsx",
       "Transaction": "Transaction.xlsx",
       "Type": "Type.xlsx",
       "User": "User.xlsx",
       "Item": "Updated_Item.xlsx",
       "Mode": "Mode.xlsx",
       "Region": "Region.xlsx"
}

# Convert CSV file
for name, path in files.items():
    csv_path = path.replace('.xlsx', '.csv')
    df = pd.read_excel(path, engine='openpyxl')
    df.to_csv(csv_path, index=False)

# Load data
dfs = {name: pd.read_csv(path.replace('.xlsx', '.csv')) for name, path in files.items()}

# Merge User Data
merged_df = dfs["Transaction"].merge(dfs["User"], on="UserId", how="left")

# Change Field Name in Transaction Data
dfs["Transaction"].rename(columns={"VisitMode": "VisitModeId"}, inplace=True)
merged_df.rename(columns={"VisitMode": "VisitModeId"}, inplace=True)

# Merge Item Data
merged_df = merged_df.merge(dfs["Item"], on="AttractionId", how="left")

# Merge Region,Country and Continent Data
merged_df = merged_df.merge(dfs["Continent"], on="ContinentId", how="left")
merged_df = merged_df.merge(dfs["Region"], on="RegionId", how="left")
merged_df = merged_df.merge(dfs["Country"], on="CountryId", how="left")

# Merge City Data
merged_df = merged_df.merge(dfs["City"], on="CityId", how="left")

# Convert Data Types
merged_df["AttractionTypeId"] = merged_df["AttractionTypeId"].astype("int64")
dfs["Type"]["AttractionTypeId"] = dfs["Type"]["AttractionTypeId"].astype("int64")

# Merge Type Data
merged_df = merged_df.merge(dfs["Type"], on="AttractionTypeId", how="left")

# Merge Visit Mode Data
merged_df = merged_df.merge(dfs["Mode"], on="VisitModeId", how="left")

merged_df.to_csv("Tourism_Mini_data.csv", index=False)

# Convert catergorical data to numerical data

# Encode categorical columns (Label Encoding or One-Hot Encoding)

encoder = LabelEncoder()

categorical_cols = ["Continent", "Country", "Region", "CityName", 
                    "AttractionType","Attraction","AttractionAddress","VisitMode"]

existing_cats = [col for col in categorical_cols if col in merged_df.columns]  # Only encode existing columns

for col in existing_cats:
    merged_df[col] = encoder.fit_transform(merged_df[col])

# === Load original and encoded CSVs ===
original_df = pd.read_csv("Tourism_Mini_data.csv")
encoded_df = pd.read_csv("Tourism_Mini.csv")

# === Define categorical columns you want to merge ===
categorical_cols = [
    "Continent", "Country", "Region", "CityName",
    "AttractionType", "Attraction", "AttractionAddress", "VisitMode"
]

# === Prepare renamed columns for encoded CSV ===
encoded_renamed = {
    col: f"{col}_Encoded" for col in categorical_cols if col in encoded_df.columns
}
encoded_df = encoded_df.rename(columns=encoded_renamed)

# === Merge original and encoded side-by-side ===
merged_df = pd.concat([original_df, encoded_df[list(encoded_renamed.values())]], axis=1)

# === Save the merged file ===
merged_df.to_csv("merged_data.csv", index=False)

# --------------------------------------------------------------------------------------------

# Regression model(Predict attraction Rating)

# Select features and  target
X = merged_df.drop(columns=["Rating"]) 
y = merged_df["Rating"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Model Evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
# --------------------------------------------------------------------------------------------

# Classification model(Predict attraction type)

# Select features and  target
XC = merged_df.drop(columns=["VisitMode"])
yC = merged_df["VisitMode"]

# Train the model
XC_train, XC_test, yC_train, yC_test = train_test_split(XC, yC, test_size=0.2, random_state=42)

# Model training
modelC = RandomForestRegressor(n_estimators=100, random_state=42)
modelC.fit(XC_train, yC_train)

# Make predictions
yC_pred = modelC.predict(XC_test)

# Model Evaluation
accuracy = accuracy_score(yC_test, yC_pred)
report = classification_report(yC_test, yC_pred)

# --------------------------------------------------------------------------------------------

# Reccomendation (Personalized Attraction Suggestion)

# User-Based Collaborative Filtering

# Load dataset 
ratings = merged_df.pivot_table(index='UserId', columns='AttractionId', values='Rating')

# Fill missing values with 0
ratings.fillna(0, inplace=True)

# Compute similarity between users
user_similarity = cosine_similarity(ratings)

# Convert to DataFrame
user_sim_df = pd.DataFrame(user_similarity, index=ratings.index, columns=ratings.index)

def recommend_for_user(user_id, num_recommendations=5):
    if user_id not in user_sim_df.index:
        return "User ID not found!"
    
    similar_users = user_sim_df[user_id].sort_values(ascending=False)[1:num_recommendations+1].index
    recommendations = ratings.loc[similar_users].mean().sort_values(ascending=False).index[:num_recommendations]
    
    return merged_df[merged_df['AttractionId'].isin(recommendations)][['AttractionId','Attraction']]

# Example
print(recommend_for_user(20))

# Attraction based on user rating

# Transpose matrix to make attractions as rows
item_ratings = ratings.T

# Compute similarity between attractions
item_similarity = cosine_similarity(item_ratings)

# Convert to DataFrame
item_sim_df = pd.DataFrame(item_similarity, index=item_ratings.index, columns=item_ratings.index)

def recommend_similar_attractions(attraction_id, num_recommendations=5):
    if attraction_id not in item_sim_df.index:
        return "Attraction ID not found!"
    
    similar_attractions = item_sim_df[attraction_id].sort_values(ascending=False)[1:num_recommendations+1].index
    return merged_df[merged_df['AttractionId'].isin(similar_attractions)][['AttractionId']]

# Example
print(recommend_similar_attractions(481))

# Content-Based Filtering 

# Create a combined feature column
merged_df['features'] = merged_df['AttractionType'] + ' ' + merged_df['CityName'] + ' ' + merged_df['Region']

# Convert text features into TF-IDF vectors
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(merged_df['features'])

# Compute similarity between attractions
content_similarity = cosine_similarity(tfidf_matrix)

# Convert to DataFrame
content_sim_df = pd.DataFrame(content_similarity, index=merged_df['AttractionId'], columns=merged_df['AttractionId'])

def recommend_attractions_by_content(attraction_id, num_recommendations=5):
    if attraction_id not in content_sim_df.index:
        return "Attraction ID not found!"
    
    similar_attractions = content_sim_df[attraction_id].sort_values(ascending=False)[1:num_recommendations+1].index
    return merged_df[merged_df['AttractionId'].isin(similar_attractions)][['AttractionId', 'Attraction']]

# Save the models
joblib.dump(model, "rating_model.pkl")


