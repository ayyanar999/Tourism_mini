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
from sklearn.neighbors import NearestNeighbors

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

# Load original and encoded CSVs 
original_df = pd.read_csv("Tourism_Mini_data.csv")
encoded_df = pd.read_csv("Tourism_Mini.csv")

# Define categorical columns you want to encode 
categorical_cols_to_encode = ["Continent", "Country", "Region", "CityName", 
                              "AttractionType", "Attraction", "AttractionAddress", "VisitMode"]

# Encode categorical columns
for col in categorical_cols_to_encode:
    if col in merged_df.columns:
        merged_df[col] = encoder.fit_transform(merged_df[col])

# Save the encoded file
encoded_df.to_csv("encoded_data.csv", index=False)

# --------------------------------------------------------------------------------------------

# Regression model(Predict attraction Rating)
# Select features and  target
X = merged_df.drop(columns=["Rating","AttractionId",
                              "CityId","AttractionTypeId",
                              "AttractionCityId",
                              "TransactionId",
                              "VisitMonth","VisitYear",
                              "ContinentId_x","CountryId_x",
                              "RegionId_x","ContinentId_y",
                              "CountryId_y","RegionId_y","CityName",
                              "AttractionType","UserId",
                              "AttractionAddress",
                              "Continent","Country",
                              "Region","AttractionTypeId",
                              "VisitModeId",]) 
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

# Save the model
joblib.dump(model, "rating_model.pkl")

# --------------------------------------------------------------------------------------------

# Classification model(Predict attraction type)
# Select features and  target
XC = merged_df.drop(columns=["VisitMode","AttractionId",
                              "CityId","AttractionTypeId",
                              "AttractionCityId",
                              "TransactionId","Rating",
                              "VisitMonth","VisitYear",
                              "ContinentId_x","CountryId_x",
                              "RegionId_x","ContinentId_y",
                              "CountryId_y","RegionId_y",
                              "VisitModeId",
                              "AttractionType"])
yC = merged_df["VisitMode"]

# Train the model
XC_train, XC_test, yC_train, yC_test = train_test_split(XC, yC, test_size=0.2, random_state=42)

# Model training
modelC = RandomForestRegressor(n_estimators=100, random_state=42)
modelC.fit(XC_train, yC_train)

# Make predictions
yC_pred = modelC.predict(XC_test)

# Model Evaluation
rmse_classification = np.sqrt(mean_squared_error(yC_test, yC_pred))
r2_classification = r2_score(yC_test, yC_pred)

# Save the model
joblib.dump(modelC, "visit_mode_model.pkl")
