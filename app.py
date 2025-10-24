import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import io

# ----------------------------
# 🔹 Load Model from Google Drive
# ----------------------------
# Example Google Drive link:
# https://drive.google.com/file/d/1AbCdEfGh12345/view?usp=sharing
# → File ID = 1AbCdEfGh12345

FILE_ID = "1aSaewL3H6aW59yleryYrwAqu3FtzUfgb"
@st.cache_resource
def load_model_from_drive(file_id):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(url)
    if response.status_code != 200:
        st.error("❌ Failed to load model from Google Drive. Check file ID or access.")
        return None
    model = joblib.load(io.BytesIO(response.content))
    return model

ensemble_model = load_model_from_drive(FILE_ID)
st.success("✅ Model loaded successfully from Google Drive!") if ensemble_model else None


# ----------------------------
# 🔹 UI Setup
# ----------------------------
st.title("🌲 Forest Cover Type Prediction App")
st.write("Enter the environmental parameters below to predict the forest cover type.")

# Numeric Inputs
Elevation = st.slider("Elevation (m)", 2000, 4000, 2600)
Aspect = st.slider("Aspect (°)", 0, 360, 100)
Slope = st.slider("Slope (°)", 0, 60, 10)
Horizontal_Distance_To_Hydrology = st.slider("Horizontal Distance to Hydrology (m)", 0, 5000, 200)
Vertical_Distance_To_Hydrology = st.slider("Vertical Distance to Hydrology (m)", -100, 500, 20)
Horizontal_Distance_To_Roadways = st.slider("Horizontal Distance to Roadways (m)", 0, 5000, 1000)
Hillshade_9am = st.slider("Hillshade 9am", 0, 255, 210)
Hillshade_Noon = st.slider("Hillshade Noon", 0, 255, 230)
Hillshade_3pm = st.slider("Hillshade 3pm", 0, 255, 150)
Horizontal_Distance_To_Fire_Points = st.slider("Horizontal Distance to Fire Points (m)", 0, 7000, 700)

# Categorical Inputs
Wilderness = st.selectbox("Wilderness Area", ["Unknown", "1", "2", "3", "4"])
Soil = st.selectbox("Soil Type", ["Unknown"] + [str(i) for i in range(1, 41)])


# ----------------------------
# 🔹 Derived Features
# ----------------------------
Mean_Hillshade = (Hillshade_9am + Hillshade_Noon + Hillshade_3pm) / 3
Road_Fire_Diff = abs(Horizontal_Distance_To_Roadways - Horizontal_Distance_To_Fire_Points)
Hydro_Road_Diff = abs(Horizontal_Distance_To_Hydrology - Horizontal_Distance_To_Roadways)
Elevation_Slope_Ratio = Elevation / (Slope + 1)

# Wilderness One-Hot
Wilderness_bin = [0]*4
if Wilderness != "Unknown":
    Wilderness_bin[int(Wilderness)-1] = 1

# Soil One-Hot
Soil_bin = [0]*40
if Soil != "Unknown":
    Soil_bin[int(Soil)-1] = 1

# Combine into DataFrame
data_dict = {
    "Elevation": [Elevation],
    "Aspect": [Aspect],
    "Slope": [Slope],
    "Horizontal_Distance_To_Hydrology": [Horizontal_Distance_To_Hydrology],
    "Vertical_Distance_To_Hydrology": [Vertical_Distance_To_Hydrology],
    "Horizontal_Distance_To_Roadways": [Horizontal_Distance_To_Roadways],
    "Hillshade_9am": [Hillshade_9am],
    "Hillshade_Noon": [Hillshade_Noon],
    "Hillshade_3pm": [Hillshade_3pm],
    "Horizontal_Distance_To_Fire_Points": [Horizontal_Distance_To_Fire_Points],
    "Mean_Hillshade": [Mean_Hillshade],
    "Road_Fire_Diff": [Road_Fire_Diff],
    "Hydro_Road_Diff": [Hydro_Road_Diff],
    "Elevation_Slope_Ratio": [Elevation_Slope_Ratio],
}

for i in range(4):
    data_dict[f"Wilderness_Area{i+1}"] = [Wilderness_bin[i]]
for i in range(40):
    data_dict[f"Soil_Type{i+1}"] = [Soil_bin[i]]

input_df = pd.DataFrame(data_dict)

# ----------------------------
# 🔹 Predict
# ----------------------------
if st.button("Predict Forest Cover Type"):
    if ensemble_model is not None:
        try:
            prediction = ensemble_model.predict(input_df)[0]
            cover_mapping = {
                1: "Spruce/Fir",
                2: "Lodgepole Pine",
                3: "Ponderosa Pine",
                4: "Cottonwood/Willow",
                5: "Aspen",
                6: "Douglas-fir",
                7: "Krummholz"
            }
            st.success(f"🌲 Predicted Forest Cover Type: **{prediction} - {cover_mapping[int(prediction)]}**")
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
    else:
        st.error("❌ Model not loaded. Please check your Google Drive File ID.")

