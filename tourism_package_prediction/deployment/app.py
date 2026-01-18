import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# Download model from Hugging Face
model_path = hf_hub_download(
    repo_id="SuhasKashyap2703/churn-model",
    filename="best_churn_model_v1.joblib",
    repo_type="model"
)

# Load model
model = joblib.load(model_path)

st.title("Tourism Package Purchase Prediction")
st.write(
    "This application predicts whether a customer is likely to purchase the "
    "tourism package based on their profile and interaction details."
)


# User Inputs
Age = st.number_input("Age", min_value=18, max_value=100, value=35)
DurationOfPitch = st.number_input("Duration of Pitch (minutes)", min_value=0, value=15)
NumberOfTrips = st.number_input("Number of Trips per Year", min_value=0, value=2)
MonthlyIncome = st.number_input("Monthly Income", min_value=0.0, value=50000.0)

TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
CityTier = st.selectbox("City Tier", [1, 2, 3])
Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
Gender = st.selectbox("Gender", ["Male", "Female"])
NumberOfPersonVisiting = st.selectbox("Number of Persons Visiting", [1, 2, 3, 4, 5])
NumberOfFollowups = st.selectbox("Number of Follow-ups", [0, 1, 2, 3, 4, 5])
ProductPitched = st.selectbox(
    "Product Pitched",
    ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
)
PreferredPropertyStar = st.selectbox("Preferred Property Star", [3, 4, 5])
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
Passport = st.selectbox("Has Passport?", ["Yes", "No"])
PitchSatisfactionScore = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
OwnCar = st.selectbox("Owns a Car?", ["Yes", "No"])
NumberOfChildrenVisiting = st.selectbox("Number of Children Visiting", [0, 1, 2, 3])
Designation = st.selectbox(
    "Designation",
    ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
)


# Input DataFrame
input_data = pd.DataFrame([{
    "Age": Age,
    "DurationOfPitch": DurationOfPitch,
    "NumberOfTrips": NumberOfTrips,
    "MonthlyIncome": MonthlyIncome,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "Occupation": Occupation,
    "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfFollowups": NumberOfFollowups,
    "ProductPitched": ProductPitched,
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "Passport": 1 if Passport == "Yes" else 0,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "OwnCar": 1 if OwnCar == "Yes" else 0,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation
}])


# Prediction
classification_threshold = 0.5

if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = int(prediction_proba >= classification_threshold)

    if prediction == 1:
        st.success("The customer is likely to purchase the tourism package.")
    else:
        st.warning("The customer is unlikely to purchase the tourism package.")

    st.write(f"Prediction Probability: **{prediction_proba:.2f}**")
