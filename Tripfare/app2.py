import streamlit as st
import numpy as np
import joblib

st.markdown(
    """
    <style>
    .stApp {
        background-color: #1e1e2f;  /* Dark blue/purple */
        color: white;               /* Text color */
    }
    
    /* Target all labels of widgets */
    label {
        color: white !important;
        font-weight: 600;
    }
    
    /* Style all buttons */
    .stButton>button {
        background-color: #f9f871;  /* Taxi yellow */
        color: black;
        font-weight: 700;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        transition: background-color 0.3s ease;
    }
    /* Button hover effect */
    .stButton>button:hover {
        background-color: #e0d247;
        cursor: pointer;
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown('<h1 style="color:black;">🚕 NYC Taxi Fare Prediction</h1>', unsafe_allow_html=True)
# trained model
model = joblib.load('Downloads/GUVI Practise/Tripfare/best_gradient_boosting_model.pkl')

#  Mapping dictionaries for user-friendly labels 
payment_type_map = {
    1: "Credit Card",
    2: "Cash",
    3: "No Charge",
    4: "Dispute"
}

ratecodeid_map = {
    1: "Standard",
    2: "JFK",
    4: "N/A",
    5: "Group Ride",
    6: "Airport"
}

day_map = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday"
}

# Reverse maps for internal code lookup
reverse_payment_map = {v: k for k, v in payment_type_map.items()}
reverse_ratecode_map = {v: k for k, v in ratecodeid_map.items()}
reverse_day_map = {v: k for k, v in day_map.items()}

st.write("""
This app predicts the **total fare amount** for a taxi ride in NYC.
Fill in the trip details below and click **Predict**!
""")

# User inputs with friendly labels 
passenger_count = st.number_input("Number of Passengers", min_value=1, step=1)
trip_distance = st.number_input("Trip Distance (miles)", min_value=0.0, step=0.1)
hour = st.number_input("Pickup Hour (0–23)", min_value=0, max_value=23, step=1)
is_night = st.selectbox("Is it a night ride?", ["No", "Yes"])

ratecode_label = st.selectbox("RatecodeID", options=list(ratecodeid_map.values()))
payment_label = st.selectbox("Payment Type", options=list(payment_type_map.values()))
pickup_day_label = st.selectbox("Pickup Day", options=list(day_map.values()))

# Encode inputs 

# Binary: is_night
is_night_num = 1 if is_night == "Yes" else 0

# Map back from labels to numeric codes
ratecode = reverse_ratecode_map[ratecode_label]
payment = reverse_payment_map[payment_label]
pickup_day = reverse_day_map[pickup_day_label]

# One-hot encode RatecodeID (1,2,4,5,6)
ratecode_dummies = [0, 0, 0, 0, 0]  # Ratecode_1, 2, 4, 5, 6
ratecode_map = {1: 0, 2: 1, 4: 2, 5: 3, 6: 4}
if ratecode in ratecode_map:
    ratecode_dummies[ratecode_map[ratecode]] = 1

# One-hot encode Payment Type (1-4)
payment_dummies = [0, 0, 0, 0]
payment_map = {1: 0, 2: 1, 3: 2, 4: 3}
if payment in payment_map:
    payment_dummies[payment_map[payment]] = 1

# One-hot encode Pickup Day (0,1,3)
day_dummies = [0, 0, 0]
day_map_onehot = {0: 0, 1: 1, 3: 2}  # Note: only these 3 days were in training
if pickup_day in day_map_onehot:
    day_dummies[day_map_onehot[pickup_day]] = 1

# Final input feature vector 
final_features = [
    passenger_count,
    trip_distance,
    hour,
    is_night_num
] + ratecode_dummies + payment_dummies + day_dummies + [0] 

input_array = np.array([final_features])

# Predict on button click 
if st.button("Predict Total Fare"):
    prediction = model.predict(input_array)
    st.success(f"💲 Estimated Total Fare: ${prediction[0]:.2f}")
