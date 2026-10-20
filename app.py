# Streamlit app: enter a product's price/promotion setup and get a predicted demand

import streamlit as st 
import pandas as pd 
import numpy as np 
import pickle
from xgboost import XGBRegressor


# Load the model and encoders once and reuse them on every rerun (Streamlit reruns the script on each input change)
@st.cache_resource
def load_artifacts():
    # Native XGBoost format: more portable across library versions than pickle
    model = XGBRegressor()
    model.load_model("xgboost_demand_model.json")

    # Fitted LabelEncoders from ML.ipynb, used to turn Category names into the same codes the model was trained on
    with open("lable_encoders.pkl" , "rb") as f:
        encoders = pickle.load(f)

    return model, encoders

model , label_encoders = load_artifacts()

st.title("Demand Forecasting App ")

st.divider() 

# User inputs: the same six features the model was trained on
st.header("Input Features")
price = st.number_input("Price" , min_value = 0.0 , value = 50.0)
discount = st.number_input("Discount (%)", min_value=0 , max_value=100 , value = 10 )
inventory_level = st.number_input("Inventory Level" , min_value = 0 , value = 100)
promotion = st.selectbox("promotion" , [0,1])
competitior_pricing = st.number_input("Competitor Price" , min_value= 0.0 , value = 10.0)


# Only offer categories the encoder has seen during training
category = st.selectbox(
    "Category",
    label_encoders["Category"].classes_.tolist()
)

# One-row dataframe; column names and order must match the training features exactly
input_data = pd.DataFrame({
    "Price" : [price],
    "Discount" : [discount],
    "Inventory Level" : [inventory_level],
    "Promotion" :[promotion],
    "Competitor Pricing": [competitior_pricing],
    "Category" : [category]
})

# Convert text columns (Category) to the integer codes used in training
for col, encoder in label_encoders.items():
    if col in input_data.columns:
        input_data[col] = encoder.transform(input_data[col])


st.divider()

# Run the model only when the user clicks the button
if st.button("Predict Demand"):

    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Demand : {int(prediction)} Units")