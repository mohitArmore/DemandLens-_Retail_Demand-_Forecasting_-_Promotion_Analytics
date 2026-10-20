# DemandLens: Streamlit app that predicts a product's demand from its price/promotion setup

import streamlit as st
import pandas as pd
import pickle
from xgboost import XGBRegressor

# Average demand in the training data, used as the benchmark for each prediction
AVG_DEMAND = 104

st.set_page_config(page_title="DemandLens", page_icon="📈", layout="centered")


# Load the model and encoders once and reuse them on every rerun (Streamlit reruns the script on each input change)
@st.cache_resource
def load_artifacts():
    # Native XGBoost format: more portable across library versions than pickle
    model = XGBRegressor()
    model.load_model("xgboost_demand_model.json")

    # Fitted LabelEncoders from ML.ipynb, used to turn Category names into the same codes the model was trained on
    with open("lable_encoders.pkl", "rb") as f:
        encoders = pickle.load(f)

    return model, encoders

model, label_encoders = load_artifacts()

# Sidebar: short context about the model for first-time users
with st.sidebar:
    st.header("About")
    st.write(
        "DemandLens estimates daily unit demand for a product from its pricing "
        "and promotion setup, so planners can test scenarios before committing stock."
    )
    st.markdown(
        "- **Model:** tuned XGBoost regressor\n"
        "- **Training data:** 76,000 store-product-day records\n"
        "- **Test error (RMSE):** ~36 units\n"
        "- **Top driver:** promotions (+30% average lift)"
    )

st.title("📈 DemandLens")
st.caption("Retail demand forecasting & promotion analytics: set up a scenario and see the expected demand.")

# Inputs are grouped in a form so the model runs only when the user submits
with st.form("scenario"):
    left, right = st.columns(2)

    with left:
        st.subheader("Product & stock")
        # Only offer categories the encoder has seen during training
        category = st.selectbox("Category", label_encoders["Category"].classes_.tolist())
        inventory_level = st.number_input("Inventory level (units)", min_value=0, value=300, step=10)
        promotion = st.toggle("Run a promotion", value=False)

    with right:
        st.subheader("Pricing")
        price = st.number_input("Price", min_value=0.0, value=65.0, step=1.0)
        competitor_pricing = st.number_input("Competitor price", min_value=0.0, value=65.0, step=1.0)
        # Training data only contains discounts of 0-25% in 5% steps
        discount = st.slider("Discount (%)", min_value=0, max_value=25, value=10, step=5)

    submitted = st.form_submit_button("Predict demand", type="primary", use_container_width=True)

if submitted:
    # Two rows, same scenario without and with a promotion, so we can also show the promotion lift.
    # Column names and order must match the training features exactly.
    input_data = pd.DataFrame({
        "Price": [price, price],
        "Discount": [discount, discount],
        "Inventory Level": [inventory_level, inventory_level],
        "Promotion": [0, 1],
        "Competitor Pricing": [competitor_pricing, competitor_pricing],
        "Category": [category, category],
    })

    # Convert text columns (Category) to the integer codes used in training
    for col, encoder in label_encoders.items():
        if col in input_data.columns:
            input_data[col] = encoder.transform(input_data[col])

    no_promo, with_promo = model.predict(input_data)
    prediction = with_promo if promotion else no_promo

    st.divider()
    st.subheader("Forecast")

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Predicted demand",
        f"{int(prediction)} units",
        delta=f"{prediction / AVG_DEMAND - 1:+.0%} vs. average",
    )
    c2.metric("Effective price", f"{price * (1 - discount / 100):.2f}", delta=f"-{discount}%", delta_color="off")
    c3.metric(
        "Promotion lift",
        f"{int(with_promo - no_promo):+d} units",
        delta=f"{with_promo / no_promo - 1:+.0%}" if no_promo else None,
    )

    # Simple stock check: flag when expected demand is higher than the inventory on hand
    if prediction > inventory_level:
        st.warning(
            f"Expected demand exceeds inventory by {int(prediction - inventory_level)} units. "
            "Consider replenishing to avoid lost sales."
        )
    else:
        st.success("Current inventory covers the expected demand.")

st.caption("Built with XGBoost & Streamlit · See README for methodology and analysis")
