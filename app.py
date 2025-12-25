import streamlit as st
import pandas as pd
import json
from catboost import CatBoostRegressor

# function that discribe how much the model is reliable for user 
def get_reliability(r2):
    if r2 >= 0.75:
        return "🟢 High Reliability"
    elif r2 >= 0.55:
        return "🟡 Medium Reliability"
    else:
        return "🔴 Low Reliability"


# Page config
st.set_page_config(
    page_title="Amazon Price Predictor",
    page_icon="🛒",
    layout="centered"
)

st.title("🛒 Amazon Product Price Predictor")
st.caption("Category-specific ML models with confidence scoring")

# Load model metadata
with open("model_info.json", "r") as f:
    MODEL_INFO = json.load(f)



# Load models once
@st.cache_resource
def load_models():
    models = {}
    for category, info in MODEL_INFO.items():
        model = CatBoostRegressor()
        model.load_model(info["file"])
        models[category] = model
    return models

MODELS = load_models()

# Category selection
category = st.selectbox(
    "Select Product Category",
    list(MODEL_INFO.keys())
)


# Inputs
brand = st.text_input("Brand Name", "Generic")

rating = st.slider("Product Rating", 1.0, 5.0, 4.0)

discount_percent = st.slider(
    "Discount Percentage (%)",
    min_value=0,
    max_value=90,
    value=10
)

# convert % → ratio (VERY IMPORTANT)
discount_ratio = discount_percent / 100.0

# Predict

if st.button("Predict Price 💰"):

    model = MODELS[category]
    confidence = MODEL_INFO[category]["confidence"]

    # EXACT SAME FEATURES AS TRAINING
    input_df = pd.DataFrame(
        [[str(brand), float(rating), float(discount_ratio)]],
        columns=["brand", "rating", "discount_ratio"]
    )

    predicted_price = model.predict(input_df)[0]
    # Confidence-based price range
    r2 = MODEL_INFO[category]["r2"]

    margin = predicted_price * (1 - r2)
    low_price = max(0, predicted_price - margin)
    high_price = predicted_price + margin


    st.success(f"💰 Predicted Price: ₹{predicted_price:,.0f}")

    st.info(
    f"📊 Expected Price Range: ₹{low_price:,.0f} – ₹{high_price:,.0f}")

    # st.metric("Prediction Confidence",f"{r2 * 100:.0f}%")

    reliability = get_reliability(r2)

    st.metric(
    "Model Reliability",
    reliability)
    
    st.caption(
    "Reliability is based on the model's R² score on unseen test data.")

