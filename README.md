# Amazon Product Price Predictor (Category-Specific ML)

🔗 Live App: https://amazonpricepredictor-s7njza8xdhdhdikgbaejcz.streamlit.app/


# Project Overview
This project predicts the fair price of Amazon products using 
category-specific machine learning models built with CatBoost.

Instead of using one generic model, each product category 
has its own trained model for higher accuracy.

# FEATURES
- Category-wise price prediction
- Uses CatBoost Regressor
- Handles categorical data (brand) efficiently
- Confidence score based on model performance (R²)
- Streamlit-based interactive UI

# Categories Supported
✔ USB Cables (R² = 0.849)
✔ Smartwatches (R² = 0.507)


# Tech Stack
- Python
- Pandas
- CatBoost
- Streamlit
- Scikit-learn

# Input Features
- Brand
- Rating
- Discount Ratio

# How to Run
pip install -r requirements.txt
streamlit run app.py

# Project Architecture
Each product category has its own trained model (.cbm file).
Models are loaded dynamically based on user selection.

# Future Improvements
- Add more product categories
- Add real-time Amazon scraping
- Confidence interval visualization
- Deploy on cloud (Render / AWS)
