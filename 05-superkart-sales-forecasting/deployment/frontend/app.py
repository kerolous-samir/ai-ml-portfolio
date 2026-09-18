import streamlit as st
import requests

# 🔧 Backend URL:
BACKEND_URL = "https://kerolous-samir-superkart-backend.hf.space"  # <-- CHANGE THIS WHEN YOU DEPLOY

st.title("SuperKart Sales Prediction")
st.write(
    "Enter the product and store details below to estimate the sales revenue "
    "for a specific product in a store."
)

# ------------------ Input fields ------------------
product_id = st.text_input("Product ID", value="FD6114")
product_weight = st.number_input("Product Weight (kg)", min_value=0.0, value=12.5)
product_sugar_content = st.selectbox(
    "Product Sugar Content", ["low sugar", "regular", "no sugar"]
)
product_allocated_area = st.number_input(
    "Product Allocated Area (ratio)", min_value=0.0, max_value=1.0, value=0.03
)
product_type = st.selectbox(
    "Product Type",
    [
        "meat",
        "snack foods",
        "hard drinks",
        "dairy",
        "canned",
        "soft drinks",
        "health and hygiene",
        "baking goods",
        "bread",
        "breakfast",
        "frozen foods",
        "fruits and vegetables",
        "household",
        "seafood",
        "starchy foods",
        "others",
    ],
)
product_mrp = st.number_input(
    "Product MRP (currency units)", min_value=0.0, value=150.0
)

store_id = st.text_input("Store ID", value="OUT010")
store_establishment_year = st.number_input(
    "Store Establishment Year", min_value=1900, max_value=2025, value=2002, step=1
)
store_size = st.selectbox("Store Size", ["High", "Medium", "Low"])
store_location_city_type = st.selectbox(
    "Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"]
)
store_type = st.selectbox(
    "Store Type",
    [
        "Departmental Store",
        "Supermarket Type1",
        "Supermarket Type2",
        "Food Mart",
    ],
)

# ------------------ Call backend ------------------
if st.button("Predict Sales"):
    payload = [
        {
            "Product_Id": product_id,
            "Product_Weight": product_weight,
            "Product_Sugar_Content": product_sugar_content,
            "Product_Allocated_Area": product_allocated_area,
            "Product_Type": product_type,
            "Product_MRP": product_mrp,
            "Store_Id": store_id,
            "Store_Establishment_Year": int(store_establishment_year),
            "Store_Size": store_size,
            "Store_Location_City_Type": store_location_city_type,
            "Store_Type": store_type,
        }
    ]

    try:
        response = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=30)

        if response.status_code == 200:
            data = response.json()

            if "predictions" in data and len(data["predictions"]) > 0:
                prediction = data["predictions"][0]
                st.success(f"Predicted sales revenue: {prediction:.2f}")
            else:
                st.error(f"Unexpected response format: {data}")

        else:
            st.error(f"Error {response.status_code}: {response.text}")

    except Exception as e:
        st.error(f"An error occurred while contacting the backend: {e}")
