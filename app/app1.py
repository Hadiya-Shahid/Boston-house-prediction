import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="Boston House Price Prediction", page_icon="🏠")
st.title("🏠 Boston House Price Prediction")
st.write("Enter the details below and get the predicted house price (in $1000s).")

# ✅ Step 1: Load Model (pickle version)
try:
    with open("best_rf_model1_pickle.pkl", "rb") as f:
        model = pickle.load(f)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Step 2: Collect Inputs
st.header("Enter House Features")

CRIM = st.number_input("CRIM (per capita crime rate by town)", min_value=0.0, step=0.1)
ZN = st.number_input("ZN (proportion of residential land zoned for lots over 25,000 sq.ft.)", min_value=0.0, step=1.0)
INDUS = st.number_input("INDUS (proportion of non-retail business acres per town)", min_value=0.0, step=0.1)
CHAS = st.selectbox("CHAS (Charles River dummy variable)", [0, 1])  # 0 = No, 1 = Yes
NOX = st.number_input("NOX (nitric oxides concentration)", min_value=0.0, step=0.01)
RM = st.number_input("RM (average number of rooms per dwelling)", min_value=0.0, step=0.1)
AGE = st.number_input("AGE (proportion of owner-occupied units built prior to 1940)", min_value=0.0, step=1.0)
DIS = st.number_input("DIS (weighted distances to employment centres)", min_value=0.0, step=0.1)
RAD = st.number_input("RAD (index of accessibility to radial highways)", min_value=1, step=1)
TAX = st.number_input("TAX (full-value property tax rate per $10,000)", min_value=0.0, step=1.0)
PTRATIO = st.number_input("PTRATIO (pupil-teacher ratio by town)", min_value=0.0, step=0.1)
B = st.number_input("B (1000(Bk - 0.63)^2 where Bk is proportion of Black residents)", min_value=0.0, step=1.0)
LSTAT = st.number_input("LSTAT (% lower status of the population)", min_value=0.0, step=0.1)

# Arrange input into array
input_data = np.array([[CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT]])

# Step 3: Prediction
if st.button("Predict Price"):
    try:
        prediction = model.predict(input_data)
        st.success(f"💰 Predicted House Price: {prediction[0]:.2f} (in $1000s)")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
