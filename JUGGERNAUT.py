import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# Create a file uploader
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded file into a pandas dataframe
    dataset = pd.read_csv(uploaded_file)

    # Ask for attribute to operate on and direction
    attribute = st.selectbox("Select an attribute to operate on", dataset.columns)
    direction = st.selectbox("Do you want to increase or decrease the attribute?", ["increase", "decrease"])

    # Split dataset into features and target
    X = dataset.drop(columns=[attribute])
    y = dataset[attribute]

    # Train gradient boosting model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    st.write(f"Model trained with MSE: {mse:.2f}")

    # Propose solutions using gradient boosting
    if direction == "increase":
        feature_importances = model.feature_importances_
        top_features = feature_importances.argsort()[-5:][::-1]
        st.write(f"To increase {attribute}, focus on the following features:")
        for feature in top_features:
            st.write(f"  - {X.columns[feature]}")
    elif direction == "decrease":
        feature_importances = model.feature_importances_
        bottom_features = feature_importances.argsort()[:5]
        st.write(f"To decrease {attribute}, focus on the following features:")
        for feature in bottom_features:
            st.write(f"  - {X.columns[feature]}")
