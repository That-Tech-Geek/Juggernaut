import streamlit as st
import xgboost as xgb
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd

# Load the Boston housing dataset
boston = load_boston()
X, y = boston.data, boston.target

# Convert y to a Pandas Series
y = pd.Series(y)

# Check for missing values in y
if y.isnull().sum() > 0:
    # Impute missing values with the mean
    y.fillna(y.mean(), inplace=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert y_train to a NumPy array
y_train_num = pd.to_numeric(y_train, errors='coerce')

# Train an XGBoost model
model = xgb.XGBRegressor()
model.fit(X_train, y_train_num)

# Serialize the model using joblib
joblib.dump(model, 'xgb_model.joblib')

# Create a Streamlit app
st.title("XGBoost Model")

# Load the serialized model
model = joblib.load('xgb_model.joblib')

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Visualize the results
st.write("Mean Squared Error:", mean_squared_error(y_test, y_pred))
st.write("Feature Importances:")
st.write(model.feature_importances_)
