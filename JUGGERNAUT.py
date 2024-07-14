import streamlit as st
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

# Create a file uploader
uploaded_file = st.file_uploader("Choose a CSV or XLSX file", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        # Read the uploaded file
        if uploaded_file.type == "text/csv":
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            df = pd.read_excel(uploaded_file)

        # Find the column with date-time values
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    pd.to_datetime(df[col], errors='coerce')
                    date_time_col = col
                    break
                except ValueError:
                    pass
            else:
                continue

        # Convert the date-time column to a datetime object
        df[date_time_col] = df[date_time_col].str.split('T').str[0]
        
        try:
            df[date_time_col] = df[date_time_col].str.replace('-', '').str.replace('/', '')
        except AttributeError:
            print("Error: Unable to remove delimiters. Check the column data type.")
        # Display the dataframe
        st.write(df.head())

        # Ask the user to select an attribute to increase or decrease
        attribute = st.selectbox("Select an attribute to increase or decrease", df.columns)

        # Ask the user to select the direction (increase or decrease)
        direction = st.selectbox("Select the direction", ["Increase", "Decrease"])

        # Prepare the data for Gradient Boosting
        X = df.drop(attribute, axis=1)
        y = df[attribute]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

        # Create a Gradient Boosting model
        model = GradientBoostingRegressor()

        # Train the model
        model.fit(X_train, y_train)

        # Generate recommendations
        if direction == "Increase":
            recommendations = model.feature_importances_.argsort()[-10:][::-1]
        else:
            recommendations = model.feature_importances_.argsort()[:10]

        # Display the recommendations
        st.write("Recommendations:")
        for i, rec in enumerate(recommendations):
            st.write(f"{i+1}. {X.columns[rec]}")

    except Exception as e:
        st.error(f"Error: {e}")
