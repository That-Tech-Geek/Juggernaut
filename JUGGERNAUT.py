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

        # Find the column with date-time values in 'MM/DD/YYYY HH:MM:SS' or 'MM-DD-YYYY HH:MM:SS' format
        date_cols = [col for col in df.columns if df[col].astype(str).str.contains(r'^\d{1,2}[/-]\d{1,2}[/-]\d{4} \d{2}:\d{2}:\d{2}$', na=False).any()]

        if len(date_cols) > 0:
            st.write("Select a column with date-time values:")
            for i, col in enumerate(date_cols):
                st.write(f"{i+1}. {col}")

            col_idx = st.number_input("Enter the column number:", min_value=1, max_value=len(date_cols), value=1)
            selected_col = date_cols[col_idx - 1]

            # Convert the selected column to a datetime object
            df[selected_col] = pd.to_datetime(df[selected_col], errors='coerce', dayfirst=True)

            # Format the datetime object to 'DDMMYYYY'
            df[selected_col] = df[selected_col].dt.strftime('%d%m%Y')

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

        else:
            st.error("No columns with date-time values found.")

    except Exception as e:
        st.error(f"Error: {e}")
