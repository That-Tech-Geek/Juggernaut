import streamlit as st
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error

# Create a file uploader
uploaded_file = st.file_uploader("Choose a CSV or XLSX file", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        # Read the uploaded file
        if uploaded_file.type == "text/csv":
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            df = pd.read_excel(uploaded_file)

        # Find the column with date-time values in 'MM/DD/YYYY HH:MM:SS' format
        date_cols = [col for col in df.columns if df[col].astype(str).str.contains(r'^\d{1,2}/\d{1,2}/\d{4} \d{2}:\d{2}:\d{2}$', na=False).any()]

        if len(date_cols) > 0:
            st.write("Select a column with date-time values:")
            for i, col in enumerate(date_cols):
                st.write(f"{i+1}. {col}")

            col_idx = st.number_input("Enter the column number:", min_value=1, max_value=len(date_cols), value=1)
            selected_col = date_cols[col_idx - 1]

            try:
                # Convert the selected column to a datetime object with '/' as delimiter
                df[selected_col] = pd.to_datetime(df[selected_col], errors='coerce', dayfirst=True, format='%m/%d/%Y %H:%M:%S')
            except ValueError:
                try:
                    # Convert the selected column to a datetime object with '-' as delimiter
                    df[selected_col] = pd.to_datetime(df[selected_col], errors='coerce', dayfirst=True, format='%m-%d-%Y %H:%M:%S')
                except ValueError:
                    st.error("Invalid date format. Please use 'MM/DD/YYYY HH:MM:SS' or 'MM-DD-YYYY HH:MM:SS' format.")

            # Extract year, month, day, hour, minute, second as separate features
            df['year'] = df[selected_col].dt.year
            df['month'] = df[selected_col].dt.month
            df['day'] = df[selected_col].dt.day
            df['hour'] = df[selected_col].dt.hour
            df['minute'] = df[selected_col].dt.minute
            df['second'] = df[selected_col].dt.second

            # Create a new column in the format DDMMYYYY
            df['date_num'] = (df['day'] * 1000000) + (df['month'] * 10000) + df['year']

            # Drop the original date-time column and the separate date features
            df.drop([selected_col, 'year', 'onth', 'day'], axis=1, inplace=True)

            # Display the dataframe
            st.write(df.head())

            # Ask the user to select an attribute to increase or decrease
            attribute = st.selectbox("Select an attribute to increase or decrease", df.columns)

            # Ask the user to select the direction (increase or decrease)
            direction = st.selectbox("Select the direction", ["Increase", "Decrease"])

            # Prepare the data for Gradient Boosting
            if attribute == 'date_num':
                st.error("Cannot select 'date_num' as the attribute to increase or decrease.")
            else:
                X = df.drop([attribute], axis=1)
                y = df[attribute]

                # Split the data into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Create a Gradient Boosting model
                model = GradientBoostingRegressor(random_state=42)

                # Train the model
                model.fit(X_train, y_train)

                # Make predictions on the test set
                y_pred = model.predict(X_test)

                # Evaluate the model
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                st.write(f'RMSE: {rmse:.2f}')

                # Plot the feature importance
                feature_importances = model.feature_importances_
                feature_importances_df = pd.DataFrame({'feature': X.columns, 'importance': feature_importances})
                feature_importances_df.sort_values(by='importance', ascending=False, inplace=True)
                st.write(feature_importances_df)

                # Plot the feature importance graph
                plt.figure(figsize=(10, 6))
                sns.barplot(x='importance', y='feature', data=feature_importances_df)
                plt.title('Feature Importance')
                plt.xlabel('Importance')
                plt.ylabel('Feature')
                st.pyplot()

                # Generate recommendations
                if direction == "Increase":
                    recommendations = model.feature_importances_.argsort()[-10:][::-1]
                else:
                    recommendations = model.feature_importances_.argsort()[:10]

                # Display the recommendations
                st.write("Recommendations:")
                for i, rec in enumerate(recommendations):
                    st.write(f"{i+1}. {X.columns[rec]}")
