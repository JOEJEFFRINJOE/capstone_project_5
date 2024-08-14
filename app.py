import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import joblib
import streamlit as st

# Load and merge CSV files
def load_data():
    registration_files = [
        r"E:\GUVI\projects\Singapore flate resale\DataSets\ResaleFlatPricesBasedonRegistrationDateFromMar2012toDec2014.csv",
        r"E:\GUVI\projects\Singapore flate resale\DataSets\ResaleflatpricesbasedonregistrationdatefromJan2017onwards.csv",
        r"E:\GUVI\projects\Singapore flate resale\DataSets\ResaleFlatPricesBasedonRegistrationDateFromJan2015toDec2016.csv"
    ]
    
    approval_files = [
        r"E:\GUVI\projects\Singapore flate resale\DataSets\ResaleFlatPricesBasedonApprovalDate19901999.csv",
        r"E:\GUVI\projects\Singapore flate resale\DataSets\ResaleFlatPricesBasedonApprovalDate2000Feb2012.csv"
    ]
    
    registration_data = pd.concat([pd.read_csv(file) for file in registration_files], ignore_index=True)
    approval_data = pd.concat([pd.read_csv(file) for file in approval_files], ignore_index=True)
    
    registration_data['source'] = 'registration'
    approval_data['source'] = 'approval'
    
    data = pd.concat([registration_data, approval_data], ignore_index=True)
    return data

# Convert 'remaining_lease' to lease duration in years
def convert_remaining_lease(remaining_lease):
    if isinstance(remaining_lease, str):
        match = re.match(r'(\d+) years (\d+) months', remaining_lease)
        if match:
            years, months = map(int, match.groups())
            return years + months / 12
        else:
            return np.nan
    return np.nan

# Data Preprocessing
def preprocess_data(data):
    # Convert remaining_lease to numerical value
    data['remaining_lease'] = data['remaining_lease'].apply(convert_remaining_lease)
    
    # Handle missing values in 'remaining_lease' and 'lease_commence_date'
    imputer = SimpleImputer(strategy='mean')
    data[['remaining_lease', 'lease_commence_date']] = imputer.fit_transform(data[['remaining_lease', 'lease_commence_date']])
    
    # Drop rows with critical missing values in 'resale_price'
    data.dropna(subset=['resale_price'], inplace=True)
    
    # Fill remaining missing values
    data.fillna(method='ffill', inplace=True)
    
    # Convert categorical columns to numerical values
    data = pd.get_dummies(data, columns=['town', 'flat_type', 'flat_model', 'storey_range', 'source'])
    
    # Create 'flat_age' feature
    data['flat_age'] = 2024 - data['lease_commence_date']
    
    # Drop columns that should not be used as features
    X = data.drop(columns=['resale_price', 'block', 'street_name', 'month'])
    y = data['resale_price']
    
    return X, y

# Model Training
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Impute missing values in the training and test sets
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Evaluation metrics
    print(f"MAE: {mean_absolute_error(y_test, y_pred)}")
    print(f"MSE: {mean_squared_error(y_test, y_pred)}")
    print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False)}")
    print(f"R² Score: {r2_score(y_test, y_pred)}")
    
    # Save the model
    joblib.dump(model, 'resale_price_model.pkl')
    return model

# Streamlit App
def run_app():
    st.title('Singapore Resale Price Predictor')

    # Load the data and preprocess
    data = load_data()
    X, y = preprocess_data(data)

    # Train the model
    model = train_model(X, y)

    # User inputs
    town = st.selectbox('Town', sorted(data['town'].unique()))
    flat_type = st.selectbox('Flat Type', sorted(data['flat_type'].unique()))
    floor_area = st.number_input('Floor Area (sqm)', min_value=0)
    storey_range = st.selectbox('Storey Range', sorted(data['storey_range'].unique()))
    lease_commence_date = st.number_input('Lease Commence Date (Year)', min_value=1900, max_value=2024)
    remaining_lease = st.text_input('Remaining Lease (e.g., "70 years 02 months")')
    flat_model = st.selectbox('Flat Model', sorted(data['flat_model'].unique()))
    source = st.selectbox('Data Source', ['approval', 'registration'])

    # Process user input for prediction
    remaining_lease_value = convert_remaining_lease(remaining_lease)

    # Create input DataFrame
    input_data = pd.DataFrame({
        'town': [town],
        'flat_type': [flat_type],
        'floor_area_sqm': [floor_area],
        'storey_range': [storey_range],
        'lease_commence_date': [lease_commence_date],
        'remaining_lease': [remaining_lease_value],
        'flat_model': [flat_model],
        'source': [source]
    })
    input_data = pd.get_dummies(input_data)

    # Align input data with model features
    missing_cols = set(X.columns) - set(input_data.columns)
    for col in missing_cols:
        input_data[col] = 0
    input_data = input_data[X.columns]

    # Impute missing values in input data
    imputer = SimpleImputer(strategy='mean')
    input_data = imputer.fit_transform(input_data)

    # Predict resale price
    prediction = model.predict(input_data)
    st.write(f"Estimated Resale Price: ${prediction[0]:,.2f}")

if __name__ == "__main__":
    run_app()
