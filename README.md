# capstone_project_5

# Singapore Resale Price Predictor

## Overview

The Singapore Resale Price Predictor is a machine learning application that estimates the resale price of flats in Singapore based on historical transaction data. The application uses a Random Forest Regressor model to provide accurate predictions and is deployed as a web application using Streamlit.

## Project Goals

- **Predictive Modeling**: Develop a machine learning model to predict resale prices of flats based on various features.
- **User-Friendly Web App**: Create an interactive web application that allows users to input details and receive predicted resale prices.
- **Deployment**: Deploy the application on the Render platform for accessibility.

## Features

- Predict resale price based on features such as town, flat type, floor area, storey range, lease commence date, remaining lease, and flat model.
- Handle missing values and preprocess data for accurate predictions.
- User-friendly interface with Streamlit.

## Data Source

The data used in this project is sourced from the Singapore Housing and Development Board (HDB) and can be accessed here: [Singapore HDB Resale Data](https://beta.data.gov.sg/collections/189/view)

## Installation

### Prerequisites

- Python 3.8 or higher
- Required Python libraries: `pandas`, `numpy`, `scikit-learn`, `joblib`, `streamlit`

### Download the Dataset

Place the dataset CSV files in the DataSets directory as specified in the code.

## Usage

### Run the Streamlit App
streamlit run app.py

### Interact with the Web Application

Open the application in your web browser (usually at http://localhost:8501).

Input the details of the flat (e.g., town, flat type, floor area, etc.).

The application will display the predicted resale price based on the provided inputs.

## Model Training

The model is trained using historical resale data from various CSV files. The following steps are involved in training the model:

1,Load and merge CSV files from different years.
2,Preprocess the data, including handling missing values and converting categorical features.
3,Train a Random Forest Regressor model.
4,Evaluate model performance using metrics such as MAE, MSE, RMSE, and R² Score.
5,Save the trained model using joblib.
