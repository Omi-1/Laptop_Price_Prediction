# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Sample dataset with features
data = {
    'Brand': ['Dell', 'Apple', 'HP', 'Asus', 'Lenovo'],
    'Processor': ['i5', 'i7', 'i5', 'Ryzen 5', 'i7'],
    'RAM': [8, 16, 8, 16, 12],
    'Storage': [512, 256, 1024, 512, 1024],  # in GB
    'GPU': ['NVIDIA', 'AMD', 'NVIDIA', 'NVIDIA', 'AMD'],
    'Screen_Size': [15.6, 13.3, 15.6, 14.0, 15.6],  # in inches
    'Price': [700, 1500, 800, 950, 1100]  # in USD
}

# Convert the data to a DataFrame
df = pd.DataFrame(data)

# Feature and label separation
X = df.drop('Price', axis=1)  # Features
y = df['Price']  # Target variable

# Convert categorical features using one-hot encoding
X = pd.get_dummies(X, drop_first=True)

# Ensure X_train is a DataFrame
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training a RandomForestRegressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predicting on the test set
y_pred = model.predict(X_test)

# Evaluating the model
#r2 = r2_score(y_test, y_pred)

# Streamlit app
st.title("Laptop Price Predictor")

#st.write(f"Model R^2 Score: {r2:.2f}")

# Input widgets for user interaction
brand = st.selectbox('Brand', ['Dell', 'Apple', 'HP', 'Asus', 'Lenovo'])
processor = st.selectbox('Processor', ['i5', 'i7', 'Ryzen 5'])
ram = st.slider('RAM (GB)', 4, 32, step=4, value=8)
storage = st.selectbox('Storage (GB)', [256, 512, 1024])
gpu = st.selectbox('GPU', ['NVIDIA', 'AMD'])
screen_size = st.selectbox('Screen Size (inches)', [13.3, 14.0, 15.6, 17.3])

# Function to predict laptop price based on user input
def predict_laptop_price(brand, processor, ram, storage, gpu, screen_size):
    # Create a DataFrame for the input
    input_data = pd.DataFrame({
        'Brand': [brand],
        'Processor': [processor],
        'RAM': [ram],
        'Storage': [storage],
        'GPU': [gpu],
        'Screen_Size': [screen_size]
    })

    # One-hot encode the input to match training features
    input_data_encoded = pd.get_dummies(input_data, drop_first=True)

    # Align the new input data with the model's expected input features
    input_data_encoded = input_data_encoded.reindex(columns=X_train.columns, fill_value=0)

    # Predict the price
    predicted_price = model.predict(input_data_encoded)
    
    return predicted_price[0]

# Button to trigger prediction
if st.button('Predict Price'):
    predicted_price = predict_laptop_price(brand, processor, ram, storage, gpu, screen_size)
    st.success(f"Predicted Price: ${predicted_price:.2f}")
