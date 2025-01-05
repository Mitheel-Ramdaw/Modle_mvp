import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# Simulate sample data (replace this with actual data)
def simulate_data():
    data = {
        'Numbers_Score': np.random.randint(50, 101, size=100),
        'Operations_Score': np.random.randint(50, 101, size=100),
        'Shapes_Score': np.random.randint(50, 101, size=100),
        'Measurement_Score': np.random.randint(50, 101, size=100),
        'Total_Score': np.random.randint(50, 101, size=100),
        'Year_End_Performance': np.random.randint(50, 101, size=100)
    }
    return pd.DataFrame(data)

# Load or simulate data
df = simulate_data()

# Split data into features and target
X = df[['Numbers_Score', 'Operations_Score', 'Shapes_Score', 'Measurement_Score', 'Total_Score']]
y = df['Year_End_Performance']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Ridge Regression model
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

# Streamlit app
st.title("Diagnostic Test Year-End Performance Predictor")
st.write("This app predicts a student's year-end performance based on diagnostic test scores.")

# User inputs
st.sidebar.header("Input Diagnostic Test Scores")
numbers_score = st.sidebar.slider("Numbers and Counting (%)", 0, 100, 50)
operations_score = st.sidebar.slider("Basic Operations (%)", 0, 100, 50)
shapes_score = st.sidebar.slider("Shapes and Space (%)", 0, 100, 50)
measurement_score = st.sidebar.slider("Measurement and Data Handling (%)", 0, 100, 50)
total_score = st.sidebar.slider("Total Diagnostic Score (%)", 0, 100, 50)

# Predict performance
if st.sidebar.button("Predict"):
    input_data = np.array([[numbers_score, operations_score, shapes_score, measurement_score, total_score]])
    input_data_scaled = scaler.transform(input_data)
    prediction = ridge.predict(input_data_scaled)
    st.write(f"Predicted Year-End Performance: **{prediction[0]:.2f}%**")

# Display evaluation metrics
st.header("Model Performance")
y_pred = ridge.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write(f"Mean Squared Error (MSE): {mse:.2f}")
st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
st.write(f"R² Score: {r2:.2f}")

# Display sample dataset
st.header("Sample Data")
st.write(df.head())
