import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load the trained model with a custom object
model = load_model("sales_forecasting_model.h5", compile=False)

# Recompile the model with the correct loss function
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')

# Load and preprocess dataset
file_path = "sales_data.csv"  # Update with correct path
df = pd.read_csv(file_path)
df['Sale_Date'] = pd.to_datetime(df['Sale_Date'])
df.set_index('Sale_Date', inplace=True)
daily_sales = df.groupby(df.index)['Sales_Amount'].sum()

# Normalize data
scaler = MinMaxScaler()
daily_sales_scaled = scaler.fit_transform(daily_sales.values.reshape(-1, 1))

# Define function to create sequences
sequence_length = 30
def create_sequences(data, seq_length):
    X = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
    return np.array(X)

# Prepare input data
X_input = create_sequences(daily_sales_scaled, sequence_length)
X_input = np.reshape(X_input, (X_input.shape[0], X_input.shape[1], 1))

# Streamlit App UI
st.title("📊 Sales Forecasting with LSTM & GRU")
st.write("Predict future sales trends using a trained deep learning model.")

# Predict future sales
if st.button("Predict Next Sales Day"):
    prediction = model.predict(X_input[-1].reshape(1, sequence_length, 1))
    predicted_sales = scaler.inverse_transform(prediction)[0][0]
    st.success(f"Predicted Sales for Next Day: {predicted_sales:.2f}")

# Plot actual vs predicted sales
st.subheader("📈 Actual vs Predicted Sales")
y_test_actual = scaler.inverse_transform(daily_sales_scaled[-len(X_input):])
predicted_sales_series = model.predict(X_input)
predicted_sales_series = scaler.inverse_transform(predicted_sales_series)

plt.figure(figsize=(10, 5))
plt.plot(y_test_actual, label='Actual Sales')
plt.plot(predicted_sales_series, label='Predicted Sales', linestyle='dashed')
plt.legend()
st.pyplot(plt)