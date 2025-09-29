📈 LSTM & GRU-Based Sales Forecasting

This project implements a time-series forecasting model using LSTM and GRU neural networks to predict future sales trends. The model is trained on historical sales data and helps businesses with demand planning.

🚀 Features

Uses LSTM and GRU for sequential data learning.

MinMax Scaling for data normalization.

Sliding window technique for time-series forecasting.

Train-test split to validate performance.

Loss function: MSE to minimize prediction errors.

Visualization of actual vs predicted sales.

📂 Project Structure

📁 Sales Forecasting Project
│── sales_data.csv              # Sales dataset
│── lstm_sales_forecasting.py   # Main ML model script
│── model/sales_forecasting_model.keras  # Trained model
│── README.md                   # Project documentation

🛠 Requirements

Install dependencies using:

pip install pandas numpy matplotlib tensorflow scikit-learn

📊 Dataset

The dataset sales_data.csv contains:

Sale_Date: Date of transaction

Sales_Amount: Total sales for the day

Other relevant features

🔧 How It Works

Load and preprocess data (convert date, normalize values, create sequences).

Build an LSTM-GRU model for sales forecasting.

Train the model with historical data.

Make predictions and compare actual vs predicted sales.

Visualize results to evaluate performance.

▶️ Running the Project

Run the script using:

python lstm_sales_forecasting.py

📈 Model Performance & Improvements

Hyperparameter tuning can improve accuracy.

Try Transformer-based models for better long-term forecasting.

Deploy using streamlit for real-time predictions.
 TRY THIS -- https://salesforecasting-6tcx3rd4up9hay9xmyzydz.streamlit.app/
📌 Future Work

Integrate the model into a web application.

Automate training & evaluation with MLOps.

🎯 This project helps businesses make data-driven sales forecasts and optimize supply chain management. 🚀


Author -- Moitri Dey
