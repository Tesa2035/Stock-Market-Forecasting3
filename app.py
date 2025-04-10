import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# App title
st.set_page_config(page_title="Stock Forecasting App", layout="wide")
st.title("📈 Stock Market Prediction using LSTM")

# File Upload
uploaded_file = st.file_uploader("Upload a stock price CSV file", type=["csv"])

if uploaded_file:
    # Load and preprocess the data
    df = pd.read_csv(uploaded_file)
    df.rename(columns={"Unnamed: 0": "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # Normalize numeric features
    numeric_cols = ["open", "high", "low", "close", "adjclose", "volume"]
    scaler = MinMaxScaler(feature_range=(0, 1))
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Plot normalized closing price
    st.subheader("📉 Normalized Closing Price Over Time")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["date"], df["close"], label="Close", color="blue")
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized Price")
    ax.legend()
    st.pyplot(fig)

    # Prepare data for LSTM
    def create_sequences(data, time_step=10):
        X, y = [], []
        for i in range(len(data) - time_step):
            X.append(data[i:i+time_step, 0])
            y.append(data[i + time_step, 0])
        return np.array(X), np.array(y)

    data = df[["close"]].values
    time_step = 10
    X, y = create_sequences(data, time_step)

    # Train/test split
    split = int(len(X) * 0.9)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    # Reshape for LSTM [samples, time_steps, features]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # LSTM model architecture
    st.subheader("🧠 Training LSTM Model")
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X_train, y_train, epochs=5, batch_size=16, verbose=1)

    # Predictions
    y_pred = model.predict(X_test)

    # Reverse scale predictions
    y_pred_inv = scaler.inverse_transform(
        np.concatenate([y_pred, np.zeros((len(y_pred), len(numeric_cols) - 1))], axis=1)
    )[:, 0]

    y_test_inv = scaler.inverse_transform(
        np.concatenate([y_test.reshape(-1, 1), np.zeros((len(y_test), len(numeric_cols) - 1))], axis=1)
    )[:, 0]

    # Evaluation
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))

    st.subheader("📌 Model Evaluation")
    st.write(f"**MAE:** {mae:.4f}")
    st.write(f"**RMSE:** {rmse:.4f}")

    # Plot actual vs predicted
    st.subheader("📊 Actual vs Predicted Prices")
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(df["date"].iloc[-len(y_test):], y_test_inv, label="Actual", color="blue")
    ax2.plot(df["date"].iloc[-len(y_test):], y_pred_inv, label="Predicted", color="red", linestyle="--")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Stock Price")
    ax2.legend()
    st.pyplot(fig2)

    # Optional: Predict future price using user input
    st.subheader("🔮 Predict Future Price from Input")
    prev_close = st.number_input("Previous Close", min_value=0.0)
    open_price = st.number_input("Open", min_value=0.0)
    high_price = st.number_input("High", min_value=0.0)
    low_price = st.number_input("Low", min_value=0.0)
    volume = st.number_input("Volume", min_value=0.0)

    if st.button("Predict Next Price"):
        user_input = [open_price, high_price, low_price, prev_close, 0.0, volume]  # added adjclose = 0.0
        input_scaled = scaler.transform([user_input])
        sequence = input_scaled[:, [3]].reshape(1, time_step, 1)  # Use the 'close' value only

        # Just repeat last 10 closes for demo prediction
        sequence = np.repeat(sequence, time_step, axis=1)  # shape: (1, 10, 1)

        prediction_scaled = model.predict(sequence)

        # Pad to inverse transform
        predicted_price = scaler.inverse_transform(
            np.concatenate((prediction_scaled, np.zeros((1, len(numeric_cols) - 1))), axis=1)
        )[0, 0]

        st.success(f"📈 Predicted Price: **{predicted_price:.2f}**")
