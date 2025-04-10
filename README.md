   Stock Market Forecasting Using LSTM:

This project presents a Streamlit web application that utilizes Long Short-Term Memory (LSTM) neural networks to predict stock market prices
based on historical data. The application allows users to upload stock data, visualize trends, train an LSTM model, and make future price predictions.

Features:
Data Upload: Users can upload a CSV file containing stock market data.

Data Visualization: The application plots the normalized closing prices over time for better insight into trends.

LSTM Model Training: Users can train an LSTM model on the uploaded data to learn patterns and make predictions.

Performance Metrics: The application displays Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) to evaluate the model's performance.

Future Price Prediction: Users can input specific stock statistics (e.g., previous close, open, high, low, volume) to predict future stock prices.

1)Installation:
To set up the application locally, follow these steps:

2)Clone the Repository:
git clone https://github.com/Tesa2035/Stock-Market-Forecasting3.git

3)Navigate to the Project Directory:
cd Stock-Market-Forecasting3

4)Install Required Dependencies:
Ensure you have Python installed. 
It's recommended to use a virtual environment:

python3 -m venv venv
source venv/bin/activate  

5)Then, install the required packages:
pip install -r requirements.txt

--Usage
*)Run the Application:
streamlit run app.py

*)Interact with the Application:

Upload Data: Upload a CSV file containing stock data. Ensure the CSV has columns like date, open, high, low, close, adjclose, and volume. The date column should be in a recognizable date format.

Visualize Data: After uploading, the application will display the first few rows of the dataset and plot the normalized closing prices over time.

Train LSTM Model: The application will preprocess the data, create sequences, and train an LSTM model. Training progress and performance metrics (MAE and RMSE) will be displayed.

Predict Future Prices: Input specific stock statistics to predict future prices. Enter values for previous close, open, high, low, and volume, then click the "Predict" button to see the forecasted price.

#Data Preparation
For optimal performance:

Data Format: Ensure the CSV file has the following columns: date, open, high, low, close, adjclose, and volume. The date column should be in YYYY-MM-DD format.

Missing Values: Handle any missing values before uploading. The application does not include preprocessing steps for missing data.

Data Frequency: Use daily stock data for consistency with the model's design.

#Model Details
Architecture: The LSTM model consists of two LSTM layers with 50 units each, followed by two Dense layers with 25 and 1 unit(s) respectively.

Input Sequence: The model uses sequences of 10 previous closing prices to predict the next closing price.

Normalization: Features are normalized using Min-Max scaling to improve training efficiency.

Screenshots

Figure 1: Normalized Closing Price Over Time


Figure 2: LSTM Model Training Progress


Figure 3: Future Price Prediction Interface

