import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("Fizer_Stock.csv")

data = data[["Date", "Close"]]
data["Date"] = pd.to_datetime(data.Date)
data["Close"].plot(figsize=(12, 8), title="Apple Stock Prices", fontsize=20, label="Close Price")
plt.legend()
plt.grid()
plt.show()
plt.savefig('close.png')

from autots import AutoTS
model = AutoTS(forecast_length=10, frequency='infer', 
               ensemble='simple', drop_data_older_than_periods=200)
model = model.fit(data, date_col='Date', value_col='Close', id_col=None)

prediction = model.predict()
forecast = prediction.forecast
print("Stock Price Prediction of Fizer")
print(forecast)