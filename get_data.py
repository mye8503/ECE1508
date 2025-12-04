import yfinance
import pandas as pd
import numpy as np
import pickle

# stocks are: apple, google, microsoft, amazon, nvidia
# stock_list = ["AAPL", "META", "GOOG", "MSFT", "AMZN"]
# stock_list = ["AAPL", "GOOG", "MSFT", "AMZN", "NVDA"]
stock_list = ["AAPL", "JNJ", "MSFT", "AMZN", "GME"]

# 6000 thousand days of observations are collected, which is roughly equivalent to from January of 2005 to January of 2025.
start_date = "2005-01-01"
end_date = "2025-01-01"

# download the data
data = yfinance.download(stock_list, start=start_date, end=end_date)

# number of na vals in the data
missing_data = len(data[data.isna().any(axis=1)])

# make sure there are no missing values
if missing_data > 0:
    raise ValueError(f"There are missing values in the dataset!")

# add the log return of each of the stocks
for stock in stock_list:
    data["Log return", stock]= np.log(data["Close"][stock]).diff()

print(len(data))  # there are 5033 data points for each stock
print(data.columns)

# save as csv
# data.to_csv("new_dataset.csv")

with open('new_dataset.pkl', 'wb') as f:
    pickle.dump(data, f)