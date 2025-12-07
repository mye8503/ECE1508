import pickle
import matplotlib.pyplot as plt

data = pickle.load(open("new_dataset.pkl", "rb"))

data2 = data['Close']
tickers = data2.columns.tolist()
print(tickers)
plt.figure(figsize=(12,6))
for column in data2.columns:
    plt.plot(data2.index, data2[column], label=column)


plt.xlabel('Date')
plt.ylabel('Stock Close Price')
plt.title('S&P 500 Stock Close Prices Over Time')

plt.legend()
plt.show()