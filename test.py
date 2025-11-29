import pickle
from env_test import TestStockTradingEnv

data = pickle.load(open("dataset.pkl", "rb"))

# print(data['Log return', 'GOOG'])

x = TestStockTradingEnv()

state, _ = x.reset()

for item in state: 
    print(item)
    print(state[item])