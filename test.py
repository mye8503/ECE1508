import pickle
from env_test import TestStockTradingEnv

data = pickle.load(open("dataset.pkl", "rb"))

# print(data['Log return', 'GOOG'])

x = TestStockTradingEnv(dataset_path="data/train/2005-2009.pkl")

state, _ = x.reset()

print(state['current_date'])

done = False
count = 0

up  = 0
down = 0

while not done:
    action = [0.2, 0.2, 0.2, 0.2, 0.2]  # equal weighting
    state, reward, done, _ = x.step(action)

    if reward > 0: up += 1
    if reward < 0: down += 1

    # if count % 50 == 0:
    #     for item in state:
    #         print(item)
    #         print(state[item])
    #         input()

    # if count % 100 == 0: print("current date:", state['current_date'], "|| reward:", reward)
    count += 1

print(up, down, count)
