import matplotlib.pyplot as plt
import pickle

if __name__ == "__main__":
    with open('new_dataset.pkl', 'rb') as f:
        data = pickle.load(f)

    x = data.index
    y = data['Close']
    print(y)

    plt.plot(x,y)
    # plt.plot(data['steps'], data['val_rewards'], label='Validation Reward')
    plt.xlabel('Time')
    plt.ylabel('Stock Close Price')
    plt.title('Stock Close Price Over Time')
    plt.legend(['AAPL', 'AMZN', 'GME', 'JNJ', 'MSFT'])
    plt.grid()
    plt.show()