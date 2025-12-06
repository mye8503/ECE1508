import yfinance
import pandas as pd
import numpy as np
import pickle
import os



def split_data(dataset_path, start_year='2005', end_year='2025', save_path='new_data'):
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)

    os.makedirs(f'{save_path}/train', exist_ok=True)
    os.makedirs(f'{save_path}/val', exist_ok=True)
    os.makedirs(f'{save_path}/test', exist_ok=True)

    # training
    for year in range(int(start_year), int(end_year)-6):
        start_date = f'{year}-01-01'
        end_date = f'{year+4}-12-31'

        data_split = data[start_date:end_date]

        with open(f'{save_path}/train/{year}-{year+4}.pkl', 'wb') as f:
            pickle.dump(data_split, f)

    # validation
    for year in range(int(start_year)+5, int(end_year)-1):
        start_date = f'{year}-01-01'
        end_date = f'{year}-12-31'

        data_split = data[start_date:end_date]

        with open(f'{save_path}/val/{year}.pkl', 'wb') as f:
            pickle.dump(data_split, f)

    # testing
    for year in range(int(start_year)+6, int(end_year)):
        start_date = f'{year}-01-01'
        end_date = f'{year}-12-31'

        data_split = data[start_date:end_date]

        with open(f'{save_path}/test/{year}.pkl', 'wb') as f:
            pickle.dump(data_split, f)

if __name__=='__main__':

    # stocks are: apple, google, microsoft, amazon, nvidia
    stock_list = ["AAPL", "GOOG", "MSFT", "AMZN", "JNJ"]

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

    split_data('new_dataset.pkl', start_year='2005', end_year='2025', save_path='new_data')

