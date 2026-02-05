import pickle

import yfinance
import pandas as pd
import numpy as np

import requests
from get_data import split_data


# list of all the compnaies that are in S&P500
url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'

header = {
  "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.75 Safari/537.36",
  "X-Requested-With": "XMLHttpRequest"
}

r = requests.get(url, headers=header)

sp_companies = pd.read_html(r.text)[0]
sp_companies['Date added'] = pd.DatetimeIndex(sp_companies['Date added'])

# only conisder companies that were added before or at 2004
candidate_comps = sp_companies[sp_companies['Date added'] <= '2004']

sectors = candidate_comps['GICS Sector'].unique()

asset_per_sector = {sector: candidate_comps[candidate_comps['GICS Sector']==sector] for sector in sectors}

assets_ticker = []
seed = 124
n_asset_per_sector = 3 # number of assets per sector to include

for sector_df in asset_per_sector.values():
    tickers = sector_df.sample(n_asset_per_sector, random_state=seed)['Symbol'].values

    assets_ticker.extend(tickers)


######### download the data of the selected tickers #########
start_date = "2005-01-01"
end_date = "2025-01-01"

data = yfinance.download(assets_ticker, start_date, end_date)

# number of na vals in the data
missing_data = len(data[data.isna().any(axis=1)])

# make sure there are no missing values
if missing_data > 0:
    raise ValueError(f"There are missing values in the dataset!")

# add the log return of each of the stocks
for stock in assets_ticker:
    data["Log return", stock]= np.log(data["Close"][stock]).diff()

######### save and split the data into train, val, test #########
file_name = 'new_dataset.pkl'

with open(file_name, 'wb') as f:
    pickle.dump(data, f)

split_data(file_name)
