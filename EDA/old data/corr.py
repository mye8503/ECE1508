import pandas as pd

data = pd.read_csv('dataset.csv', header=[0,1], index_col=0)
correl_price = data['Close'].corr()

print(f'The co-relation coefficient between the assets is\n:{correl_price}')
correl_price.to_csv('pearson correlation.csv')