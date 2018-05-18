import pandas as pd
from datetime import datetime
from pydatastream import Datastream
from utils import write_pickle
from private import DS_USERNAME, DS_PASSWORD


DWE = Datastream(username=DS_USERNAME, password=DS_PASSWORD)


def get_prices(mnem, date_from, date_to):
    template = '{}(RI)~~USD'.format(mnem)
    data = DWE.fetch(template, date_from=date_from, date_to=date_to)
    return data['P'].rename(mnem)


holdings = pd.read_csv('assets/holdings.csv')
date_from = datetime(2016, 12, 31)
date_to = datetime(2017, 12, 31)
prices = []
exceptions = []
for i, row in holdings.iterrows():
    try:
        print(i/len(holdings))
        prices.append(get_prices(row['ISIN'], date_from, date_to))
    except Exception as error:
        print(error)
        print(row['ISIN'])
        exceptions.append(row['ISIN'])

print('Re-download exceptions')
for isin in exceptions:
    try:
        print(i/len(exceptions))
        prices.append(get_prices(isin, date_from, date_to))
    except Exception as error:
        print(error)
        print(isin)
raw_prices = pd.DataFrame(prices).transpose()
selector = raw_prices.nunique(dropna=False) > 10
prices = raw_prices.loc[:, selector]
returns = prices.pct_change().iloc[1:, :]
correlations = returns.corr()
covariance = returns.cov()
print('Number of NA in corr matrix', correlations.isnull().sum().sum())

symbols = prices.columns
s2i = {e: i for i, e in enumerate(symbols)}
i2s = {i: e for i, e in enumerate(symbols)}

write_pickle(correlations, 'assets/correlations.pkl')
write_pickle(covariance, 'assets/covariance.pkl')
write_pickle(raw_prices, 'assets/raw_prices.pkl')
write_pickle(prices, 'assets/prices.pkl')
write_pickle(returns, 'assets/returns.pkl')
write_pickle(s2i, 'assets/s2i.pkl')
write_pickle(i2s, 'assets/i2s.pkl')
