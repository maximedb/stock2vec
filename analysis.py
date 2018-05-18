import pandas as pd
import numpy as np
from utils import get_pickle, to_corr

# Generals
returns = get_pickle('assets/returns.pkl')
corr = get_pickle('assets/correlations.pkl').values
cov = get_pickle('assets/covariance.pkl').values
days = returns.index
metadata = pd.read_csv('assets/metadata.tsv', sep='\t', index_col=False)
glove_cor = np.loadtxt('embeddings/glove_cor_tensors.tsv', delimiter='\t')
glove_cov = np.loadtxt('embeddings/glove_cov_tensors.tsv', delimiter='\t')
skip = np.loadtxt('embeddings/skip_tensors.tsv', delimiter='\t')
holdings = pd.read_csv('assets/holdings.csv', index_col=6)
holdings = holdings.reindex(returns.columns)
holdings = holdings.loc[holdings.index.notnull(), :]
aum = holdings['Mkt Val'].sum()
holdings.loc[:, 'Weight'] = holdings.loc[:, 'Mkt Val'] / aum
sectors = holdings['Sector'].dropna().unique().tolist()
weights = holdings['Weight']


def error_num(estimated, correct):
    mean = np.nanmean(np.abs(correct - estimated), axis=(0, 1))
    std = np.nanstd(np.abs(correct - estimated), axis=(0, 1))
    return mean, std


# Covariance and correlations
glove_cov_cov = np.matmul(glove_cov, np.transpose(glove_cov))
glove_cov_cor = to_corr(glove_cov)
glove_cor_cor = to_corr(glove_cor)
skip_cor = to_corr(skip)
mean_cov = np.nanmean(cov, axis=(0, 1))

tmp = error_num(glove_cov_cov, cov)
print('Glove Cov covariance estimation', tmp)
print('Glove Cov correlation estimation', error_num(glove_cov_cor, corr))
print('Glove Cor correlation estimation', error_num(glove_cor_cor, corr))
print('Skip-Gram correlation estimation', error_num(skip_cor, corr))

# Bag-of-securities
sec_returns = []
for sector in sectors:
    def compute_ret(day_ret, selector):
        total_weight = (day_ret[selector].notnull()*weights[selector]).sum()
        return (weights[selector]*day_ret[selector]).sum()/total_weight
    selector = holdings['Sector'] == sector
    sec_ret = returns.apply(lambda x: compute_ret(x, selector), axis=1)
    sec_ret = sec_ret.rename(sector)
    sec_returns.append(sec_ret)

sec_returns_df = pd.DataFrame(sec_returns).transpose()
sec_cov = sec_returns_df.cov().values
sec_corr = sec_returns_df.corr().values

glove_cov_sec = np.zeros((len(sectors), glove_cov.shape[1]))
glove_cor_sec = np.zeros((len(sectors), glove_cor.shape[1]))
skip_sec = np.zeros((len(sectors), skip.shape[1]))
for i, sector in enumerate(sectors):
    indices = metadata.loc[metadata['Sector'] == sector, :].index.tolist()
    isin = metadata.loc[metadata['Sector'] == sector, 'ISIN'].tolist()
    sec_weights = weights.reindex(isin)
    sec_weights = (sec_weights / weights.sum()).values
    # GloveCov
    vec = np.average(glove_cov[indices, :], axis=0, weights=sec_weights)
    glove_cov_sec[i, :] = vec
    # GloveCov
    vec = np.average(glove_cor[indices, :], axis=0, weights=sec_weights)
    glove_cor_sec[i, :] = vec
    # Skip
    vec = np.average(skip[indices, :], axis=0, weights=sec_weights)
    skip_sec[i, :] = vec

glove_cov_sec_cov = np.matmul(glove_cov_sec, np.transpose(glove_cov_sec))
glove_cov_sec_cor = to_corr(glove_cov_sec)
glove_cor_sec_cor = to_corr(glove_cor_sec)
skip_sec_cor = to_corr(skip_sec)

print('Glove Cov sector covariance', error_num(glove_cov_sec_cov, sec_cov))
print('Glove Cov sector correlation', error_num(glove_cov_sec_cor, sec_corr))
print('Glove Cor sector correlation', error_num(glove_cor_sec_cor, sec_corr))
print('Skip-Gram sector correlation', error_num(skip_sec_cor, sec_corr))
