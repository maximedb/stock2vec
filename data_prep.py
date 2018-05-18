import pandas as pd
from utils import write_pickle, get_pickle

CONTEXT_WINDOW = 3

s2i = get_pickle('assets/s2i.pkl')
returns = get_pickle('assets/returns.pkl')

returns.columns = [s2i[e] for e in returns.columns]
dataset = []
for date, row in returns.iterrows():
    print(date)
    for i, symbol in enumerate(row.index):
        sym_ret = row[symbol]
        if pd.isnull(sym_ret):
            continue
        abs = row[(row - sym_ret).dropna().abs().argsort()]
        similars = abs.iloc[1: (1 + CONTEXT_WINDOW)].index.tolist()
        for similar in similars:
            dataset.append(dict(input=symbol, target=similar))

dataset = pd.DataFrame(dataset)
write_pickle(dataset, 'assets/dataset.pkl')
