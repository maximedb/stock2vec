import torch
import numpy as np
import pandas as pd
from model import GloVeCor, GloVeCov, SkipGram
from utils import get_pickle

s2i = get_pickle('assets/s2i.pkl')
i2s = get_pickle('assets/i2s.pkl')
holdings = pd.read_csv('assets/holdings.csv', index_col=6)

glove_cor_checkpoint = torch.load('assets/model/model_glove_cor.torch')
model_glove = GloVeCor(len(s2i), 300)
model_glove.load_state_dict(glove_cor_checkpoint['state_dict'])
weights = model_glove.embeddings.weight.detach()
np.savetxt('embeddings/glove_cor_tensors.tsv', weights, delimiter='\t')

glove_cov_checkpoint = torch.load('assets/model/model_glove_cov.torch')
model_glove = GloVeCov(len(s2i), 300)
model_glove.load_state_dict(glove_cov_checkpoint['state_dict'])
weights = model_glove.embeddings.weight.detach()
np.savetxt('embeddings/glove_cov_tensors.tsv', weights, delimiter='\t')

skip_checkpoint = torch.load('assets/model/model_skip.torch')
model_skip = SkipGram(len(s2i), 300)
model_skip.load_state_dict(skip_checkpoint['state_dict'])
weights = model_skip.embeddings.weight.detach()
np.savetxt('embeddings/skip_tensors.tsv', weights, delimiter='\t')

selector = [i2s[e] for e in range(len(weights))]
cols = ['Name', 'Sector', 'Industry Group', 'Country', 'Currency']
metadata = holdings.loc[selector, cols]
metadata.to_csv('assets/metadata.tsv', sep='\t')
cols = ['Name', 'Currency']
metadata = holdings.loc[selector, cols]
metadata.to_csv('embeddings/metadata.tsv', sep='\t')
