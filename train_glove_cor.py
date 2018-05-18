import torch
import argparse
import numpy as np
import torch.optim as optim
from data_set import gloveDataset, collate_fn
from torch.utils.data import DataLoader
from model import GloVe
from utils import get_pickle


def main(args):
    s2i = get_pickle('assets/s2i.pkl')
    correlations = get_pickle('assets/correlations.pkl')
    indices = correlations[correlations != 1].stack().index.tolist()
    dataset = gloveDataset(indices, correlations, s2i)
    model = GloVe(len(s2i), 300)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    losses = []
    epoch_losses = [np.inf, np.inf, np.inf]
    total_n = len(dataset)
    tmplt = "E:{:2d} - i:{:5d}({:4.2f}%) - L:{:5.5f}"
    for epoch in range(args.epoch):
        dataloader = DataLoader(dataset, batch_size=args.bs,
                                collate_fn=collate_fn, shuffle=True)
        model.train()
        losses = []
        for i, batch in enumerate(dataloader):
            left, right, correlations = batch
            left = torch.LongTensor(left)
            right = torch.LongTensor(right)
            correlations = torch.FloatTensor(correlations)
            loss = model(left, right, correlations)
            loss.backward()
            optimizer.step()
            model.zero_grad()
            losses.append(np.sqrt(loss.data))
            if i % 100 == 0:
                ml = np.mean(losses)
                t = tmplt.format(epoch, i, i*args.bs/total_n*100, ml)
                print(t)
                losses = []
        model.eval()
        dataloader = DataLoader(dataset, batch_size=args.bs,
                                collate_fn=collate_fn, shuffle=True)
        losses = []
        for i, batch in enumerate(dataloader):
            left, right, correlations = batch
            left = torch.LongTensor(left)
            right = torch.LongTensor(right)
            correlations = torch.FloatTensor(correlations)
            loss = model(left, right, correlations)
            losses.append(np.sqrt(loss.data))
        epoch_losses.append(np.mean(losses))
        print('Epoch loss {}'.format(epoch_losses[-1]))
        if epoch_losses[-1] > epoch_losses[-4]:
            break
        else:
            filename = 'assets/model/model_glove_cor.torch'
            state = dict(state_dict=model.state_dict(),
                         loss=epoch_losses,
                         args=args)
            torch.save(state, filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=2500)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--epoch', type=int, default=25)
    args = parser.parse_args()
    main(args)
