import torch
import pickle
import argparse
import numpy as np
import torch.optim as optim
from data_set import skipDataset, collate_fn
from torch.utils.data import DataLoader
from model import SkipGram


def get_pickle(filename):
    with open(filename, 'rb') as handle:
        obj = pickle.load(handle)
    return obj


def main(args):
    LongTensor = torch.cuda.LongTensor if args.gpu else torch.LongTensor
    data = get_pickle('assets/dataset.pkl')
    i2s = get_pickle('assets/i2s.pkl')
    dataset = skipDataset(data)
    model = SkipGram(len(i2s), 300)
    if args.gpu:
        model.cuda()
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
            center, target = batch
            center = LongTensor(center)
            target = LongTensor(target)
            loss = model(center, target)
            loss.backward()
            optimizer.step()
            model.zero_grad()
            losses.append(loss.data)
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
            center, target = batch
            center = torch.LongTensor(center)
            target = torch.LongTensor(target)
            loss = model(center, target)
            losses.append(loss.data)
        epoch_losses.append(np.mean(losses))
        print('Epoch loss {}'.format(epoch_losses[-1]))
        if epoch_losses[-1] > epoch_losses[-4]:
            break
        else:
            filename = 'assets/model/model_skip.torch'
            state = dict(state_dict=model.state_dict(),
                         loss=epoch_losses,
                         args=args)
            torch.save(state, filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=2500)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    main(args)
