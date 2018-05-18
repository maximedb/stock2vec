import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipGramNegativeSampling(nn.Module):

    def __init__(self, size, dimension):
        super(SkipGram, self).__init__()
        self.embeddings = nn.Embedding(size, dimension)
        self._logsigmoid = nn.LogSigmoid()

    def forward(self, center, target, noise):
        center = self.embeddings(center)  # [Batch, Embedding]
        center = center.unsqueeze(1)  # [Batch, 1, Embedding]
        target = self.embeddings(target)  # [Batch, Embedding]
        target = target.unsqueeze(1)  # [Batch, 1, Embedding]
        noise = -1*self.embeddings(noise)  # [Batch, num_negs, Embedding]
        p_scre = target.bmm(center.transpose(1, 2))
        n_scre = torch.sum(noise.bmm(center.transpose(1, 2)), 1)
        loss = self._logsigmoid(p_scre) + self._logsigmoid(n_scre)
        return -torch.mean(loss)


class SkipGram(nn.Module):

    def __init__(self, size, dimension):
        super(SkipGram, self).__init__()
        self.embeddings = nn.Embedding(size, dimension)

    def forward(self, center, target):
        weights = self.embeddings.weight
        center = self.embeddings(center)
        mm = weights.matmul(center.transpose(0, 1)).transpose(0, 1)
        log_softmax = F.log_softmax(mm, dim=0)
        return F.nll_loss(log_softmax, target)


class GloVeCor(nn.Module):

    def __init__(self, size, dimension):
        super(GloVeCor, self).__init__()
        self.embeddings = nn.Embedding(size, dimension)
        self.loss = nn.MSELoss()

    def forward(self, left, right, correlations):
        left = self.embeddings(left)  # [Batch, Embedding]
        right = self.embeddings(right)  # [Batch, Embedding]
        similarity = F.cosine_similarity(left, right, 1, 1e-8)
        return self.loss(similarity, correlations)


class GloVeCov(nn.Module):

    def __init__(self, size, dimension):
        super(GloVeCov, self).__init__()
        self.embeddings = nn.Embedding(size, dimension)
        self.loss = nn.MSELoss()

    def forward(self, left, right, covariances):
        left = self.embeddings(left)  # [Batch, Embedding]
        right = self.embeddings(right)  # [Batch, Embedding]
        similarity = torch.sum(left * right, dim=1)
        return self.loss(similarity, covariances)
