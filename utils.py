import pickle
import numpy as np


def write_pickle(obj, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(obj, handle)


def get_pickle(filename):
    with open(filename, 'rb') as handle:
        obj = pickle.load(handle)
    return obj


def to_corr(matrix):
    matrix_t = np.transpose(matrix)
    num = np.matmul(matrix, matrix_t)
    norm = np.sum(np.abs(matrix)**2, axis=1)**(1./2)
    den = np.matmul(np.expand_dims(norm, 1), np.expand_dims(norm, 0))
    return num/den
