import numpy as np
from os.path import dirname, realpath, exists, basename
from os import makedirs
import os
import math
import pickle
from image_preprocessing.logger import get_logger

logger = get_logger(__file__)
NEAR_ZERO = 4.9406564584124654e-300
# MODEL_PATH = dirname(realpath(__file__)) + '/../data/trained_model'
MODEL_PATH = 'D:/trained_model'


def expect(iterable):
    return sum([i / len(iterable) for i in iterable])


def standard_deviation(iterable):
    e = expect(iterable)
    n = len(iterable)
    return e, math.sqrt(sum([(i / n - e / n) ** 2 for i in iterable]))


class GaussNB:
    def __init__(self, name='My_Classifier'):
        self.label = []
        self.over_all_probability = []
        self.knowledge = None
        self.name = name.replace(' ', '_')

    def fit(self, Xs, Ys):
        process = 0
        l = len(Xs)
        data = []
        for X, Y in zip(Xs, Ys):
            try:
                X = X.reshape(-1)
                if Y not in self.label:
                    self.label.append(Y)
                    data.append(np.array([X]))
                else:
                    idx = self.label.index(Y)
                    data[idx] = np.append(data[idx], [X], axis=0)
                process += 1
                logger.info('finished {}%'.format(int(process / l * 1000) / 10))
            except Exception as e:
                logger.error(e)
        self.knowledge = []
        for idx in range(len(self.label)):
            self.knowledge.append([])
            for i in range(data[idx][0].shape[0]):
                e = np.mean(data[idx][:, i])
                std = math.sqrt(np.mean((data[idx][:, i] - e) ** 2))
                self.knowledge[-1].append([e, std])
        self.over_all_probability = [len(data[idx])/len(Xs) for idx in range(len(self.label))]

    def predict(self, X):
        X = X.reshape(-1)
        lb = []
        e_ = []
        for idx in range(len(self.label)):
            p = 1
            _e = 0
            for i in range(X.shape[0]):
                x = X[i]
                e = self.knowledge[idx][i][0]
                std = self.knowledge[idx][i][1]
                try:
                    pp = math.exp(-(x - e) ** 2 / 2 / std ** 2) / math.sqrt(2 * math.pi * std ** 2)
                except ZeroDivisionError:
                    pp = NEAR_ZERO

                while pp < 1:
                    pp *= 10
                    _e -= 1
                p *= pp
                while p > 1:
                    p /= 10
                    _e += 1
            p *= self.over_all_probability[idx]
            while p < 1:
                p *= 10
                _e -= 1
            lb.append(p)
            e_.append(_e)
        i = idx = -1
        m = None
        string = ''
        e_[0] +=2010

        for p, e in zip(lb, e_):
            i += 1
            string += self.label[i] + ' {} e {}'.format(p,e) + '\n'
            if m is None or m[1] < e or (m[1] == e and m[0] < p):
                m = [p, e]
                idx = i
        logger.info(string)
        Y = self.label[idx]
        return Y

    def save(self, path=MODEL_PATH):
        path += '/' + self.name
        if not exists(path):
            makedirs(path)
        with open(path + '/label.pkl', 'wb') as f:
            pickle.dump(self.label, f)
        with open(path + '/over_all_probability.npy', 'wb') as f:
            np.save(f, np.array(self.over_all_probability))
        for idx in range(len(self.knowledge)):
            with open(path + '/' + str(idx) + '.npy', 'wb') as f:
                np.save(f, self.knowledge[idx])

    def load(self, path=MODEL_PATH):
        self.name = basename(path)
        with open(path + '/label.pkl', 'rb') as f:
            self.label = pickle.load(f)
        self.knowledge = [0 for i in self.label]
        with open(path + '/over_all_probability.npy', 'rb') as f:
            self.over_all_probability = np.load(f)
        for idx in range(len(self.label)):
            with open(path + '/' + str(idx) + '.npy', 'rb') as f:
                self.knowledge[idx] = np.load(f)
        return self
