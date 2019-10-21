# TODO: Improve chunking strategy

import numpy as np
import chainer
import os
import librosa
from tqdm import tqdm

class ESDataset(chainer.dataset.DatasetMixin):
    def __init__(self, paths, n_mfcc=63, label_index=0):
        self.paths = paths
        data = []

        for path in tqdm(paths):
            X, sample_rate = librosa.load(path, res_type='kaiser_best')
            X = self.normalize(X)

            mfcc = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=n_mfcc)
            
            t = mfcc.shape[1]
            start = int((t - n_mfcc)/2)
            end = int((t + n_mfcc)/2)
            mfcc = mfcc[:, start:end]
            mfcc = np.expand_dims(mfcc, axis=0)

            label = int(os.path.split(path)[1][label_index])
            data.append((mfcc, label))

        self.data = data

    def __len__(self):
        return len(self.data)
        
    def get_example(self, i):
        return self.data[i]

    def normalize(self, x):
        x = (x - x.min()) / (x.max() - x.min())
        return x

