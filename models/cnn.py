import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F

class CNN(chainer.Chain):
    def __init__(self):
        # クラスの初期化
        super(CNN, self).__init__(            
            conv1 = L.Convolution2D(1, 20, 4), # フィルター3　ボケを作ってる、入力、出力
            conv2 = L.Convolution2D(20, 30, 3), # フィルター4
            conv3 = L.Convolution2D(30, 40, 3),
            conv4 = L.Convolution2D(40, 50, 3),

            l1 = L.Linear(800, 500),
            l2 = L.Linear(500, 500),
            l3 = L.Linear(500, 10, initialW=np.zeros((10, 500), dtype=np.float32))
        )
        
    def __call__(self,x):
        # 順伝播の計算を行う関数
        # :param x: 入力値
        h = F.max_pooling_2d(F.dropout(F.relu(self.conv1(x))), 2)
        h = F.max_pooling_2d(F.dropout(F.relu(self.conv2(h))), 2)
        h = F.max_pooling_2d(F.dropout(F.relu(self.conv3(h))), 2)
        h = F.dropout(F.relu(self.conv4(h)))

        h = F.relu(self.l1(h))
        h = F.relu(self.l2(h))
        y = self.l3(h)
        return y