import chainer
import os, sys, glob
import japanize_matplotlib 

from arguments import parse_args

sys.path.append('..')
from models.cnn import CNN
from datasets.dataset import ESDataset

import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils.multiclass import unique_labels


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(unique_labels(y_true, y_pred))

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    im.set_clim(0,1)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def predict(model, x):
    y = np.argmax(model.predictor(x=np.array([x], dtype="float32")).data)
    return int(y)

def main(args):
    model = CNN()
    chainer.serializers.load_npz(args.init_weights, model)
    model = chainer.links.Classifier(model)

    paths = glob.glob(os.path.join(args.dataset_dir, '**/*.wav'))
    testset = ESDataset(paths, label_index = args.label_index)

    y_targs = []
    y_preds = [] 
    for data in testset:
        x, y = data

        y_targs.append(int(y))
        y_preds.append(int(predict(model, x)))

    class_names = ['中立', '穏やか', '幸せ', '悲しみ', '怒り', '恐怖', '嫌悪', '驚き']


    accuracy = accuracy_score(y_targs, y_preds)
    plot_confusion_matrix(y_targs, y_preds, classes=class_names, normalize=True,
                        title='{0} 精度:{1:.1%}'.format(args.title,accuracy))

    plt.savefig(os.path.join(args.out_dir, 'confusion_matrix.png'))


if __name__ == '__main__':
    args = parse_args()

    main(args)