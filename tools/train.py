import os, sys, glob
import chainer
from chainer.training import extensions

from sklearn.model_selection import train_test_split

from arguments import parse_args
from utils import create_result_dir
sys.path.append('..')
from models.cnn import CNN
from datasets.dataset import ESDataset


def main(args):

    print(args.dataset_dir, os.path.isfile(args.dataset_dir))
    if os.path.isfile(args.dataset_dir):
        paths = [args.dataset_dir]
    else:
        paths = glob.glob(os.path.join(args.dataset_dir, '**/*.wav'))
    
    print(len(paths))
    train_paths, test_path = train_test_split(paths, train_size=0.9, test_size=0.1)
    train = ESDataset(train_paths, label_index = args.label_index)
    test = ESDataset(test_path, label_index = args.label_index)

    batchsize = args.batch_size
    train_iter = chainer.iterators.SerialIterator(train, batchsize)
    test_iter = chainer.iterators.SerialIterator(test, batchsize, False, False)

    gpu_id = 0  # Set to -1 if you use CPU

    model = CNN()
    if args.device >= 0:
        model.to_gpu(args.device)
    if args.init_weights != '':
        chainer.serializers.load_npz(args.init_weights, model)

    # Since we do not specify a loss function here, the default 'softmax_cross_entropy' is used.
    model = chainer.links.Classifier(model)

    # selection of your optimizing method
    optimizer = chainer.optimizers.Adam()

    # Give the optimizer a reference to the model
    optimizer.setup(model)

    # Get an updater that uses the Iterator and Optimizer
    updater = chainer.training.updaters.StandardUpdater(train_iter, optimizer, device=gpu_id)

    # Setup a Trainer
    trainer = chainer.training.Trainer(updater, (args.epoch, 'epoch'), out=args.out_dir)
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.device))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport( ["epoch", "main/loss", "validation/main/loss", "main/accuracy", "validation/main/accuracy", "elapsed_time"]))
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], x_key='epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
    trainer.extend(extensions.snapshot_object(model.predictor, filename='model_epoch-{.updater.epoch}'), trigger=(args.epoch, 'epoch'))
    trainer.extend(extensions.ProgressBar())

    trainer.run()

    chainer.serializers.save_npz(os.path.join(args.out_dir, 'weights.npz'), model)



if __name__ =='__main__':
    args = parse_args()

    # create result dir
    create_result_dir(args.out_dir, args, args.overwrite)

    main(args)