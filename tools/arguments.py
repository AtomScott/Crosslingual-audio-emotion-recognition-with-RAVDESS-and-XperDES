import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Training script')
    
    parser.add_argument(
        '--dataset_dir', default='./datasets/RAVDESS', type=str, help='path to dataset dir'    )
    parser.add_argument(
        '--out_dir', default='./results/', type=str, help='path to dataset dir'    )
    parser.add_argument(
        '--batch_size', default=128, type=int,
        help='batch size')
    parser.add_argument(
        '--device', default=-1, type=int,
        help='device to use cpu is -1')
    parser.add_argument(
        '--epoch', default=50, type=int,
        help='number of epochs to train')
    parser.add_argument(
        '--label_index', default=0, type=int,
        help='index of filename to use as a label')
    parser.add_argument(
        '--init_weights', default='', type=str,
        help='Add default weights to init the model.'
    )
    parser.add_argument(
        '--overwrite', default=False, action='store_true',
        help='overwrite out dir.'
    )
    parser.add_argument(
        '--title', default='Confusin Matrix', type=str,
        help='title for confusion matrix.'
    )

    args = parser.parse_args()
    return args
