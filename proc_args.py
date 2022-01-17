import argparse

def proc_args():
    parser = argparse.ArgumentParser(description='base argument of train and dev loop')
    parser.add_argument(
    '--dataset',
    type = str,
    required=True,
    help='the dataset for train and dev which can be following:\n \
    casia\n \
    celeb\n \
    msu\n \
    oulu\n \
    replay\n \
    rose\n \
    siw\n '
    )
    parser.add_argument(
    '--backbone',
    type = str,
    required=True,
    help='the backbone for train and dev which can be following:\n \
    resnext50_32x4d\n \
    resnext101_32x8d\n '
    )

    parser.add_argument(
    '--use_lbp',
    type = int,
    default=0,
    help='whether or not using lbp befor backbone.\n '
    )
    parser.add_argument(
    '--lbp_ch',
    type = int,
    default=3,
    help='nubmer of channel outputs for lbp operator (default: 8).\n '
    )
    parser.add_argument(
    '--optimizer',
    type = str,
    default='adam',
    help='the optimizer for training which can be following:\n \
    adam\n \
    sgd\n '
    ) 
    parser.add_argument(
    '--criterion',
    type = str,
    default='BCEWithLogits',
    help='the criterion for claculation loss which can be following:\n \
    BCEWithLogits\n \
    ArcB\n \
    IdBce\n \
    '
    )
    parser.add_argument(
    '--num_epochs',
    default=10,
    type = int,
    help='nubmers of epoch for runing train and dev (default: 20)'
    )
    parser.add_argument(
    '--emb_size',
    default=512,
    type = int,
    help='size of embeddeing space (default: 512)'
    )
    #
    parser.add_argument(
    '--input_size',
    default=224,
    type = int,
    help='size of input image to model (default: 112)'
    )
    parser.add_argument(
    '--lr',
    default=0.00005,
    type = float,
    help='learning rate of optimizer'
    )
    parser.add_argument(
    '--train_batch_size',
    default=64,
    type = int,
    help='batch size for train (default: 128)'
    )

    parser.add_argument(
    '--devel_batch_size',
    default=64,
    type = int,
    help='batch size for development (default: 128)'
    )
    parser.add_argument(
    '--dbg',
    default=False,
    type = bool,
    help='Print Debug info (default: False)'
    )

    
    parser.add_argument(
    '--path',
    default='./',
    type = str,
    help='path (default: 2)'
    )
    cfg = parser.parse_args()

    return cfg


