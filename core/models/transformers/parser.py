import argparse

def add_cmdline_args(argparser):
    argparser.add_argument_group("Transformer Vaswani Arguments")
    argparser.add_argument('-nl', '--n-layers', type=int, default=2)
    argparser.add_argument(
        '-hid',
        '--ffn-size',
        type=int,
        default=300,
        help='Hidden size of the FFN layers',
    )
    argparser.add_argument(
        '--dropout', type=float, default=0.0,
        help='Dropout used in Vaswani 2017 (for embeddings).'
    )
    argparser.add_argument(
        '--attention-dropout',
        type=float,
        default=0.0,
        help='Dropout used after attention softmax.',
    )
    argparser.add_argument(
        '--relu-dropout',
        type=float,
        default=0.0,
        help='Dropout used after ReLU. From tensor2tensor.',
    )
    argparser.add_argument(
        '--n-heads', type=int, default=2,
        help='Number of multihead attention heads'
    )
    argparser.add_argument('--learn-positional-embeddings',
                           action='store_true', default=False)
    argparser.add_argument('--learn-embeddings',
                           action='store_true', default=False,
                           help="learn embeddings")
    argparser.add_argument('--embeddings-scale', action='store_false',
                           default=True)
    argparser.add_argument(
        '--n-positions',
        type=int,
        default=1024,
        help='Number of positional embeddings to learn. Defaults '
             'to truncate or 1024 if not provided.',
    )
    argparser.add_argument(
        '--n-segments',
        type=int,
        default=0,
        help='The number of segments that support the model. '
             'If zero no segment and no langs_embedding.',
    )
    argparser.add_argument(
        '--variant',
        choices={'aiayn', 'xlm', 'prelayernorm'},
        default='aiayn',
        help='Chooses locations of layer norms, etc. prelayernorm '
             'is used to match some fairseq models',
    )
    argparser.add_argument(
        '--activation',
        choices={'relu', 'gelu'},
        default='relu',
        help='Nonlinear activation to use. AIAYN uses relu, but '
             'more recent papers prefer gelu.',
    )
    argparser.add_argument(
        '--output-scaling',
        type=float,
        default=1.0,
        help='scale the output of every transformer by this quantity.',
    )
    argparser.add_argument(
        '--share-word-embeddings',
        action='store_false',
        default=True,
        help='Share word embeddings table for candidate and context'
             'in the memory network',
    )
    argparser.add_argument(
        '-nel',
        '--n-encoder-layers',
        type=int,
        default=-1,
        help='This will overide the n-layers for asymmetrical transformers',
    )
    argparser.add_argument(
        '-ndl',
        '--n-decoder-layers',
        type=int,
        default=-1,
        help='This will overide the n-layers for asymmetrical transformers',
    )
    argparser.add_argument(
        '--model-parallel',
        action='store_true',
        default=False,
        help='Shard the layers across multiple GPUs.',
    )
    return argparser