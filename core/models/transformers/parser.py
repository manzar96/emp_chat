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
        '--nheads', type=int, default=2,dest='n_heads',
        help='Number of multihead attention heads'
    )
    argparser.add_argument("--embeddings", type=str,
                        help="Path to embeddings file")
    argparser.add_argument("--embeddings_size", type=int,
                        help="Emb size", required=True)
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


def add_cmdline_args_gen(argparser):
    argparser.add_argument_group('Generation arguments')
    argparser.add_argument(
        '--beam-size',
        type=int,
        default=1,
        help='Beam size, if 1 then greedy search',
    )
    argparser.add_argument(
        '--beam-min-length',
        type=int,
        default=1,
        help='Minimum length of prediction to be generated by the beam search',
    )
    argparser.add_argument(
        '--beam-context-block-ngram',
        type=int,
        default=-1,
        help=(
            'Size n-grams to block in beam search from the context. val <= 0 '
            'implies no blocking'
        ),
    )
    argparser.add_argument(
        '--beam-block-ngram',
        type=int,
        default=-1,
        help='Size n-grams to block in beam search. val <= 0 implies no blocking',
    )
    argparser.add_argument(
        '--beam-length-penalty',
        type=float,
        default=0.65,
        help='Applies a length penalty. Set to 0 for no penalty.',
    )
    # argparser.add_argument(
    #     '--skip-generation',
    #     action='store_true',
    #     default=False,
    #     help='Skip beam search. Useful for speeding up training, '
    #          'if perplexity is the validation metric.',
    # )
    argparser.add_argument(
        '--method',
        choices={'beam', 'greedy', 'topk', 'nucleus', 'delayedbeam'},
        default='greedy',
        help='Generation algorithm',
    )
    argparser.add_argument(
        '--topk', type=int, default=10, help='K used in Top K sampling'
    )
    argparser.add_argument(
        '--topp', type=float, default=0.9, help='p used in nucleus sampling'
    )
    argparser.add_argument(
        '--beam-delay', type=int, default=30, help='used in delayedbeam search'
    )
    # argparser.add_argument(
    #     '--beam-blacklist-filename',
    #     type=str,
    #     default=None,
    #     help='Load a text file of hard blocks for beam search to never say.',
    # )
    argparser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='temperature to add during decoding',
    )
    # argparser.add_argument(
    #     '--compute-tokenized-bleu',
    #     action='store_true',
    #     default=False,
    #     help='if true, compute tokenized bleu scores',
    # )

    return argparser