import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="empchat",
        choices=["reddit", "empchat", "dailydialog"],
        help="Data to train/eval on",
    )

    parser.add_argument(
        "--max-hist-len",
        type=int,
        default=1,
        help="Max num conversation turns to use in context",
    )
    parser.add_argument(
        "--max-sent-len", type=int, default=100, help="Max num tokens per sentence"
    )

    parser.add_argument(
        "-bs", "--batch-size", type=int, default=32, help="Training/eval batch size"
    )
    parser.add_argument(
        "-es", "--epochs", type=int, default=20, help="epochs to train model"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="./checkpoints/tmp",
        help="Checkpoints folder to save the model.",
    )

    parser.add_argument("--cuda", action="store_true", help="Use CUDA")


    parser.add_argument(
        "--dict-max-words",
        type=int,
        default=250000,
        help="Max dictionary size (not used with BERT)",
    )

    parser.add_argument("--embeddings", type=str, help="Path to embeddings file")
    parser.add_argument("--embeddings_size", type=int,
                        help="Emb size", required=True)
    return parser


def get_test_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="empchat",
        choices=["reddit", "empchat", "dailydialog"],
        help="Data to train/eval on",
    )

    parser.add_argument(
        "--outfolder",
        type=str,
        default="outputs/tmp",
        help="Folder where the generated answers while be stored.",
    )

    parser.add_argument(
        "--max-hist-len",
        type=int,
        default=1,
        help="Max num conversation turns to use in context",
    )
    parser.add_argument(
        "--max-sent-len", type=int, default=100, help="Max num tokens per sentence"
    )

    parser.add_argument(
        "-bs", "--batch-size", type=int, default=32, help="Training/eval batch size"
    )


    parser.add_argument(
        "--ckpt",
        type=str,
        default="./checkpoints/tmp",
        help="Checkpoint file to load the model.",
    )

    parser.add_argument("--cuda", action="store_true", help="Use CUDA")


    parser.add_argument(
        "--dict-max-words",
        type=int,
        default=250000,
        help="Max dictionary size (not used with BERT)",
    )

    parser.add_argument("--embeddings", type=str, help="Path to embeddings file")

    # generate options
    parser.add_argument(
        '--sampling',
        action="store_true",
        help='Perform sampling'
    )
    parser.add_argument(
        '--beam-size',
        type=int,
        default=1,
        help='Beam size, if 1 then greedy search',
    )
    parser.add_argument(
        '-Nbest',
        type=int,
        default=1,
        help='Number of return sequences',
    )
    parser.add_argument(
        '--beam-length-penalty',
        type=float,
        default=0.65,
        help='Applies a length penalty. Set to 0 for no penalty.',
    )
    parser.add_argument(
        '--topk', type=int, default=10, help='K used in Top K sampling'
    )
    parser.add_argument(
        '--topp', type=float, default=0.9, help='p used in nucleus sampling'
    )

    parser.add_argument(
    '--temp',
    type = float,
    default = 1.0,
    help = 'temperature to add during decoding',

    )
    return parser

def get_options():
    parser = get_parser()
    options = parser.parse_args()
    return options

def get_generate_options():
    parser = get_test_parser()
    options = parser.parse_args()
    return options