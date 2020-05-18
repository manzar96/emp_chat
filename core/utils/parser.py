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

    parser.add_argument(
        "--learn-embeddings", action="store_true", help="Train on embeddings"
    )
    return parser


def get_options():
    parser = get_parser()
    options = parser.parse_args()
    return options