import argparse


def get_train_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument_group("Main arguments")
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="empchat",
        choices=["reddit", "empchat", "dailydialog","persona"],
        help="Data to train/eval on",
    )
    parser.add_argument(
        "-bs", "--batch-size", type=int, default=32, help="Training/eval batch size"
    )
    parser.add_argument(
        "-es", "--epochs", type=int, default=20, help="epochs to train model"
    )
    parser.add_argument(
        "-lr", type=float, default=0.01, help="learning rate used"
    )
    parser.add_argument(
        "--modelckpt",
        type=str,
        help="Checkpoint to load pretrained model",
    )
    parser.add_argument(
        "--optimckpt",
        type=str,
        help="Checkpoint to load optimizer.",
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        default="./checkpoints/tmp",
        help="Checkpoints folder to save the model and vocab.",
    )

    parser.add_argument(
        "--vocabckpt",
        type=str,
        help="Checkpoint folder to load vocabulary. (set when pretrained True)",
    )
    parser.add_argument(
        "--pretrained",
        default=False,
        action='store_true',
        help="Set if a pretrained model is used ",
    )

    parser.add_argument(
        "--max-hist-len",
        type=int,
        default=4,
        help="Max num conversation turns to use in context (used for empchat)",
    )
    parser.add_argument(
        "--max-sent-len", type=int, default=100, help="Max num tokens per sentence"
    )

    parser.add_argument(
        "--dict-max-words",
        type=int,
        default=250000,
        help="Max dictionary size (not used with BERT)",
    )

    parser.add_argument(
        "--multitask1", type=float, default=1.0, help="weight for "
                                                      "first auxilary loss"
    )

    parser.add_argument(
        "--multitask2", type=float, default=1.0, help="weight for "
                                                      "second auxilary loss"
    )

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
        required=True,
        help="Folder where the generated answers while be stored.",
    )

    parser.add_argument(
        "--modelckpt",
        type=str,
        required=True,
        help="Checkpoint file to load the model.",
    )
    parser.add_argument(
        "--vocabckpt",
        type=str,
        default=None,
        help="Checkpoint folder to load vocab (not used for huggingface "
             "models)",
    )

    parser.add_argument(
        "--max-hist-len",
        type=int,
        default=4,
        help="Max num conversation turns to use in context (used for empchat)",
    )
    parser.add_argument(
        "--max-sent-len", type=int, default=100, help="Max num tokens per sentence"
    )

    parser.add_argument(
        "-bs", "--batch-size", type=int, default=32, help="Training/eval batch size"
    )

    parser.add_argument(
        "--dict-max-words",
        type=int,
        default=250000,
        help="Max dictionary size (not used with BERT)",
    )

    return parser
