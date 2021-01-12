import logging

import click
import pandas as pd

from scripts.cmd import CMDHelper
from scripts.utils import apply_learned_bpe
from scripts.utils import learn_bpe_and_vocab
from scripts.utils import remove_similar_sentences
from scripts.utils import write_pc

logging.root.setLevel(logging.INFO)


@click.command()
@click.argument(
    "data_path", type=click.File("r", encoding="utf-8")
)
@click.option(
    "--seed",
    "-s",
    type=int,
    default=42,
    show_default=True,
    help="Random state seed for dataset shuffling. Necessary for reproducibility")
@click.option(
    "--test_size",
    "-t",
    type=float,
    default=0.3,
    show_default=True,
    help="Fraction of train data to use as test set during training")
@click.option(
    "--check_similarity",
    "-cs",
    is_flag=True,
    help="Remove sentences in train that are similar to one or more sentences in test."
)
@click.option(
    "pad",
    "-p",
    type=int,
    default=5,
    show_default=True,
    help="Used to narrow down candidates for fuzzfiltering. See remove_similar_sentences"
)
@click.option(
    "--similarity_threshold",
    "-st",
    type=int,
    default=95,
    show_default=True,
    help="Sentences with scores above threshold are dropped. See remove_similar_sentences"
)
@click.option(
    "--lowercase_corpora",
    "-l",
    is_flag=True,
    help="Apply lower() to data"
)
@click.option(
    "--bpe_exists",
    "-b",
    is_flag=True,
    help="Indicate whether BPE already exists"
)
@click.option(
    "--num_symbols",
    "-n",
    type=int,
    default=10000,
    show_default=True,
    help="Number of symbols for BPE learning"
)
def main(
        data_path,
        seed,
        test_size,
        check_similarity,
        pad,
        similarity_threshold,
        lowercase_corpora,
        bpe_exists,
        num_symbols
):
    assert similarity_threshold <= 100, "similarity_threshold must be <= 100"
    data = pd.read_csv(data_path)
    data.drop_duplicates(subset=["Yoruba", "English"], inplace=True)

    # Shuffle to remove bias in dev-test selection
    data = data.sample(frac=1, random_state=seed).reset_index(drop=True)
    logging.info(f"Dropped duplicates in data and shuffled. No of parallel sentences left: {len(data)}")

    if lowercase_corpora:
        data["Yoruba"] = data["Yoruba"].str.lower()
        data["English"] = data["English"].str.lower()
        logging.warning("Applied lower() to dataset")

    no_test_sentences = int(test_size * len(data))
    test = data.tail(no_test_sentences)
    train = data.drop(test.index)
    logging.info("Split dataset into train and test")

    # This operation can take up to 4 hours
    if check_similarity:
        logging.info("Removing sentences in train that are similar to some in test.")
        filtered_train = remove_similar_sentences(
            train,
            comparison_sentences=list(test["Yoruba"]),
            pad=pad,
            similarity_threshold=similarity_threshold
        )
        logging.info(f"Removed sentences in train with similarity scores > {similarity_threshold}. "
                     f"No of parallel sentences in train: {len(filtered_train)}")
        write_pc(filtered_train, filename="train")
    else:
        write_pc(train, filename="train")

    write_pc(test, filename="test")
    logging.info("Wrote train and test data to 'data/'")

    if not bpe_exists:
        logging.info(f"Learning BPE for corpus. num_symbols is set to {num_symbols}")
        learn_bpe_and_vocab(
            src_file="data/train.yo",
            tgt_file="data/train.en",
            src_vocab_file="data/vocab.yo",
            tgt_vocab_file="data/vocab.en",
            num_symbols=num_symbols
        )
    
    apply_bpe_args = [
        [
            "data/train.yo", "data/train.bpe.yo",
            f"data/ bpe.codes.{num_symbols}", "data/vocab.yo"
        ],
        [
            "data/train.en", f"data/train.bpe.en",
            f"data/bpe.codes.{num_symbols}", "data/vocab.en"
        ],
        [
            "data/test.yo", "data/test.bpe.yo",
            f"data/bpe.codes.{num_symbols}", "data/vocab.yo"
        ],
        [
            "data/test.en", "data/test.bpe.en",
            f"data/bpe.codes.{num_symbols}", "data/vocab.en"
        ]
    ]
    
    for args in apply_bpe_args:
        logging.info(f"Applying BPE to {args[0]}")
        apply_learned_bpe(*args)

    logging.info("Building vocabulary. Output will be written to 'data/vocab.txt'")
    jnmt_cmd_helper = CMDHelper()
    output, error = jnmt_cmd_helper.run_cmd_command(
        [
            "python", "joeynmt/scripts/build_vocab.py",
            "data/train.bpe.yo", "data/train.bpe.en",
            "--output_path", "data/vocab.txt"
        ]
    )
    if error:
        logging.warning(error)
    if output:
        logging.info(output)


if __name__ == "__main__":
    main()
