import logging
import os

import click
import pandas as pd

from scripts.cmd import CMDHelper
from scripts.utils import remove_similar_sentences

logging.root.setLevel(logging.INFO)


def write_pc(df: pd.DataFrame, filename: str, src_lang: str = "yo", tgt_lang: str = "en"):
    """
    Writes the parallel corpus (YO - EN) to separate files with extensions .yo & .en respectively

    :param df: Dataframe containing the parallel corpus 
    :param filename: Ideally train or test but can be any filename 
    :param src_lang: Source language - Yoruba
    :param tgt_lang: Target language - Yoruba
    :return: None
    """
    with open("data/" + filename + f".{src_lang}", "w") as src_file, \
            open("data/" + filename + f".{tgt_lang}", "w") as tgt_file:
        for idx, row in df.iterrows():
            print(idx)
            src_file.write(row["Yoruba"] + "\n")
            tgt_file.write(row["English"] + "\n")


def learn_bpe_and_vocab(
        src_file: str,
        tgt_file: str,
        src_vocab_file: str,
        tgt_vocab_file: str,
        num_symbols: int = 10000,
):
    """
    Runs the cmd command below:

    `subword-nmt learn-joint-bpe-and-vocab --input <src_file> <tgt_file> \
        --symbols <num_symbols> --output <output_file> --write-vocabulary <src_vocab> <tgt_vocab>`
    
    See subword-nmt/learn-joint-bpe-and-vocab.py for details 

    :param src_file: Input source language text
    :param tgt_file: Input target language text
    :param src_vocab_file: Output file for source language vocabulary
    :param tgt_vocab_file: Output file for target language vocabulary
    :param num_symbols: Num of symbols to learn in BPE 
    :return: None
    """
    output_file = "bpe.codes." + str(num_symbols)
    cmd_helper = CMDHelper()
    output = cmd_helper.run_cmd_command(
        [
            "subword-nmt", "learn-joint-bpe-and-vocab",
            "--input", src_file, tgt_file,
            "--symbols", num_symbols,
            "--output", output_file,
            "--write_vocabulary", src_vocab_file, tgt_vocab_file
        ]
    )
    print(output)


def apply_learned_bpe(input_file: str, output_file: str, code_file: str, vocab_file: str):
    """
    Applies learned BPE and vocabulary to corpus and writes encoded text to new file

    :param input_file: Input file containing text to be encoded 
    :param output_file: Output file for the encoded text 
    :param code_file: Existing file containing the learned BPE
    :param vocab_file: Existing file containing the vocabulary for the language 
    :return: None
    """
    cmd_helper = CMDHelper()
    output = cmd_helper.run_cmd_command(
        [
            "subword-nmt", "apply-bpe",
            "--input", input_file,
            "--output", output_file,
            "--codes", code_file,
            "--vocabulary", vocab_file
        ]
    )
    print(output)


@click.command()
@click.argument(
    "data_path", type=click.File("r", encoding="utf-8")
)
@click.option(
    "--seed",
    "-s",
    type=int,
    default=777,
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
    data.sample(frac=1, random_state=seed).reset_index(drop=True, inplace=True)
    logging.info(f"Dropped duplicates in data and shuffled. No of parallel sentences left: {len(data)}")

    if lowercase_corpora:
        data["Yoruba"] = data["Yoruba"].str.lower()
        data["English"] = data["English"].str.lower()
        logging.warning("Applied lower() to dataset")

    no_test_sentences = int(test_size * len(data))
    test = data.tail(no_test_sentences)
    train = data.drop(test.index)
    logging.info("Split dataset into train and test")
    
    filtered_train = remove_similar_sentences(
        train,
        comparison_sentences=list(test["Yoruba"]),
        pad=pad,
        similarity_threshold=similarity_threshold
    )
    logging.info(f"Removed sentences in train with similarity scores > {similarity_threshold}."
                 f"No of parallel sentences in train: {len(filtered_train)}")

    write_pc(filtered_train, filename="train")       
    write_pc(test, filename="test")
    logging.info("Writing train and test data to 'data/'")

    if not bpe_exists:
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
            f"data/bpe.codes.{num_symbols}", "data/vocab.yo"
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
        apply_learned_bpe(*args)

    """
    # Build joeynmt vocabulary
    jnmt_cmd_helper = CMDHelper()
    output = jnmt_cmd_helper.run_cmd_command(
        [
            ""
        ]
    )
    """


if __name__ == "__main__":
    main()
