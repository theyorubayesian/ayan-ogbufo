import logging
import os
import requests
import time

from fuzzywuzzy import process
from numpy import nan
from pandas import DataFrame

from scripts.cmd import CMDHelper

logging.root.setLevel(logging.INFO)


def get_data(subset: str = "train"):
    """
    Download competition data using the Zindi API to '/data' if they do not exist.
    API is always of the format: 'https://api.zindi.africa/v1/competitions/<competition name>/files/<filename>

    :param subset: one of ['train', 'test'] to be downloaded
    :return: path to downloaded/existing file
    """
    allowed = ["train", "test"]
    assert subset.lower() in allowed, "subset must be one of ['train', 'test']"

    api = "https://api.zindi.africa/v1/competitions/ai4d-yoruba-machine-translation-challenge/files/{}.csv"
    token = {"auth_token": "hKhCphfxxZk6yjG6kJVbbj92"}  # Constant for all their competitions

    os.makedirs(os.path.abspath("data"), exist_ok=True)
    path = os.path.abspath(f"data/{subset}.csv")

    if not os.path.isfile(path):
        response = requests.post(api.format(subset.capitalize()), data=token)
        data = response.content

        with open(path, "wb") as f:
            f.write(data)

    return path


def fuzzfilter(sample: str, candidates: list, pad: int):
    """
    Wraps fuzzywuzzy.process.extractOne to return best match from
    a computed list of choices

    Choices are computed only from the subset of candidates that
    satisfy len(word) - pad < len(word) < len(word) + pad.

    :param sample:  Sentences to be filtered
    :param candidates:  List of sentences to compare sample sentences to
    :param pad: Used to narrow down candidates for filtering
    :return: Highest similarity score obtained else np.nan if no candidates satisfy specified condition
    """
    actual_candidates = [
        x for x in candidates if len(sample) + pad >= len(x) >= len(sample) - pad
    ]

    if actual_candidates:
        return process.extractOne(sample, candidates)[1]
    else:
        return nan


def remove_similar_sentences(
        df: DataFrame,
        comparison_sentences: list,
        comparison_col: str = "Yoruba",
        pad: int = 5,
        similarity_threshold: int = 95,
        verbose: int = 1000
):
    """
    Similar sentences may exist when train/test split is done & may affect BLEU score
    This function removes sentences with similarity scores below specified threshold

    pad & similarity_threshold are the most important parameters there.
    Choices may affect downstream tasks performed.
    This function may be time-consuming.

    :param df: Dataframe containing sentences to be filtered
    :param comparison_sentences: References sentences for comparison
    :param comparison_col: Dataframe column for comparison
    :param pad: See fuzzfilter
    :param similarity_threshold: Max allowed similarity score
    :param verbose: Set 0 for silent computation else any integer for regular outputs
    :return: filtered dataframe
    """
    # TODO: Check if multiprocessing speeds up this function
    scores = []
    start_time = time.time()
    temp = df.copy()

    for idx, row in temp.iterrows():
        scores.append(
            fuzzfilter(row[comparison_col], candidates=comparison_sentences, pad=pad)
        )

        if verbose and (idx % verbose == 0):
            hours, rem = divmod(time.time() - start_time, 3600)
            minutes, seconds = divmod(rem, 60)
            print(
                "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds),
                "%0.2f percent complete" % (100.0 * float(idx) / float(len(temp)))
            )

    temp.loc[:, "scores"] = scores
    temp.scores.fillna(0, inplace=True)
    return temp[temp["scores"] < similarity_threshold].reset_index(drop=True)


def write_pc(df: DataFrame, filename: str, src_lang: str = "yo", tgt_lang: str = "en"):
    """
    Writes the parallel corpus (YO - EN) to separate files with extensions .yo & .en respectively

    :param df: Dataframe containing the parallel corpus
    :param filename: Ideally train or test but can be any filename
    :param src_lang: Source language - Yoruba
    :param tgt_lang: Target language - Yoruba
    :return: None
    """
    with open("data/" + filename + f".{src_lang}", "wb") as src_file, \
            open("data/" + filename + f".{tgt_lang}", "wb") as tgt_file:
        for _, row in df.iterrows():
            src_file.write((row["Yoruba"]).encode("utf-8") + b"\n")
            tgt_file.write((row["English"]).encode("utf-8") + b"\n")


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
    output_file = "data/bpe.codes." + str(num_symbols)
    cmd_helper = CMDHelper()
    output, error = cmd_helper.run_cmd_command(
        [
            "subword-nmt", "learn-joint-bpe-and-vocab",
            "--input", src_file, tgt_file,
            "--symbols", str(num_symbols),
            "--output", output_file,
            "--write-vocabulary", src_vocab_file, tgt_vocab_file
        ]
    )
    if error:
        logging.error(error)
    if output:
        logging.info(output)


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
    output, error = cmd_helper.run_cmd_command(
        [
            "subword-nmt", "apply-bpe",
            "--input", input_file,
            "--output", output_file,
            "--codes", code_file,
            "--vocabulary", vocab_file
        ]
    )
    if error:
        logging.error(error)
    if output:
        logging.info(output)
