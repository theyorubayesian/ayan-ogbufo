import os
import requests
import time

from fuzzywuzzy import process


def get_data(subset: str = "train"):
    """
    Download competition data using the Zindi API to '/data' if they do not exist.
    API is always of the format: 'https://api.zindi.africa/v1/competitions/<competition name>/files/<filename>
    Returns path to downloaded/existing file
    """
    allowed = ["train", "test"]
    assert subset.lower() in allowed, "subset must be one of ['train', 'test']"

    api = "https://api.zindi.africa/v1/competitions/ai4d-yoruba-machine-translation-challenge/files/{}.csv"
    token = {"auth_token": "hKhCphfxxZk6yjG6kJVbbj92"}      # Constant for all their competitions

    os.makedirs(os.path.abspath("data"), exist_ok=True)
    path = os.path.abspath(f"data/{subset}.csv")

    if not os.path.isfile(path):
        response = requests.post(api.format(subset.capitalize()), data=token)
        data = response.content

        with open(path, "wb") as f:
            f.write(data)

    return path


def fuzzfilter(sample: list, candidates: list, pad: int):
    """
    Wraps fuzzywuzzy.process.extractOne to return best match from 
    a computed list of choices

    Choices are computed only from the subset of candidates that
    satisfy len(word) - pad < len(word) < len(word) + pad.

    Returns np.nan if no candidates satisfy specified condition
    """
    actual_candidates = [
        x for x in candidates if len(x) <= len(sample)+pad and len(x) >= len(sample)-pad
        ]

    if actual_candidates:
        return process.extractOne(sample, candidates)[1]
    else:
        return np.nan


def remove_similar_sentences(
    df: pd.DataFrame,
    col: str = "Yoruba",
    comparison_sentences: list, 
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
    """
    # TODO: Check if multiprocessing speeds up this function
    scores = []
    start_time = time.time()
    temp = df.copy()
    
    for idx, row in temp.iterrows():
        scores.append(
            fuzzfilter(row["Yoruba"], candidates=comparison_sentences, pad=pad)
            )

        if verbose and (idx % verbose == 0):
            hours, rem = divmod(time.time() - start_time, 3600)
            minutes, seconds = divmod(rem, 60)
            print(
                "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds),
                "%0.2f percent complete" % (100.0*float(idx)/float(len(df_pp)))
                )

    temp.loc[:, "scores"] =  scores
    temp.scores.fillna(0, inplace=True)
    return temp[temp["scores"] < similarity_threshold]
