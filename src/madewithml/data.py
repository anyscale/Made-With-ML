# madewithml/data.py
import re
from functools import partial
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import ray
from ray.data.preprocessors import BatchMapper, Chain
from transformers import BertTokenizer

from config.config import ACCEPTED_TAGS, DATASET_URL, STOPWORDS


def load_data(num_samples: int = None, num_partitions: int = 1) -> ray.data.Dataset:
    """Load data from source into a Ray Dataset.

    Args:
        num_samples (int, optional): The number of samples to load. Defaults to None.
        num_partitions (int, optional): Number of shards to separate the data into. Defaults to 1.

    Returns:
        ray.data.Dataset: Our dataset represented by a Ray Dataset.
    """
    ds = ray.data.read_csv(DATASET_URL).repartition(num_partitions)
    ds = ds.random_shuffle(seed=1234)
    ds = (
        ray.data.from_items(ds.take(num_samples)).repartition(num_partitions) if num_samples else ds
    )
    return ds


def split_data(ds: ray.data.Dataset, test_size: float) -> Tuple[ray.data.Dataset]:
    """Split the dataset into train, val and test splits.

    Args:
        ds (ray.data.Dataset): Ray Dataset to split.
        test_size (float): proportion of entire dataset to use for the test split.
            Train and val splits will equally split the remaining proportion of the dataset.

    Returns:
        Tuple[ray.data.Dataset]: three data splits (train, val and test).
    """
    train_ds, _ = ds.train_test_split(test_size=test_size)
    val_ds, test_ds = _.train_test_split(test_size=0.5)
    return train_ds, val_ds, test_ds


def clean_text(text: str, stopwords: List = STOPWORDS) -> str:
    """Clean raw text string.

    Args:
        text (str): Raw text to clean.
        stopwords (List, optional): _description_. Defaults to STOPWORDS.

    Returns:
        str: _description_
    """
    # Lower
    text = text.lower()

    # Remove stopwords
    pattern = re.compile(r"\b(" + r"|".join(stopwords) + r")\b\s*")
    text = pattern.sub("", text)

    # Spacing and filters
    text = re.sub(r"([!\"'#$%&()*\+,-./:;<=>?@\\\[\]^_`{|}~])", r" \1 ", text)  # add spacing
    text = re.sub("[^A-Za-z0-9]+", " ", text)  # remove non alphanumeric chars
    text = re.sub(" +", " ", text)  # remove multiple spaces
    text = text.strip()  # strip white space at the ends
    text = re.sub(r"http\S+", "", text)  # remove links

    return text


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data in our dataframe.

    Args:
        df (pd.DataFrame): Raw dataframe to preprocess.

    Returns:
        pd.DataFrame: Dataframe with preprocessing applied to it.
    """
    df["text"] = df.title + " " + df.description  # feature engineering
    df["text"] = df.text.apply(clean_text)  # clean text
    df = df.drop(
        columns=["id", "created_on", "title", "description"], errors="ignore"
    )  # drop columns
    df["tag"] = df.tag.apply(lambda x: x if x in ACCEPTED_TAGS else "other")  # replace OOS tags
    df = df[["text", "tag"]]  # rearrange columns
    return df


def tokenize(batch: Dict) -> Dict:
    """Tokenize the text input in our batch using a tokenizer.

    Args:
        batch (Dict): batch of data with the text inputs to tokenize.

    Returns:
        Dict: batch of data with the results of tokenization (`input_ids` and `attention_mask`) on the text inputs.
    """
    tokenizer = BertTokenizer.from_pretrained("allenai/scibert_scivocab_uncased", return_dict=False)
    encoded_inputs = tokenizer(
        batch["text"].tolist(), return_tensors="np", padding="max_length", max_length=50
    )
    return dict(
        ids=encoded_inputs["input_ids"],
        masks=encoded_inputs["attention_mask"],
        targets=np.array(batch["tag"]),
    )


def to_one_hot(batch: Dict, num_classes: int) -> Dict:
    """Convert the encoded labels into one-hot vectors.

    Args:
        batch (Dict): batch of data with targets to make into one-hot vectors.
        num_classes (int): number of classes so we can determine width of our one-hot vectors.

    Returns:
        Dict: batch of data with one-hot encoded targets.
    """
    targets = batch["targets"]
    one_hot = np.zeros((len(targets), num_classes))
    one_hot[np.arange(len(targets)), targets] = 1
    batch["targets"] = one_hot
    return batch


def get_preprocessor() -> ray.data.preprocessor.Preprocessor:
    """Create the preprocessor for our task.

    Returns:
        ray.data.preprocessor.Preprocessor: A single combined `Preprocessor` created from our multiple preprocessors.
    """
    num_classes = len(ACCEPTED_TAGS)
    preprocessor = Chain(
        BatchMapper(preprocess, batch_format="pandas"),
        ray.data.preprocessors.LabelEncoder(label_column="tag"),
        BatchMapper(tokenize, batch_format="pandas"),
        BatchMapper(partial(to_one_hot, num_classes=num_classes), batch_format="numpy"),
    )
    return preprocessor
