import re
from functools import partial
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import ray
from ray.data import Dataset
from ray.data.preprocessor import Preprocessor
from ray.data.preprocessors import BatchMapper, Chain
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

from madewithml.config import ACCEPTED_TAGS, STOPWORDS


def load_data(dataset_loc: str, num_samples: int = None, num_partitions: int = 1) -> Dataset:
    """Load data from source into a Ray Dataset.

    Args:
        dataset_loc (str): Location of the dataset.
        num_samples (int, optional): The number of samples to load. Defaults to None.
        num_partitions (int, optional): Number of shards to separate the data into. Defaults to 1.

    Returns:
        Dataset: Our dataset represented by a Ray Dataset.
    """
    ds = ray.data.read_csv(dataset_loc).repartition(num_partitions)
    ds = ds.random_shuffle(seed=1234)
    ds = ray.data.from_items(ds.take(num_samples)).repartition(num_partitions) if num_samples else ds
    return ds


def stratify_split(
    ds: Dataset,
    stratify: str,
    test_size: float,
    shuffle: bool = True,
    seed: int = 1234,
) -> Tuple[Dataset, Dataset]:  # pragma: no cover, (eventual) Ray functionality
    """Split a dataset into train and test splits with equal
    amounts of data points from each class in the column we
    want to stratify on.

    Args:
        ds (Dataset): Input dataset to split.
        stratify (str): Name of column to split on.
        test_size (float): Proportion of dataset to split for test set.
        shuffle (bool, optional): whether to shuffle the dataset. Defaults to True.
        seed (int, optional): seed for shuffling. Defaults to 1234.

    Returns:
        Tuple[Dataset, Dataset]: the stratified train and test datasets.
    """

    def _add_split(
        df: pd.DataFrame,
    ) -> pd.DataFrame:  # pragma: no cover, used in parent function
        """Naively split a dataframe into train and test splits.
        Add a column specifying whether it's the train or test split."""
        train, test = train_test_split(df, test_size=test_size, shuffle=shuffle, random_state=seed)
        train["_split"] = "train"
        test["_split"] = "test"
        return pd.concat([train, test])

    def _filter_split(df: pd.DataFrame, split: str) -> pd.DataFrame:  # pragma: no cover, used in parent function
        """Filter by data points that match the split column's value
        and return the dataframe with the _split column dropped."""
        return df[df["_split"] == split].drop("_split", axis=1)

    # Train, test split with stratify
    grouped = ds.groupby(stratify).map_groups(
        _add_split, batch_format="pandas"
    )  # group by each unique value in the column we want to stratify on
    train_ds = grouped.map_batches(
        _filter_split, fn_kwargs={"split": "train"}, batch_format="pandas"
    )  # Combine data points from all groups for train split
    test_ds = grouped.map_batches(
        _filter_split, fn_kwargs={"split": "test"}, batch_format="pandas"
    )  # Combine data points from all groups for test split

    # Shuffle each split (required)
    train_ds = train_ds.random_shuffle(seed=seed)
    test_ds = test_ds.random_shuffle(seed=seed)

    return train_ds, test_ds


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
    text = pattern.sub(" ", text)

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
    df = df.drop(columns=["id", "created_on", "title", "description"], errors="ignore")  # drop columns
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
    encoded_inputs = tokenizer(batch["text"].tolist(), return_tensors="np", padding="longest")
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


def get_preprocessor() -> Preprocessor:  # pragma: no cover, just returns a chained preprocessor
    """Create the preprocessor for our task.

    Returns:
        Preprocessor: A single combined `Preprocessor` created from our multiple preprocessors.
    """
    num_classes = len(ACCEPTED_TAGS)
    preprocessor = Chain(
        BatchMapper(preprocess, batch_format="pandas"),
        ray.data.preprocessors.LabelEncoder(label_column="tag"),
        BatchMapper(tokenize, batch_format="pandas"),
        BatchMapper(partial(to_one_hot, num_classes=num_classes), batch_format="numpy"),
    )
    return preprocessor
