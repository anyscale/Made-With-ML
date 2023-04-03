
# madewithml/data.py
from collections import Counter
import json
from nltk.stem import PorterStemmer
import numpy as np
import pandas as pd
from pathlib import Path
import ray.train as train
import re
from sklearn.model_selection import train_test_split
import torch
from transformers import BertTokenizer

from config import config
from madewithml import utils


def elt_data():
    """Extract, load and transform our data assets."""
    # Extract and load
    projects = pd.read_csv(config.PROJECTS_URL)
    tags = pd.read_csv(config.TAGS_URL)
    projects.to_csv(Path(config.DATA_DIR, "projects.csv"), index=False)
    tags.to_csv(Path(config.DATA_DIR, "tags.csv"), index=False)

    # Transform
    df = pd.merge(projects, tags, on="id")
    df = df[df.tag.notnull()]  # drop rows w/ no tag
    df.to_csv(Path(config.DATA_DIR, config.LABELED_PROJECTS_FP), index=False)

    print("✅ Saved labeled data!")


def clean_text(text, lower=True, stem=False, stopwords=config.STOPWORDS):
    """Clean raw text."""
    # Lower
    if lower:
        text = text.lower()

    # Remove stopwords
    if len(stopwords):
        pattern = re.compile(r'\b(' + r"|".join(stopwords) + r")\b\s*")
        text = pattern.sub('', text)

    # Spacing and filters
    text = re.sub(
        r"([!\"'#$%&()*\+,-./:;<=>?@\\\[\]^_`{|}~])", r" \1 ", text
    )  # add spacing between objects to be filtered
    text = re.sub("[^A-Za-z0-9]+", " ", text)  # remove non alphanumeric chars
    text = re.sub(" +", " ", text)  # remove multiple spaces
    text = text.strip()  # strip white space at the ends

    # Remove links
    text = re.sub(r"http\S+", "", text)

    # Stemming
    if stem:
        stemmer = PorterStemmer()
        text = " ".join([stemmer.stem(word, to_lowercase=lower) for word in text.split(" ")])

    return text


def replace_oos_labels(df, labels, label_col, oos_label="other"):
    """Replace out of scope (oos) labels."""
    oos_tags = [item for item in df[label_col].unique() if item not in labels]
    df[label_col] = df[label_col].apply(lambda x: oos_label if x in oos_tags else x)
    return df


def replace_minority_labels(df, label_col, min_freq, new_label="other"):
    """Replace minority labels with another label."""
    labels = Counter(df[label_col].values)
    labels_above_freq = Counter(label for label in labels.elements() if (labels[label] >= min_freq))
    df[label_col] = df[label_col].apply(lambda label: label if label in labels_above_freq else None)
    df[label_col] = df[label_col].fillna(new_label)
    return df


def preprocess(df, lower, stem, min_freq):
    """Preprocess the data."""
    df["text"] = df.title + " " + df.description  # feature engineering
    df.text = df.text.apply(clean_text, lower=lower, stem=stem)  # clean text
    df = replace_oos_labels(df=df, labels=config.ACCEPTED_TAGS, label_col="tag", oos_label="other")  # replace OOS labels
    df = replace_minority_labels(df=df, label_col="tag", min_freq=min_freq, new_label="other")  # replace labels below min freq
    return df


class LabelEncoder(object):
    """Encode labels into unique indices."""
    def __init__(self, class_to_index={}):
        self.class_to_index = class_to_index or {}  # mutable defaults ;)
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}
        self.classes = list(self.class_to_index.keys())

    def __len__(self):
        return len(self.class_to_index)

    def __str__(self):
        return f"<LabelEncoder(num_classes={len(self)})>"

    def fit(self, y):
        classes = np.unique(y)
        for i, class_ in enumerate(classes):
            self.class_to_index[class_] = i
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}
        self.classes = list(self.class_to_index.keys())
        return self

    def encode(self, y):
        encoded = np.zeros((len(y)), dtype=int)
        for i, item in enumerate(y):
            encoded[i] = self.class_to_index[item]
        return encoded

    def decode(self, y):
        classes = []
        for i, item in enumerate(y):
            classes.append(self.index_to_class[item])
        return classes

    def save(self, fp):
        with open(fp, "w") as fp:
            contents = {"class_to_index": self.class_to_index}
            json.dump(contents, fp, indent=4, sort_keys=False)

    @classmethod
    def load(cls, fp):
        with open(fp, "r") as fp:
            kwargs = json.load(fp=fp)
        return cls(**kwargs)


def get_data_splits(X, y, train_size=0.7):
    """Generate balanced data splits."""
    X_train, X_, y_train, y_ = train_test_split(
        X, y, train_size=train_size, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_, y_, train_size=0.5, stratify=y_)
    return X_train, X_val, X_test, y_train, y_val, y_test


def prep_data(args):
    """Prepare data."""
    # Setup
    utils.set_seeds()

    # Preprocess
    df = pd.read_csv(config.LABELED_PROJECTS_FP)
    df = df.sample(frac=1).reset_index(drop=True)
    df = df[: args["subset"]]  # None/null = all samples
    df = preprocess(df, lower=args["lower"], stem=args["stem"], min_freq=args["min_freq"])
    label_encoder = LabelEncoder().fit(df.tag)

    # Split
    X_train, X_val, X_test, y_train, y_val, y_test = \
        get_data_splits(X=df.text.to_numpy(), y=label_encoder.encode(df.tag))
    counts = np.bincount(y_train)
    class_weights = {i: 1.0/count for i, count in enumerate(counts)}

    # Tokenize inputs
    tokenizer = BertTokenizer.from_pretrained("allenai/scibert_scivocab_uncased", return_dict=False)
    encoded_input = tokenizer(X_train.tolist(), return_tensors="pt", padding=True)
    X_train_ids = encoded_input["input_ids"]
    X_train_masks = encoded_input["attention_mask"]
    encoded_input = tokenizer(X_val.tolist(), return_tensors="pt", padding=True)
    X_val_ids = encoded_input["input_ids"]
    X_val_masks = encoded_input["attention_mask"]
    encoded_input = tokenizer(X_test.tolist(), return_tensors="pt", padding=True)
    X_test_ids = encoded_input["input_ids"]
    X_test_masks = encoded_input["attention_mask"]

    return [X_train_ids, X_train_masks, y_train], [X_val_ids, X_val_masks, y_val], [X_test_ids, X_test_masks, y_test], class_weights


class TransformerTextDataset(torch.utils.data.Dataset):
    def __init__(self, ids, masks, targets):
        self.ids = ids
        self.masks = masks
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __str__(self):
        return f"<Dataset(N={len(self)})>"

    def __getitem__(self, index):
        ids = self.ids[index]
        masks = self.masks[index]
        targets = self.targets[index]
        return ids, masks, targets

    def create_dataloader(self, batch_size, shuffle=False, drop_last=False):
        return torch.utils.data.DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            pin_memory=False)


def to_one_hot(x):
    one_hot = np.zeros((x.size, x.max()+1))
    one_hot[np.arange(x.size), x] = 1
    return one_hot


def prep_data_loaders(train_data, val_data, test_data, batch_size):
    # Separate data
    X_train_ids, X_train_masks, y_train = train_data
    X_val_ids, X_val_masks, y_val = val_data
    X_test_ids, X_test_masks, y_test = test_data

    # Dataset
    train_dataset = TransformerTextDataset(ids=X_train_ids, masks=X_train_masks, targets=to_one_hot(y_train))
    val_dataset = TransformerTextDataset(ids=X_val_ids, masks=X_val_masks, targets=to_one_hot(y_val))
    test_dataset = TransformerTextDataset(ids=X_test_ids, masks=X_test_masks, targets=to_one_hot(y_test))

    # Create dataloader
    train_dataloader = train_dataset.create_dataloader(batch_size=batch_size)
    val_dataloader = val_dataset.create_dataloader(batch_size=batch_size)
    test_dataloader = test_dataset.create_dataloader(batch_size=batch_size)

    # Prepare dataloader
    train_dataloader = train.torch.prepare_data_loader(train_dataloader)
    val_dataloader = train.torch.prepare_data_loader(val_dataloader)
    test_dataloader = train.torch.prepare_data_loader(test_dataloader)

    return train_dataloader, val_dataloader, test_dataloader
