import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from typing import Tuple, Any, Callable, List, Optional
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
import re
from datasets import load_dataset
from src.my_engine.text import build_vocab

def get_torchvision_datasets(name: str, train_transform: transforms.Compose = None,
                             test_transform: transforms.Compose = None) -> Tuple[Dataset, Dataset]:
    """
    Gets the training and test datasets for a given torchvision dataset name.
    
    :param name: The name of the dataset (e.g., 'mnist', 'fashion_mnist')
    :return: A tuple of (train_dataset, test_dataset)
    """
    if not train_transform:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    if not test_transform:
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    if name.lower() == "mnist":
        train_ds = datasets.MNIST(root="data", train=True, download=True, transform=train_transform)
        test_ds = datasets.MNIST(root="data", train=False, download=True, transform=test_transform)
    elif name.lower() in ["fashion_mnist", "fashionmnist"]:
        train_ds = datasets.FashionMNIST(root="data", train=True, download=True, transform=train_transform)
        test_ds = datasets.FashionMNIST(root="data", train=False, download=True, transform=test_transform)
    elif name.lower() == "cifar10":
        train_ds = datasets.CIFAR10(root="data", train=True, download=True, transform=train_transform)
        test_ds = datasets.CIFAR10(root="data", train=False, download=True, transform=test_transform)
    else:
        # Fallback for other datasets that might follow the same structure
        dataset_class = getattr(datasets, name, None)
        if dataset_class is not None:
            train_ds = dataset_class(root="data", train=True, download=True, transform=train_transform)
            test_ds = dataset_class(root="data", train=False, download=True, transform=test_transform)
        else:
            raise ValueError(f"Dataset {name} not supported.")
        
    return train_ds, test_ds

def get_ucimlrepo_datasets(dataset_id: int, dataset_name:str = "", test_size: float = 0.2, stratify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Loads a dataset from the UCI ML repo and splits it into train and test sets.
    
    :param dataset_id: The ID of the dataset to retrieve
    :param dataset_name: The name of the dataset
    :param test_size: The proportion of the dataset to include in the test split
    :param stratify: Whether to stratify the split on the target variable
    :return: A tuple of (X_train, X_test, y_train, y_test)
    """
    # fetch dataset 
    dataset = fetch_ucirepo(id=dataset_id) 
    
    # data (as pandas dataframes) 
    X = dataset.data.features 
    y = dataset.data.targets 

    # Some UCI datasets return targets as DataFrames, but train_test_split and many others prefer Series
    if isinstance(y, pd.DataFrame) and y.shape[1] == 1:
        y = y.iloc[:, 0]
    if dataset_name == 'bike_sharing':
        X['y_raw'] = y
        X.sort_values(by=['dteday', 'hr'], inplace=True)
        y = X['y_raw'].astype(np.float32)
        # X.drop(columns=['instant', 'dteday', 'casual', 'registered', 'y_raw'], inplace=True)
        return None, None, y, None
    elif dataset_name == "appliances_energy":
        features_df = dataset.data.features.copy()
        targets_df = dataset.data.targets.reset_index(drop=True)
        features_df = features_df.reset_index(drop=True)
        # Sort chronologically by date if present
        if "date" in features_df.columns:
            sort_idx = features_df["date"].argsort()
            features_df = features_df.iloc[sort_idx].reset_index(drop=True)
            targets_df = targets_df.iloc[sort_idx].reset_index(drop=True)
            features_df = features_df.drop(columns=["date"])
        # Primary target: first column (Appliances energy consumption in Wh)
        target_col = targets_df.columns[0]
        y = targets_df[target_col].values.astype(np.float32)
        feature_names = list(features_df.columns)
        target_names = [target_col]
        X = features_df.values.astype(np.float32)
        return X, y, target_names, feature_names

    stratify_param = y if stratify else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=stratify_param, random_state=42
    )
    
    return X_train, X_test, y_train, y_test


def get_dataloaders(
    train_ds: Dataset, 
    eval_ds: Dataset = None,
    test_ds: Dataset = None,
    train_batch_size: int = 64,
    eval_batch_size: int = 512,
    test_batch_size: int = 512,
    num_workers:int = 0,
    pin_memory:bool = False,
    collate_fn: Callable = None,
    time_series: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates dataloaders for the training and test datasets.
    
    :param train_ds: The training dataset
    :param test_ds: The test dataset
    :param train_batch_size: Batch size for the training dataloader
    :param test_batch_size: Batch size for the test dataloader
    :return: A tuple of (train_loader, test_loader)
    """
    train_shuffle = True
    if time_series:
        train_shuffle = False

    train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=train_shuffle, num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn)
    eval_loader = None
    test_loader = None
    if eval_ds:
        eval_loader = DataLoader(eval_ds, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn)

    if test_ds:
        test_loader = DataLoader(test_ds, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn)
    
    return train_loader, eval_loader, test_loader

class TextDataset(Dataset):
    """Dataset that stores tokenized text as integer sequences with labels.

    Each sample is a (token_ids_tensor, label_tensor) pair where token_ids_tensor
    is a 1D LongTensor of variable length and label_tensor is a scalar LongTensor.
    """
    def __init__(self, token_id_sequences: list, labels: list):
        self.samples = [
            (torch.tensor(ids, dtype=torch.long), torch.tensor(lbl, dtype=torch.long))
            for ids, lbl in zip(token_id_sequences, labels)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def tokenize(text):
    text = text.lower()
    # remove punctuation but keep letters, numbers, spaces and apostrophes
    text = re.sub(r'[^\w\s\']', '', text)
    return text.split()

def encode(tokens, vocab):
    return [vocab.get(t, vocab["<UNK>"]) for t in tokens]

def get_hf_text_dataset(
    dataset_name: str,
    max_vocab_size: int = 25000,
    min_freq: int = 2,
    train_subset_fn: Callable = None,
    test_subset_fn: Callable = None,
):
    """Load a HuggingFace text dataset and return TextDatasets with a shared vocabulary.

    Downloads the requested dataset via the HuggingFace ``datasets`` library,
    builds a vocabulary from the training split using whitespace and punctuation removal tokenization
    (lowercased and punctuation removed), and encodes both splits into integer-index sequences.

    Supported Datasets
    ------------------
    "imdb"
        The Large Movie Review Dataset for binary sentiment classification
        (positive / negative). Contains 25,000 training reviews and 25,000
        test reviews drawn from IMDB. Labels are 0 (negative) and 1 (positive).

    Args:
        dataset_name: Identifier for which dataset to load.
            Supported values: "imdb".
        max_vocab_size: Maximum vocabulary size (excluding special tokens
            ``<PAD>`` and ``<UNK>``). Defaults to 25000.
        min_freq: Minimum word frequency required for a token to be included
            in the vocabulary. Defaults to 2.

    Returns:
        Tuple of (train_dataset, test_dataset, vocab):
            train_dataset: TextDataset for the training split.
            test_dataset: TextDataset for the test/evaluation split.
            vocab: Word-to-index dictionary (includes ``<PAD>`` at 0 and
                ``<UNK>`` at 1).

    Raises:
        ValueError: If dataset_name is unsupported.
    """
    if dataset_name == "imdb":
        hf_name = "imdb"
        train_split = "train"
        test_split = "test"
        text_key = "text"
        label_key = "label"
    elif dataset_name == "ag_news":
        hf_name = "ag_news"
        train_split = "train"
        test_split = "test"
        text_key = "text"
        label_key = "label"
    elif dataset_name == "yelp_review_full":
        hf_name = "yelp_review_full"
        train_split = "train"
        test_split = "test"
        text_key = "text"
        label_key = "label"
    else:
        raise ValueError(
            f"Unsupported dataset_name='{dataset_name}'. "
            "Supported values: imdb."
        )

    ds = load_dataset(hf_name)
    train_data = ds[train_split]
    test_data = ds[test_split]

    # Apply subset function if provided
    if train_subset_fn is not None:
        train_data = train_subset_fn(train_data)
    if test_subset_fn is not None:
        test_data = test_subset_fn(test_data)

    train_tokens = [tokenize(sample[text_key]) for sample in train_data]
    test_tokens = [tokenize(sample[text_key]) for sample in test_data]

    vocab = build_vocab(train_tokens, max_vocab_size=max_vocab_size, min_freq=min_freq)

    train_ids = [encode(tokens, vocab) for tokens in train_tokens]
    test_ids = [encode(tokens, vocab) for tokens in test_tokens]

    train_labels = [sample[label_key] for sample in train_data]
    test_labels = [sample[label_key] for sample in test_data]

    train_dataset = TextDataset(train_ids, train_labels)
    test_dataset = TextDataset(test_ids, test_labels)

    return train_dataset, test_dataset, vocab


class TimeSeriesDataset(Dataset):
    def __init__(self, data: np.ndarray, seq_len: int, forecast_horizon: int = 1):
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        self.data = torch.tensor(data, dtype=torch.float32)
        self.seq_len = seq_len
        self.forecast_horizon = forecast_horizon

    def __len__(self) -> int:
        # DONE: return number of instances in dataset. Careful!
        return self.data.shape[0] - self.seq_len - self.forecast_horizon + 1

    def __getitem__(self, idx: int):
        x = self.data[idx: idx+self.seq_len]  # (seq_len, num_features)
        y = self.data[idx+self.seq_len: idx+self.seq_len+self.forecast_horizon, 0]  # (forecast_horizon,)
        return x, y