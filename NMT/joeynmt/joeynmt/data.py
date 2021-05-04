# coding: utf-8
"""
Data module
"""
import io
import sys
import random
import os
import os.path
from typing import Optional
import logging

import torch
from torch.utils.data import Dataset, DataLoader

from torchtext.legacy.datasets import TranslationDataset
from torchtext.legacy import data
# from torchtext.legacy.data import Dataset, Iterator, Field
from torch.nn.utils.rnn import pad_sequence


from joeynmt.constants import UNK_TOKEN, EOS_TOKEN, BOS_TOKEN, PAD_TOKEN
from joeynmt.vocabulary import build_vocab, Vocabulary

logger = logging.getLogger(__name__)


def load_data(data_cfg: dict, datasets: list = None) \
        -> (Dataset, Dataset, Optional[Dataset], Vocabulary, Vocabulary):
    """
    Load train, dev and optionally test data as specified in configuration.
    Vocabularies are created from the training set with a limit of `voc_limit`
    tokens and a minimum token frequency of `voc_min_freq`
    (specified in the configuration dictionary).

    The training data is filtered to include sentences up to `max_sent_length`
    on source and target side.

    If you set ``random_train_subset``, a random selection of this size is used
    from the training set instead of the full training set.

    :param data_cfg: configuration dictionary for data
        ("data" part of configuation file)
    :param datasets: list of dataset names to load
    :return:
        - train_data: training dataset
        - dev_data: development dataset
        - test_data: testdata set if given, otherwise None
        - src_vocab: source vocabulary extracted from training data
        - trg_vocab: target vocabulary extracted from training data
    """
    if datasets is None:
        datasets = ["train", "dev", "test"]

    # load data from files
    src_lang = data_cfg["src"]
    trg_lang = data_cfg["trg"]
    train_path = data_cfg.get("train", None)
    dev_path = data_cfg.get("dev", None)
    test_path = data_cfg.get("test", None)

    if train_path is None and dev_path is None and test_path is None:
        raise ValueError('Please specify at least one data source path.')

    level = data_cfg["level"]
    lowercase = data_cfg["lowercase"]
    max_sent_length = data_cfg["max_sent_length"]

    # tok_fun = lambda s: list(s) if level == "char" else s.split()

    train_data = None
    if "train" in datasets and train_path is not None:
        logger.info("Loading training data...")
        train_data = TranslationTextDataset(file_path=train_path,
                                            lang_extensions=("." + src_lang, "." + trg_lang),
                                            max_sent_length=max_sent_length,
                                            src_vocab=None, trg_vocab=None)

        random_train_subset = data_cfg.get("random_train_subset", -1)
        if random_train_subset > -1:
            # select this many training examples randomly and discard the rest
            train_data, _ = torch.utils.data.random_split(train_data,
                                                          [random_train_subset, len(train_data) - random_train_subset],
                                                          torch.Generator().manual_seed(data_cfg.get("seed", 42)))

    src_max_size = data_cfg.get("src_voc_limit", sys.maxsize)
    src_min_freq = data_cfg.get("src_voc_min_freq", 1)
    trg_max_size = data_cfg.get("trg_voc_limit", sys.maxsize)
    trg_min_freq = data_cfg.get("trg_voc_min_freq", 1)

    src_vocab_file = data_cfg.get("src_vocab", None)
    trg_vocab_file = data_cfg.get("trg_vocab", None)

    assert (train_data is not None) or (src_vocab_file is not None)
    assert (train_data is not None) or (trg_vocab_file is not None)
    # todo maybe build vocab before train_data...
    logger.info("Building vocabulary...")
    src_vocab = build_vocab(language="src", min_freq=src_min_freq,
                            max_size=src_max_size,
                            dataset=train_data, vocab_file=src_vocab_file)
    trg_vocab = build_vocab(language="trg", min_freq=trg_min_freq,
                            max_size=trg_max_size,
                            dataset=train_data, vocab_file=trg_vocab_file)
    train_data.src_vocab = src_vocab
    train_data.trg_vocab = trg_vocab

    dev_data = None
    if "dev" in datasets and dev_path is not None:
        logger.info("Loading dev data...")
        dev_data = TranslationTextDataset(file_path=dev_path,
                                          lang_extensions=("." + src_lang, "." + trg_lang),
                                          max_sent_length=max_sent_length,
                                          src_vocab=src_vocab, trg_vocab=trg_vocab)

    test_data = None
    if "test" in datasets and test_path is not None:
        logger.info("Loading test data...")
        # check if target exists
        if os.path.isfile(test_path + "." + trg_lang):
            test_data = TranslationTextDataset(file_path=test_path,
                                               lang_extensions=("." + src_lang, "." + trg_lang),
                                               max_sent_length=max_sent_length,
                                               src_vocab=src_vocab, trg_vocab=trg_vocab)
        else:
            # no target is given -> create dataset from src only
            test_data = MonoDataset(path=test_path, lang_extensions="." + src_lang, vocab=src_vocab)

    logger.info("Data loaded.")
    return train_data, dev_data, test_data, src_vocab, trg_vocab


def make_dataloader(dataset: Dataset,
                    batch_size: int,
                    batch_type: str = "sentence",
                    train: bool = True,
                    shuffle: bool = False) -> DataLoader:
    """
    Returns a dataloader for a dataset.

    :param dataset: dataset containing src and optionally trg
    :param batch_size: size of the batches the iterator prepares
    :param train: whether it's training time, when turned off,
        bucketing, sorting within batches and shuffling is disabled
    :param shuffle: whether to shuffle the data before each epoch
        (no effect if set to True for testing)
    :return: torch dataloader
    """

    def generate_batch(data_batch):
        src_batch, trg_batch = [], []
        # we need to compute length for validation
        if not train:
            src_len, trg_len = [], []
        for (src, trg) in data_batch:
            # transform raw text into vocab indices
            src_batch.append(vocab_transform(src, src_vocab))
            trg_batch.append(vocab_transform(trg, trg_vocab))
            if not train:
                src_len.append(len(src))
                trg_len.append(len(trg))

        if train:
            return pad_sequence(src_batch, batch_first=True, padding_value=src_vocab.stoi[PAD_TOKEN]), pad_sequence(trg_batch, batch_first=True, padding_value=trg_vocab.stoi[PAD_TOKEN])
        else:
            return pad_sequence(src_batch, batch_first=True, padding_value=src_vocab.stoi[PAD_TOKEN]), torch.tensor(src_len), pad_sequence(trg_batch, batch_first=True, padding_value=trg_vocab.stoi[PAD_TOKEN]),  torch.tensor(trg_len)

    src_vocab = dataset.src_vocab
    trg_vocab = dataset.trg_vocab
    vocab_transform = lambda x, vocab: torch.tensor([vocab.stoi[BOS_TOKEN]] + [vocab.stoi[token] for token in x] + [vocab.stoi[EOS_TOKEN]])
    if train:
        # optionally shuffle and sort during training
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=shuffle, collate_fn=generate_batch)
        print(dataloader)
    else:
        # don't sort/shuffle for validation/inference
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=False, collate_fn=generate_batch)

    return dataloader


class TranslationTextDataset(Dataset):
    """Defines an abstraction for text iterable datasets.
    """

    def __init__(self, file_path, lang_extensions=('.en', '.de'), max_sent_length=100, src_vocab=None, trg_vocab=None,
                 description="Translation dataset"):
        """Initiate the dataset .
        """
        super(TranslationTextDataset, self).__init__()
        # add the extensions to the file path
        src_path, trg_path = tuple(os.path.expanduser(file_path + x) for x in lang_extensions)
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        # if self.vocab is not None:
        #     vocab_transform = lambda x: [self.vocab['<BOS>']] + [self.vocab[token] for token in x] + [
        #         self.vocab['<EOS>']]

        self.data = []
        # load the files into a list
        with io.open(src_path, mode='r', encoding='utf-8') as src_file, \
                io.open(trg_path, mode='r', encoding='utf-8') as trg_file:
            # read each line
            for src_line, trg_line in zip(src_file, trg_file):
                src_line, trg_line = src_line.strip(), trg_line.strip()
                # if non-empty string
                if src_line != '' and trg_line != '':
                    # seperate into words
                    src_line, trg_line = src_line.split(), trg_line.split()
                    # if vocab is provided, transform into indices
                    # todo include lengths?
                    # src_len, trg_len = len(src_line), len(trg_line)
                    # if src_len<=max_sent_length and trg_len<=max_sent_length:
                    # if vocab is None:
                    self.data.append((src_line, trg_line))
                    # else:
                    #     self.data.append((vocab_transform(src_line), vocab_transform(trg_line)))

        self.len_data = len(self.data)
        self.description = description

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len_data

    def __str__(self):
        return self.description


class MonoDataset(Dataset):
    """Defines a dataset for machine translation without targets."""

    #
    # @staticmethod
    # def sort_key(ex):
    #     return len(ex.src)

    def __init__(self, path, lang_extensions='.en', vocab=None, **kwargs) -> None:
        """
        Create a monolingual dataset (=only sources) given path.

        :param path: Prefix of path to the data file
        :param ext: Containing the extension to path for this language.
        :param kwargs: Passed to the constructor of data.Dataset.
        """
        super(MonoDataset, self).__init__()
        path = os.path.expanduser(path + lang_extensions)
        self.data = []
        # load the files into a list
        with io.open(path, mode='r', encoding='utf-8') as src_file:
            for src_line in src_file:
                src_line = src_line.strip()
                # if non-empty string
                if src_line != '':
                    # seperate into words
                    src_line = src_line.split()
                    # if vocab is provided, transform into indices
                    # todo include lengths?
                    # src_len, trg_len = len(src_line), len(trg_line)
                    # if src_len<=max_sent_length and trg_len<=max_sent_length:
                    if vocab is None:
                        self.data.append(src_line)
                    else:
                        self.data.append(vocab_transform(src_line))

        self.len_data = len(self.data)

        def __getitem__(self, index):
            return self.data[index]

        def __len__(self):
            return self.len_data
