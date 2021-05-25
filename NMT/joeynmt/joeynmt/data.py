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
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
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
                    shuffle: bool = False,
                    sampler: Sampler = None,
                    batch_sampler: Sampler = None) -> DataLoader:
    """
    Returns a dataloader for a dataset.

    :param dataset: dataset containing src and optionally trg
    :param batch_size: size of the batches the iterator prepares
    :param train: whether it's training time, when turned off,
        bucketing, sorting within batches and shuffling is disabled
        and lengths will be returned
    :param shuffle: whether to shuffle the data before each epoch
        (no effect if set to True for testing)
    :return: torch dataloader
    """


    # transform words into indices
    def vocab_transform(x, vocab):
        return torch.tensor(
            [vocab.stoi[BOS_TOKEN]] + [vocab.stoi[token] for token in x] + [vocab.stoi[EOS_TOKEN]])
    # extract vocab
    src_vocab = dataset.src_vocab
    trg_vocab = dataset.trg_vocab
    # collate batch
    def generate_batch(data_batch):
        src_batch, trg_batch = [], []
        # we need to compute length for validation
        src_len_batch, trg_len_batch = [], []
        for (src, src_len, trg, trg_len) in data_batch:
            # transform raw text into vocab indices
            src_batch.append(vocab_transform(src, src_vocab))
            trg_batch.append(vocab_transform(trg, trg_vocab))
            # append length
            src_len_batch.append(src_len)
            trg_len_batch.append(trg_len)

        return pad_sequence(src_batch, batch_first=True, padding_value=src_vocab.stoi[PAD_TOKEN]), torch.tensor(
            src_len_batch), \
               pad_sequence(trg_batch, batch_first=True, padding_value=trg_vocab.stoi[PAD_TOKEN]), torch.tensor(
            trg_len_batch)


    kwargs = {"batch_size": batch_size,
              "shuffle": shuffle, "collate_fn": generate_batch,
              "sampler": sampler, "batch_sampler": batch_sampler}
    # if not training, we want the same order every time
    if not train:
        kwargs['shuffle'] = False
    # make sure that batch sampler has no incompatible keywords
    if batch_sampler is not None:
        assert sampler is None, "You've passed both batch sampler and sampler to Dataloader"
        del kwargs['shuffle']
        del kwargs['batch_size']
    # make sure that sampler has no incompatbile keywords
    if sampler is not None:
        assert batch_sampler is None, "You've passed both batch sampler and sampler to Dataloader"

    dataloader = DataLoader(dataset, **kwargs)

    return dataloader


class BatchSamplerSimilarLength(Sampler):
    def __init__(self, dataset, batch_size, indices=None, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        # get the indicies and length
        self.indices = [(i, src_len) for i, (src, src_len, trg, trg_len) in enumerate(dataset)]
        if indices is not None:
            self.indices = self.indices[indices]

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.indices)

        pooled_indices = []
        # create pool of indices with similar lengths
        for i in range(0, len(self.indices), self.batch_size * 100):
            pooled_indices.extend(sorted(self.indices[i:i + self.batch_size * 100], key=lambda x: x[1]))
        self.pooled_indices = [x[0] for x in pooled_indices]

        # yield indices for current batch
        batches = [self.pooled_indices[i:i + self.batch_size] for i in
                   range(0, len(self.pooled_indices), self.batch_size)]

        if self.shuffle:
            random.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self):
        return len(self.pooled_indices) // self.batch_size


class DistributedBatchSamplerSimilarLength(DistributedSampler):
    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False, batch_size=10) -> None:
        super().__init__(dataset=dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed,
                         drop_last=drop_last)
        self.batch_size = batch_size

    def __iter__(self):
        indices = list(super.__iter__())
        batch_sampler = BatchSamplerSimilarLength(self.dataset, batch_size=self.batch_size, indices=indices)
        return iter(batch_sampler)

    def __len__(self) -> int:
        return self.num_samples // batch_size


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
                    # add lengths. choosing to do so, since length is needed in many things
                    src_len, trg_len = len(src_line), len(trg_line)
                    # make sure we don't append too big of examples
                    if src_len <= max_sent_length and trg_len <= max_sent_length:
                        self.data.append((src_line, src_len, trg_line, trg_len))
                    # if vocab is not None:
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


class DistributedEvalSampler(Sampler):
    r"""
    DistributedEvalSampler is different from DistributedSampler.
    It does NOT add extra samples to make it evenly divisible.
    DistributedEvalSampler should NOT be used for training. The distributed processes could hang forever.
    See this issue for details: https://github.com/pytorch/pytorch/issues/22584
    shuffle is disabled by default
    DistributedEvalSampler is for evaluation purpose where synchronization does not happen every epoch.
    Synchronization should be done outside the dataloader loop.
    Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`rank` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
    .. warning::
        In distributed mode, calling the :meth`set_epoch(epoch) <set_epoch>` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.
    Example::
        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.total_size = len(self.dataset)         # true value without extra samples
        indices = list(range(self.total_size))
        self.indices = indices[self.rank:self.total_size:self.num_replicas]
        self.num_samples = len(indices)             # true value without extra samples

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return self.num_samples
