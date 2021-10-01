import io
import math
import time

import torchtext
import torch
from torch import nn
import torch.optim as optim
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
from torchtext.utils import download_from_url, extract_archive
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from transformer import TransformerModel


def download_data(url_base='https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/',
                  train_urls=('train.de.gz', 'train.en.gz'),
                  val_urls=('val.de.gz', 'val.en.gz'),
                  test_urls=('test_2016_flickr.de.gz', 'test_2016_flickr.en.gz')):
    train_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in train_urls]
    val_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in val_urls]
    test_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in test_urls]
    return train_filepaths, val_filepaths, test_filepaths


def build_vocab(filepath, tokenizer):
    counter = Counter()
    with io.open(filepath, encoding="utf8") as f:
        for string_ in f:
            counter.update(tokenizer(string_))
    return Vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])


def data_process(filepaths, de_tokenizer, de_vocab, en_tokenizer, en_vocab):
    raw_de_iter = iter(io.open(filepaths[0], encoding="utf8"))
    raw_en_iter = iter(io.open(filepaths[1], encoding="utf8"))
    data = []
    for (raw_de, raw_en) in zip(raw_de_iter, raw_en_iter):
        de_tensor_ = torch.tensor([de_vocab[token] for token in de_tokenizer(raw_de)],
                                  dtype=torch.long)
        en_tensor_ = torch.tensor([en_vocab[token] for token in en_tokenizer(raw_en)],
                                  dtype=torch.long)

        if de_tensor_.numel() > 0 and en_tensor_.numel() > 0:
            data.append((de_tensor_, en_tensor_))
    return data


def generate_batch(data_batch, BOS_IDX=None, EOS_IDX=None, PAD_IDX=None):
    de_batch, en_batch = [], []
    for (de_item, en_item) in data_batch:
        de_batch.append(torch.cat([torch.tensor([BOS_IDX]), de_item, torch.tensor([EOS_IDX])], dim=0))
        en_batch.append(torch.cat([torch.tensor([BOS_IDX]), en_item, torch.tensor([EOS_IDX])], dim=0))
    de_batch = pad_sequence(de_batch, padding_value=PAD_IDX)
    en_batch = pad_sequence(en_batch, padding_value=PAD_IDX)
    return de_batch, en_batch


def generate_batch_wrapper(BOS_IDX=None, EOS_IDX=None, PAD_IDX=None):
    def wrapper(data_batch):
        return generate_batch(data_batch, BOS_IDX=BOS_IDX, EOS_IDX=EOS_IDX, PAD_IDX=PAD_IDX)

    return wrapper


def get_dataloader():
    # download data to filepaths
    train_filepaths, val_filepaths, test_filepaths = download_data()

    # create tokenizers
    de_tokenizer = get_tokenizer('spacy', language='de_core_news_sm')
    en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

    # create vocab
    de_vocab = build_vocab(train_filepaths[0], de_tokenizer)
    en_vocab = build_vocab(train_filepaths[1], en_tokenizer)

    BATCH_SIZE = 128
    PAD_IDX = de_vocab['<pad>']
    BOS_IDX = de_vocab['<bos>']
    EOS_IDX = de_vocab['<eos>']

    generate_batch_fn = generate_batch_wrapper(BOS_IDX=BOS_IDX, EOS_IDX=EOS_IDX, PAD_IDX=PAD_IDX)

    train_val_test_data = list(map(lambda x: data_process(x, de_tokenizer, de_vocab, en_tokenizer, en_vocab),
                                   [train_filepaths, val_filepaths, test_filepaths]))

    train_loader = DataLoader(train_val_test_data[0], batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=generate_batch_fn)
    valid_loader = DataLoader(train_val_test_data[1], batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=generate_batch_fn)
    test_loader = DataLoader(train_val_test_data[2], batch_size=BATCH_SIZE,
                             shuffle=True, collate_fn=generate_batch_fn)

    return train_loader, valid_loader, test_loader, de_vocab, en_vocab


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(model, iterator, optimizer, criterion, clip, device):
    model.train()

    epoch_loss = 0

    for i, (src, trg) in enumerate(iterator):
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()

        output = model(src, trg[:-1, :])

        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        trg = trg[1:, :].contiguous().view(-1)


        loss = criterion(output, trg)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion, device):
    model.eval()

    epoch_loss = 0
    with torch.no_grad():
        for i, (src, trg) in enumerate(iterator):
            src, trg = src.to(device), trg.to(device)

            output = model(src, trg[:-1, :])
            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            trg = trg[1:, :].contiguous().view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, valid_loader, test_loader, de_vocab, en_vocab = get_dataloader()

    model = TransformerModel(len(de_vocab), len(en_vocab),
                             src_pad_idx=de_vocab['<pad>'], trg_pad_idx=en_vocab['<pad>'])
    # model = nn.Transformer(num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=512)
    model.to(device)
    print(f'The model has {count_parameters(model)}: trainable parameters')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=en_vocab['<pad>'])
    # train(model, train_loader, optimizer, criterion, 5, device)

    N_EPOCHS = 10
    CLIP = 1

    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):

        start_time = time.time()

        train_loss = train(model, train_loader, optimizer, criterion, CLIP, device)
        valid_loss = evaluate(model, valid_loader, criterion, device)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'tut6-model.pt')

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')


if __name__ == "__main__":
    main()
