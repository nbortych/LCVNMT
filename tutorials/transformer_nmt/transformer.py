import math

import torch
import torch.nn as nn

import torch.nn.functional as F


class TransformerModel(nn.Module):

    def __init__(self, ntoken_in, ntoken_out, d_model=256, nhead=2, nhid=256, num_encoder_layers=1,
                 num_decoder_layers=1, dropout=0.5, src_pad_idx=0, trg_pad_idx=0):
        super(TransformerModel, self).__init__()
        # token embedding
        self.src_embedding = nn.Embedding(ntoken_in, d_model)
        self.out_embedding = nn.Embedding(ntoken_out, d_model)
        # positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        # encoder network
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, nhid, dropout)
        encoder_norm = nn.LayerNorm(d_model)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers, encoder_norm)
        # decoder network
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, nhid, dropout)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        # from hidden to vocab
        self.fc_out = nn.Linear(d_model, ntoken_out)

        self.init_weights()

        # store parameters
        self.model_type = 'Transformer'
        self.d_model = d_model
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

    def make_src_mask(self, src):
        # src = [batch size, src len]
        src_mask = (src == self.src_pad_idx)
        return src_mask

    def make_trg_mask(self, trg):

        trg_pad_mask = (trg == self.trg_pad_idx)  # .unsqueeze(1).unsqueeze(2)

        trg_len = trg.shape[0]

        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=trg.device)).t()

        return trg_pad_mask, trg_sub_mask

    @staticmethod
    def xavier_weights(m):
        if hasattr(m, 'weight') and m.weight.dim() > 1:
            nn.init.xavier_uniform_(m.weight.data)

    def init_weights(self):
        initrange = 0.1
        self.src_embedding.weight.data.uniform_(-initrange, initrange)
        self.out_embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)
        self.transformer_encoder.apply(self.xavier_weights)
        self.decoder.apply(self.xavier_weights)

    def forward(self, src, trg):
        # make masks
        src_mask = self.make_src_mask(src).t()
        trg_pad_mask, trg_sub_mask = self.make_trg_mask(trg)
        trg_pad_mask = trg_pad_mask.t()
        # embed
        src = self.src_embedding(src)  # * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        trg = self.out_embedding(trg)

        # encode
        encoder_output = self.transformer_encoder(src, src_key_padding_mask=src_mask)

        # decode
        decoder_output = self.decoder(trg, encoder_output, tgt_mask=trg_sub_mask, tgt_key_padding_mask=trg_pad_mask)
        # back to tokens
        # decoder_output = decoder_output.masked_fill(torch.isnan(decoder_output), 0)
        output = self.fc_out(decoder_output)

        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
