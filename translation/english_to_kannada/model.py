import math

import torch
import torch.nn as nn


PAD_ID = 0


def make_causal_mask(length, device):
    return torch.triu(torch.ones((length, length), dtype=torch.bool, device=device), diagonal=1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1)])


class TransformerNMT(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        nhead,
        num_enc_layers,
        num_dec_layers,
        d_ff,
        dropout=0.1,
        pad_id=PAD_ID,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.position = PositionalEncoding(d_model=d_model, dropout=dropout)
        self.scale = math.sqrt(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_enc_layers)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_dec_layers)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        self.output_proj.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self):
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def encode(self, src_ids):
        src_pad_mask = src_ids.eq(self.pad_id)
        src = self.position(self.embedding(src_ids) * self.scale)
        memory = self.encoder(src, src_key_padding_mask=src_pad_mask)
        return memory, src_pad_mask

    def decode_step(self, tgt_ids, memory, src_pad_mask):
        tgt = self.position(self.embedding(tgt_ids) * self.scale)
        tgt_mask = make_causal_mask(tgt_ids.size(1), tgt_ids.device)
        out = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=src_pad_mask,
        )
        return self.output_proj(out[:, -1, :])

    def forward(self, src_ids, tgt_ids):
        src_pad_mask = src_ids.eq(self.pad_id)
        tgt_pad_mask = tgt_ids.eq(self.pad_id)
        tgt_mask = make_causal_mask(tgt_ids.size(1), tgt_ids.device)

        src = self.position(self.embedding(src_ids) * self.scale)
        tgt = self.position(self.embedding(tgt_ids) * self.scale)

        memory = self.encoder(src, src_key_padding_mask=src_pad_mask)
        out = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask,
        )
        return self.output_proj(out)


MODEL_CONFIG = dict(d_model=256, nhead=4, num_enc_layers=3, num_dec_layers=3, d_ff=1024, dropout=0.1)