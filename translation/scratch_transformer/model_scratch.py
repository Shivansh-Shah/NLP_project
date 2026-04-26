import math

import torch
import torch.nn as nn
import torch.nn.functional as F


PAD_ID = 0


def make_causal_mask(length, device):
    mask = torch.triu(torch.ones((length, length), dtype=torch.bool, device=device), diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)


def make_padding_mask(token_ids, pad_id=PAD_ID):
    return token_ids.eq(pad_id).unsqueeze(1).unsqueeze(2)


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=-1, keepdim=True)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * x_hat + self.bias


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        batch, seq_len, _ = x.size()
        x = x.view(batch, seq_len, self.n_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x):
        batch, _, seq_len, _ = x.size()
        x = x.transpose(1, 2).contiguous()
        return x.view(batch, seq_len, self.d_model)

    def forward(self, query, key, value, attn_mask=None):
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key))
        v = self._split_heads(self.v_proj(value))

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, torch.finfo(scores.dtype).min)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = self._merge_heads(out)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.dropout(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.self_attn = MultiHeadSelfAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        attn_out = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), attn_mask=src_mask)
        x = x + self.dropout1(attn_out)
        x = x + self.dropout2(self.ffn(self.norm2(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.self_attn = MultiHeadSelfAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.cross_attn = MultiHeadSelfAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask, src_mask):
        self_out = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), attn_mask=tgt_mask)
        x = x + self.dropout1(self_out)

        cross_out = self.cross_attn(self.norm2(x), memory, memory, attn_mask=src_mask)
        x = x + self.dropout2(cross_out)

        x = x + self.dropout3(self.ffn(self.norm3(x)))
        return x


class TransformerScratch(nn.Module):
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
        max_len=2048,
        tie_embeddings=True,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.position = SinusoidalPositionalEncoding(d_model=d_model, max_len=max_len, dropout=dropout)
        self.scale = math.sqrt(d_model)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model=d_model, n_heads=nhead, d_ff=d_ff, dropout=dropout) for _ in range(num_enc_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model=d_model, n_heads=nhead, d_ff=d_ff, dropout=dropout) for _ in range(num_dec_layers)]
        )

        self.encoder_norm = LayerNorm(d_model)
        self.decoder_norm = LayerNorm(d_model)

        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.output_proj.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=self.d_model ** -0.5)

    def encode(self, src_ids):
        src_mask = make_padding_mask(src_ids, self.pad_id)

        x = self.embedding(src_ids) * self.scale
        x = self.position(x)

        for layer in self.encoder_layers:
            x = layer(x, src_mask=src_mask)

        memory = self.encoder_norm(x)
        return memory, src_mask

    def decode(self, tgt_ids, memory, src_mask):
        tgt_pad_mask = make_padding_mask(tgt_ids, self.pad_id)
        tgt_causal_mask = make_causal_mask(tgt_ids.size(1), tgt_ids.device)
        tgt_mask = tgt_pad_mask | tgt_causal_mask

        x = self.embedding(tgt_ids) * self.scale
        x = self.position(x)

        for layer in self.decoder_layers:
            x = layer(x, memory=memory, tgt_mask=tgt_mask, src_mask=src_mask)

        return self.decoder_norm(x)

    def decode_step(self, tgt_ids, memory, src_mask):
        dec = self.decode(tgt_ids, memory, src_mask)
        logits = self.output_proj(dec[:, -1, :])
        return logits

    def forward(self, src_ids, tgt_ids):
        memory, src_mask = self.encode(src_ids)
        dec = self.decode(tgt_ids, memory=memory, src_mask=src_mask)
        return self.output_proj(dec)


MODEL_CONFIG = dict(d_model=512, nhead=8, num_enc_layers=4, num_dec_layers=4, d_ff=2048, dropout=0.1)
