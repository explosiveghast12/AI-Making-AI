# model.py
import math
import torch
import torch.nn as nn

BOS = 256
EOS = 257

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 4096):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class CharTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, nhead: int,
                 num_encoder_layers: int, num_decoder_layers: int,
                 dim_feedforward: int, dropout: float):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.src_embed = nn.Embedding(vocab_size, d_model)
        self.tgt_embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        # src: [B, S], tgt: [B, T]
        src_mask = None
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)

        src_emb = self.pos_enc(self.src_embed(src))
        tgt_emb = self.pos_enc(self.tgt_embed(tgt))

        memory = self.transformer.encoder(src_emb, mask=src_mask)
        out = self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        logits = self.out(out)  # [B, T, V]
        return logits

    @torch.no_grad()
    def greedy_decode(self, src, max_len: int = 512):
        # src: [B, S]
        device = src.device
        memory = self.transformer.encoder(self.pos_enc(self.src_embed(src)))

        B = src.size(0)
        ys = torch.full((B, 1), BOS, dtype=torch.long, device=device)
        for _ in range(max_len - 1):
            tgt_emb = self.pos_enc(self.tgt_embed(ys))
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(ys.size(1)).to(device)
            out = self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
            logits = self.out(out[:, -1:, :])  # last step
            next_token = logits.argmax(dim=-1)  # [B, 1]
            ys = torch.cat([ys, next_token], dim=1)
            if (next_token == EOS).all():
                break
        return ys