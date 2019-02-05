import random
from typing import Optional, List

import torch
from torch import nn
from torch import optim


class Encoder(nn.Module):

    def __init__(self, embedding: nn.Embedding,
                 hidden_size=128, dropout_prob=0.5, n_layers=1):
        super().__init__()
        # nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=embedding_dim,
        #                                       padding_idx=padding_token_idx)
        self.embedding = embedding
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.emb_dim = embedding.weight.size()[1]
        self.embed_dropout = nn.Dropout(p=dropout_prob)
        self.encoder = nn.GRU(self.emb_dim, hidden_size, num_layers=n_layers, dropout=dropout_prob)

    def forward(self, input):
        embedded = self.embed_dropout(self.embedding(input))

        out, hidden = self.encoder(embedded)

        return out, hidden


class LSTMEncoder(Encoder):

    def __init__(self, embedding: nn.Embedding, hidden_size=128, dropout_prob=0.5, n_layers=1):
        super().__init__(embedding, hidden_size, dropout_prob, n_layers)

        self.encoder = nn.LSTM(self.emb_dim, hidden_size, n_layers, dropout=dropout_prob)

    def forward(self, input):
        embedded = self.embed_dropout(self.embedding(input))

        _, (hidden, cell) = self.encoder(embedded)

        return _, (hidden, cell)


class Decoder(nn.Module):

    def __init__(self, embedding: nn.Embedding, vocabulary_size: int,
                 hidden_size=128, dropout_prob=0.5, n_layers=1):
        super().__init__()

        self.embedding: nn.Embedding = embedding
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.vocabulary_size = vocabulary_size

        self.emb_dim = embedding.weight.size()[1]
        self.decoder = nn.GRU(self.emb_dim, hidden_size, n_layers, dropout=dropout_prob)

        self.out = nn.Linear(hidden_size, vocabulary_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input.unsqueeze(0))

        o, hidden = self.decoder(embedded, hidden)

        pred = self.out(o.squeeze(0))

        return pred, hidden


class LSTMDecoder(Decoder):
    def __init__(self, embedding: nn.Embedding, vocabulary_size: int,
                 hidden_size=128, dropout_prob=0.5, n_layers=1):
        super().__init__(embedding, vocabulary_size, hidden_size, dropout_prob, n_layers)

        self.decoder = nn.LSTM(self.emb_dim, hidden_size, n_layers, dropout=dropout_prob)

    def forward(self, input, hidden):
        embedded = self.embedding(input.unsqueeze(0))

        o, (hidden, cell) = self.decoder(embedded, hidden)

        pred = self.out(o.squeeze(0))

        return pred, (hidden, cell)


class Seq2Seq(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert self.encoder.hidden_size == self.decoder.hidden_size
        assert self.encoder.n_layers == self.decoder.n_layers

    def forward(self, input_batch: torch.LongTensor, ground_truth: torch.LongTensor, teacher_forcing_ratio=0.5):
        batch_size = input_batch.shape[1]
        max_len = ground_truth.shape[0]
        out_size = self.decoder.vocabulary_size

        outputs = torch.zeros(max_len, batch_size, out_size).to(self.device)

        # Start tokens
        inp = ground_truth[0, :]

        _, hidden = self.encoder(input_batch)

        for i in range(1, max_len):
            output, hidden = self.decoder(inp, hidden)
            outputs[i] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            inp = ground_truth[i, :] if teacher_force else top1


        return outputs
