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

        input_size = embedding.weight.size()[1]
        self.encoder = nn.GRU(input_size, hidden_size, num_layers=n_layers, dropout=dropout_prob)

    def forward(self, input):
        embedded = self.embedding(input.view(input.shape[1], input.shape[0]))

        out, hidden = self.encoder(embedded)

        return out, hidden


class Decoder(nn.Module):

    def __init__(self, embedding: nn.Embedding, vocabulary_size: int,
                 hidden_size=128, dropout_prob=0.5, n_layers=1):
        super().__init__()

        self.embedding: nn.Embedding = embedding
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.vocabulary_size = vocabulary_size

        input_size = embedding.weight.size()[1]
        self.decoder = nn.GRU(input_size, hidden_size, n_layers, dropout=dropout_prob)

        self.out = nn.Linear(hidden_size, vocabulary_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input.unsqueeze(0))

        o, hidden = self.decoder(embedded, hidden)

        pred = self.out(o.squeeze(0))

        return pred, hidden


class Seq2Seq(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert self.encoder.hidden_size == self.decoder.hidden_size
        assert self.encoder.n_layers == self.decoder.n_layers

    def forward(self, input_batch: List[List[int]], ground_truth: List[List[int]], sos_idx: int,
                teacher_forcing_ratio=0.5):
        batch_size = len(input_batch)
        max_len = max(len(ib) for ib in ground_truth)
        out_size = self.decoder.vocabulary_size

        outputs = torch.zeros(max_len, batch_size, out_size).to(self.device)

        sos_tensor = torch.LongTensor([[sos_idx]]*batch_size)
        batch_tensor = torch.LongTensor(input_batch)
        gt_tensor = torch.LongTensor(ground_truth)

        assert sos_tensor.shape[0] == batch_tensor.shape[0]
        assert sos_tensor.shape[0] == batch_tensor.shape[0]

        input = torch.cat((sos_tensor, batch_tensor), dim=1)
        gt = torch.cat((sos_tensor, gt_tensor), dim=1)

        # Start tokens
        inp = gt[:, 0]

        _, hidden = self.encoder(input)

        for i in range(1, max_len):
            output, hidden = self.decoder(inp, hidden)
            outputs[i] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            inp = gt[:, i] if teacher_force else top1


        return outputs