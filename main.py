import argparse
import json
import math
import os

import torch
from torch import nn
from torch import optim

from utils import *
from model import *

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--train_samples', type=int, default=100000)
parser.add_argument('--valid_samples', type=int, default=10000)
parser.add_argument('--test_samples', type=int, default=10000)
parser.add_argument('--allowed_operations', type=str, default='+-')
parser.add_argument('--min_value', type=int, default=0)
parser.add_argument('--max_value', type=int, default=999999)
parser.add_argument('--model_config', type=str, default='model_config.json')

args = parser.parse_args()

BATCH_SIZE = args.batch_size
TRAIN_SAMPLES = args.train_samples
VALIDATION_SAMPLES = args.valid_samples
TEST_SAMPLES = args.test_samples
ALLOWED_OPERATIONS = args.allowed_operations
MIN_VALUE = args.min_value
MAX_VALUE = args.max_value
MODEL_CONFIG_PATH = args.model_config
MAX_LEN = 20

TRAIN_DATA = generate_equations(ALLOWED_OPERATIONS, TRAIN_SAMPLES, MIN_VALUE, MAX_VALUE)
TEST_DATA = generate_equations(ALLOWED_OPERATIONS, TEST_SAMPLES, MIN_VALUE, MAX_VALUE)
VALIDATION_DATA = generate_equations(ALLOWED_OPERATIONS, VALIDATION_SAMPLES, MIN_VALUE, MAX_VALUE)

word2id = {symbol:i for i, symbol in enumerate('#^$+-1234567890')}
id2word = {i:symbol for symbol, i in word2id.items()}

START_SYMBOL = '^'
PADDING_SYMBOL = '#'
END_SYMBOL = '$'

print(f'In test set: {len(TRAIN_DATA)}, In validation set {len(VALIDATION_DATA)}, In test set {len(TEST_DATA)}')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {DEVICE}')


with open(MODEL_CONFIG_PATH, 'r') as f:
    model_params = json.load(f)
    model_name = model_params['model']['name']
    learning_rate = model_params['model']['learning_rate']
    dropout_prob_enc= model_params['model']['dropout_prob_enc']
    n_layers_enc = model_params['model']['n_layers_enc']
    dropout_prob_dec = model_params['model']['dropout_prob_dec']
    n_layers_dec = model_params['model']['n_layers_dec']
    embedding_dim = model_params['model']['embedding_dim']
    n_epochs = model_params['model']['n_epochs']
    hidden_size = model_params['model']['hidden_dim']
    teacher_forcing_ratio = model_params['model']['teacher_forcing_ratio']
    clip_grad = model_params['model']['gradient_clip']
    model_save_path = model_params['model']['save_path']


embedding = nn.Embedding(len(word2id), embedding_dim, padding_idx=word2id[PADDING_SYMBOL])

encoder = Encoder(embedding, hidden_size, dropout_prob_enc, n_layers_enc)
decoder = Decoder(embedding, len(word2id), hidden_size, dropout_prob_dec, n_layers_dec)
model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

criterion = nn.CrossEntropyLoss(ignore_index=word2id[PADDING_SYMBOL])
optimizer = optim.Adam(model.parameters(), learning_rate)


def train(model: Seq2Seq, train_data, optimizer, criterion, clip):
    model.train()

    epoch_loss = 0
    i = 1

    for i, (x_batch, y_batch) in enumerate(generate_batches(train_data, BATCH_SIZE)):

        x_batch_prep, x_len = batch_to_ids(x_batch, word2id, MAX_LEN, END_SYMBOL, PADDING_SYMBOL)
        y_batch_prep, y_len = batch_to_ids(y_batch, word2id, MAX_LEN, END_SYMBOL, PADDING_SYMBOL)

        output = model.forward(x_batch_prep, y_batch_prep, word2id[START_SYMBOL], teacher_forcing_ratio)


        y_true = torch.LongTensor(y_batch_prep).view(-1)
        loss = criterion(output.view(-1, output.shape[2]), y_true)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / i

def evaluate(model: Seq2Seq, validation_data, criterion):
    model.eval()

    epoch_loss = 0
    i = 1

    with torch.no_grad():

        for i, (x_val, y_val) in enumerate(generate_batches(validation_data, BATCH_SIZE)):
            x_val_prep, x_len = batch_to_ids(x_val, word2id, MAX_LEN, END_SYMBOL, PADDING_SYMBOL)
            y_val_prep, y_len = batch_to_ids(y_val, word2id, MAX_LEN, END_SYMBOL, PADDING_SYMBOL)

            y_true = torch.LongTensor(y_val_prep).view(-1)

            output = model.forward(x_val_prep, y_val_prep, word2id[START_SYMBOL], 0)

            loss = criterion(output.view(-1, output.shape[2]), y_true)

            epoch_loss += loss.item()

    return epoch_loss / i


N_EPOCHS = 10
CLIP = 10
MODEL_SAVE_PATH = os.path.join(model_save_path, 'model1.pt')

best_valid_loss = float('inf')

if not os.path.isdir(f'{model_save_path}'):
    os.makedirs(f'{model_save_path}')

for epoch in range(N_EPOCHS):

    train_loss = train(model, TRAIN_DATA, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, VALIDATION_DATA, criterion)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print(
        f'| Epoch: {epoch+1:03} | Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f} | Val. Loss: {valid_loss:.3f} | Val. PPL: {math.exp(valid_loss):7.3f} |')