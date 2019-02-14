import argparse
import json
import math
import os

from tqdm import tqdm
from torchtext.data import BucketIterator

from utils import *
from model import *

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--train_samples', type=int, default=100000)
parser.add_argument('--valid_samples', type=int, default=1000)
parser.add_argument('--test_samples', type=int, default=1000)
parser.add_argument('--allowed_operations', type=str, default='+-')
parser.add_argument('--min_value', type=int, default=0)
parser.add_argument('--max_value', type=int, default=9999)
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

word2id = {symbol: i for i, symbol in enumerate('?#^$+-1234567890')}
id2word = {i: symbol for symbol, i in word2id.items()}

START_SYMBOL = '^'
PADDING_SYMBOL = '#'
END_SYMBOL = '$'
UNK_SYMBOL = '?'

x = Field(eos_token=END_SYMBOL, pad_token=PADDING_SYMBOL, init_token=START_SYMBOL, tokenize=lambda x: [xi for xi in x],
          unk_token=UNK_SYMBOL)
y = Field(eos_token=END_SYMBOL, pad_token=PADDING_SYMBOL, init_token=START_SYMBOL, tokenize=lambda x: [xi for xi in x],
          is_target=True, unk_token=UNK_SYMBOL)

TRAIN_DATASET, VALIDATION_DATASET, TEST_DATASET = generate_equation_for_torch(ALLOWED_OPERATIONS, MIN_VALUE, MAX_VALUE,
                                                                              TRAIN_SAMPLES, VALIDATION_SAMPLES,
                                                                              TEST_SAMPLES, x, y)

TRAIN_DATA, VALIDATION_DATA, TEST_DATA = BucketIterator.splits((TRAIN_DATASET, VALIDATION_DATASET, TEST_DATASET),
                                                               batch_sizes=[BATCH_SIZE, BATCH_SIZE, BATCH_SIZE],
                                                               sort_key = lambda x: len(x.x))

x.build_vocab(TRAIN_DATASET)
y.build_vocab(TRAIN_DATASET)

print(f'In train set: {len(TRAIN_DATASET)}, In validation set {len(VALIDATION_DATASET)}, '
      f'In test set {len(TEST_DATASET)}')


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {DEVICE}')


with open(MODEL_CONFIG_PATH, 'r') as f:
    model_params = json.load(f)
    # Model params
    model_name = model_params['model']['name']
    dropout_prob_enc = model_params['model']['dropout_prob_enc']
    n_layers_enc = model_params['model']['n_layers_enc']
    dropout_prob_dec = model_params['model']['dropout_prob_dec']
    n_layers_dec = model_params['model']['n_layers_dec']
    embedding_dim = model_params['model']['embedding_dim']
    n_epochs = model_params['model']['n_epochs']
    hidden_size = model_params['model']['hidden_dim']
    teacher_forcing_ratio = model_params['model']['teacher_forcing_ratio']
    clip_grad = model_params['model']['gradient_clip']
    model_save_path = model_params['model']['save_path']
    # Optimizer
    learning_rate = model_params['model']['optimizer']['learning_rate']

embedding = nn.Embedding(len(word2id), embedding_dim)

if model_name == 'Seq2Seq':
    encoder = Encoder(embedding, hidden_size, dropout_prob_enc, n_layers_enc)
    decoder = Decoder(embedding, len(word2id), hidden_size, dropout_prob_dec, n_layers_dec)
elif model_name == 'LSTMSeq2Seq':
    encoder = LSTMEncoder(embedding, hidden_size, dropout_prob_enc, n_layers_enc)
    decoder = LSTMDecoder(embedding, len(word2id), hidden_size, dropout_prob_dec, n_layers_dec)
else:
    raise NotImplementedError(f'Model {model_name} not implemented')

model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

print(f'Training {model_name}')

criterion = nn.CrossEntropyLoss(ignore_index=word2id[PADDING_SYMBOL])
optimizer = optim.Adam(model.parameters(), learning_rate)


def train(model: Seq2Seq, train_data, optimizer, criterion, clip):
    model.train()

    epoch_loss = 0
    i = 1

    with tqdm(bar_format='{postfix[0]} {postfix[3][iter]}/{postfix[2]} {postfix[1]}: {postfix[1][loss]}',
              postfix=['Training iter:', 'Loss', f'{TRAIN_SAMPLES//BATCH_SIZE}', dict(loss=0, iter=0)]) as t:
        for i, (x_train, y_train) in enumerate(train_data):
            optimizer.zero_grad()

            x_train, y_train = x_train.to(DEVICE), y_train.to(DEVICE)

            output = model.forward(x_train, y_train, teacher_forcing_ratio)

            y_true = y_train[1:].view(-1)
            loss = criterion(output[1:].view(-1, output.shape[2]), y_true)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            optimizer.step()

            epoch_loss += loss.item()

            t.postfix[3]['loss'] = loss.item()
            t.postfix[3]['iter'] = i
            t.update()

    return epoch_loss / i


def evaluate(model: Seq2Seq, validation_data, criterion):
    model.eval()

    epoch_loss = 0
    i = 1

    with torch.no_grad():
        for i, (x_val, y_val) in tqdm(enumerate(validation_data),
                                      total=VALIDATION_SAMPLES // BATCH_SIZE, desc='Validating'):

            y_val, y_val = y_val.to(DEVICE), y_val.to(DEVICE)

            y_true = y_val[1:].view(-1)

            output = model.forward(x_val, y_val, 0)

            loss = criterion(output[1:].view(-1, output.shape[2]), y_true)

            epoch_loss += loss.item()

    return epoch_loss / i


def predict(model: Seq2Seq, x, max_len):
    model.eval()

    with torch.no_grad():
        _, hidden = model.encoder(x)
        inp = x[0, :]

        symbol = ''
        output = []

        while symbol != END_SYMBOL and len(output) < max_len:
            out, hidden = model.decoder(inp, hidden)

            idx = out.max(1)[1]
            symbol = id2word[idx.item()]

            output.append(symbol)
            inp = idx

    return ''.join(output)


MODEL_SAVE_PATH = os.path.join(model_save_path, 'model1.pt')

best_valid_loss = float('inf')

if not os.path.isdir(f'{model_save_path}'):
    os.makedirs(f'{model_save_path}')

for epoch in range(n_epochs):

    train_loss = train(model, TRAIN_DATA, optimizer, criterion, clip_grad)
    valid_loss = evaluate(model, VALIDATION_DATA, criterion)

    # for t in TEST_DATA:
    #     ex = t.x[:, 0].unsqueeze(0)
    #     break
    #
    # res = predict(model, ex, MAX_LEN)
    # print(''.join([id2word[idx] for idx in ex.detach().numpy()]), res)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print(
        f'| Epoch: {epoch+1:03} | Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f} | Val. Loss: {valid_loss:.3f} | Val. PPL: {math.exp(valid_loss):7.3f} |')
