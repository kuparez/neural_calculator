import random
import torch


def generate_equations(allowed_operators, dataset_size, min_value, max_value):
    """Generates pairs of equations and solutions to them.

       Each equation has a form of two integers with an operator in between.
       Each solution is an integer with the result of the operaion.

        allowed_operators: list of strings, allowed operators.
        dataset_size: an integer, number of equations to be generated.
        min_value: an integer, min value of each operand.
        max_value: an integer, max value of each operand.

        result: a list of tuples of strings (equation, solution).
    """
    sample = []
    for _ in range(dataset_size):
        a, b = random.randint(min_value, max_value), random.randint(min_value, max_value)
        op = random.choice(allowed_operators)
        eq = '{}{}{}'.format(a, op, b)
        res = eval(eq)
        sample.append((eq, str(res)))
    return sample


def sentence_to_ids(sentence, word2id, padded_len, end_symbol, padding_symbol):
    """ Converts a sequence of symbols to a padded sequence of their ids.

      sentence: a string, input/output sequence of symbols.
      word2id: a dict, a mapping from original symbols to ids.
      padded_len: an integer, a desirable length of the sequence.

      result: a tuple of (a list of ids, an actual length of sentence).
    """

    sent_len = len(sentence) + 1
    sent_ids = [word2id[w] for w in sentence[:padded_len - 1]
                ] + [word2id[end_symbol]] + [word2id[padding_symbol]] * (padded_len - sent_len)

    return sent_ids, min(len(sent_ids), sent_len)


def ids_to_sentence(ids, id2word):
    """ Converts a sequence of ids to a sequence of symbols.

          ids: a list, indices for the padded sequence.
          id2word:  a dict, a mapping from ids to original symbols.

          result: a list of symbols.
    """

    return [id2word[i] for i in ids]


def batch_to_ids(sentences, word2id, max_len, end_symbol, padding_symbol):
    """Prepares batches of indices.

       Sequences are padded to match the longest sequence in the batch,
       if it's longer than max_len, then max_len is used instead.

        sentences: a list of strings, original sequences.
        word2id: a dict, a mapping from original symbols to ids.
        max_len: an integer, max len of sequences allowed.

        result: a list of lists of ids, a list of actual lengths.
    """

    max_len_in_batch = min(max(len(s) for s in sentences) + 1, max_len)
    batch_ids, batch_ids_len = [], []
    for sentence in sentences:
        ids, ids_len = sentence_to_ids(sentence, word2id, max_len_in_batch, end_symbol, padding_symbol)
        batch_ids.append(ids)
        batch_ids_len.append(ids_len)
    return batch_ids, batch_ids_len


def batch_to_tensor(batch, sentence_start_token_idx):
    batch_size = len(batch)

    sos_tensor = torch.LongTensor([[sentence_start_token_idx]] * batch_size)
    batch_tensor = torch.LongTensor(batch)

    output = torch.cat((sos_tensor, batch_tensor), dim=1)

    return output


def generate_batches(samples, batch_size=64):
    X, Y = [], []
    for i, (x, y) in enumerate(samples, 1):
        X.append(x)
        Y.append(y)
        if i % batch_size == 0:
            yield X, Y
            X, Y = [], []
    if X and Y:
        yield X, Y
