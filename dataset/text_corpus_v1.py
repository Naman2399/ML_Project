import datasets
import torchtext
import torch

from torchtext. data import get_tokenizer
from torchtext. vocab import build_vocab_from_iterator


def convert_to_batches(data, batch_size) :
    num_batches = data.shape[0] // batch_size
    data = data[:num_batches * batch_size]
    data = data.view(batch_size, num_batches)
    return data
def get_data_to_tensor(dataset, vocab):
    data = []
    for example in dataset:
        if example['tokens']:
            tokens = example['tokens'].append('<eos>')
            tokens = [vocab[token] for token in example['tokens']]
            data.extend(tokens)
    data = torch.LongTensor(data)
    return data

def describe_dataset(batch_size = 128) :

    '''

    Returns: train_batch, valid_batch, test_batch : Dim (batch_size * number of batches)
             tokenizer,
             vocab

    '''

    dataset = datasets.load_dataset('wikitext', 'wikitext-2-raw-v1')
    print("Dataset Information")
    print(dataset)

    # Dataset Details
    print("Train : ", dataset['train'].num_rows)
    print("Test :", dataset['test'].num_rows)
    print("Validation : ", dataset['validation'].num_rows)

    # Some samples
    for i in range(10) :
        print(f"Training Samples {i + 1} : ", dataset['train'][i]['text'])

    # Tokenizing Dataset
    tokenizer = get_tokenizer("basic_english")
    tokenize_data = lambda example, tokenizer : {'tokens': tokenizer(example['text'])}

    # Applying function to each element in dataset (using tokenize_data) and fn_kwargs :
    # here tokenizer is the key for the function arguments what to map

    tokenized_dataset = dataset.map(tokenize_data, remove_columns=['text'], fn_kwargs={'tokenizer': tokenizer})
    # Some samples
    for i in range(10) :
        print(f"Training Samples {i + 1} : ", tokenized_dataset['train'][i]['tokens'])

    # Constructing Vocabulary
    vocab = build_vocab_from_iterator(tokenized_dataset['train']['tokens'], min_freq=3)
    vocab.insert_token('<unk>', 0)
    vocab.insert_token('<eos>', 1)
    vocab.set_default_index(vocab['<unk>'])
    print("Vocabulary Size : ",len(vocab))
    # Print first 10 values in vocabulary
    print(vocab.get_itos()[:10]) # List mapping indices to tokens.

    # Converting dataset to tensor format
    train_data = get_data_to_tensor(tokenized_dataset['train'], vocab)
    valid_data = get_data_to_tensor(tokenized_dataset['validation'], vocab)
    test_data = get_data_to_tensor(tokenized_dataset['test'], vocab)


    print("Train Data : ", type(train_data), train_data.shape)
    print("Validation Data : ", type(valid_data), valid_data.shape)
    print("Test Data : ", type(test_data), test_data.shape)

    # Converting to batches
    train_batch = convert_to_batches(train_data, batch_size)
    valid_batch = convert_to_batches(valid_data, batch_size)
    test_batch = convert_to_batches(test_data, batch_size)

    print("Train Data : ", type(train_batch), train_batch.shape)
    print("Validation Data : ", type(valid_batch), valid_batch.shape)
    print("Test Data : ", type(test_batch), test_batch.shape)

    return train_batch, valid_batch, test_batch, tokenizer, vocab

def load_dataset(args) :
    return describe_dataset()