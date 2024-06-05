import torch
import torch.nn as nn
from torch.autograd import Variable


class RNN(nn.Module):

    '''
        Input Dimension are as follows :
        L : Sequence Length
        B : Batch Size
        E : Encoded vector dim
        N : Number of layers in RNN
        H : Hidden Size

        Input : (L * B)
        Embedding Size : ( L * B * E)
        Hidden Size : ( N * B * H)


    '''

    def __init__(self, input_size, embedding_size, hidden_size, output_size, vocab_size, n_layers=1):
        super(RNN, self).__init__()
        # Parameter Initialization
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.embedding_size = embedding_size

        # Adding different layers
        self.encoder = nn.Embedding(num_embeddings= vocab_size  , embedding_dim= embedding_size)
        self.rnn = nn.RNN(input_size=embedding_size, hidden_size=hidden_size, num_layers= n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)


    def forward(self, input, hidden):
        input = self.encoder(input)
        output, hidden = self.rnn(input, hidden)
        output = self.decoder(output)
        return output, hidden

    def init_hidden(self, batch_size):
        return Variable(torch.zeros(self.n_layers,  batch_size, self.hidden_size))

    def get_embedding(self, input):
        return self.encoder(input)