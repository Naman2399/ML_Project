import random
import string

import torch
import unidecode
from torch.autograd import Variable


class DatasetShakespeare :

    def __init__(self, chunk_len, batch_size):

        # Initialize chunk len
        self.chunk_len = chunk_len
        self.describe_dataset()
        self.batch_size = batch_size

    def describe_dataset(self):

        self.all_characters = string.printable
        # code to find length of all_characters and storing the value in n_characters
        self.n_characters = len(self.all_characters)
        print("Character count :", self.n_characters)

        # code to convert unicode characters into plain ASCII.
        self.file = unidecode.unidecode(open('D:\\Placement\\ML_Project\\dataset\\data\\shakespeare.txt').read())

        # code to find length of the file
        self.file_len = len(self.file)
        print("File length : ", self.file_len)

        # print some few words from file
        print("Some words from file : ", self.file[:100])

    def random_chunk(self):
        '''

        Returns: Here will return the string from file whose length is chunk_len

        '''

        start_index = random.randint(0, self.file_len - self.chunk_len)  # Initializing the starting index value of the big string
        end_index = start_index + self.chunk_len + 1  # Initializing the ending index of the string
        return self.file[start_index:end_index]  # returning the chunk

    def char_tensor(self, string):
        # Tensor is a array
        tensor = torch.zeros(len(string)).long()
        for c in range(len(string)):
            tensor[c] = self.all_characters.index(string[c])
        return Variable(tensor)

    def random_training_set(self):
        chunk = self.random_chunk()
        inp = self.char_tensor(chunk[:-1])
        target = self.char_tensor(chunk[1:])
        return inp, target

    def random_training_batch(self):
        '''

        Returns: Will return number of inputs and targets of Dim : ( L * B) where L : Sequence Length and B : Batch Size

        '''
        input_batch = []
        target_batch = []
        for i in range(self.batch_size):
            input, target = self.random_training_set()
            input_batch.append(input)
            target_batch.append(target)

        input_batch = torch.stack(input_batch)
        target_batch = torch.stack(target_batch)
        input_batch = torch.reshape(input_batch, (input_batch.shape[1], input_batch.shape[0]))
        target_batch = torch.reshape(target_batch, (target_batch.shape[1], target_batch.shape[0]))

        return input_batch, target_batch


def load_dataset(args) :
    return DatasetShakespeare(args.chunk, args.batch)