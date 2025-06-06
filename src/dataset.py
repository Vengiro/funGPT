import random
import torch

import numpy as np

class CustomDataLoader:
    def __init__(self, tokenDataset, block_size, batch_size=32, shuffle=True):

        self.tokenDataset = tokenDataset
        self.batch_size = batch_size
        self.block_size = block_size
        self.shuffle = shuffle
        self.length = len(tokenDataset) - block_size * batch_size
        self.current_index = 0

    def __len__(self):
        # total tokens available for start positions
        total_tokens = len(self.tokenDataset) - self.block_size
        # how many full batches you can get
        return total_tokens // (self.batch_size * self.block_size)

    def __iter__(self):
        if self.shuffle:
            # Random offset to start the iteration
            self.current_index = random.randint(0, self.block_size * self.batch_size - 1)
        else:
            self.current_index = 0
        return self


    def __next__(self):
        if  self.current_index >= self.length:
            raise StopIteration


        idx = self.current_index
        tokens = self.tokenDataset[idx:idx + self.block_size * self.batch_size + 1]
        tokens = torch.tensor(tokens, dtype=torch.long)
        batch_x = tokens[:-1].view(self.batch_size, self.block_size)
        batch_y = tokens[1:].view(self.batch_size, self.block_size)
        self.current_index += self.batch_size * self.block_size

        return batch_x, batch_y

class FileTokenizer:
    def __init__(self, data_path, encoder):
        self.data_path = data_path
        self.encoder = encoder


    def encode_save(self, output_path):
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = f.read()

        tokens = self.encoder.encode(data)
        tokens_array = np.array(tokens, dtype=np.uint16)
        np.save(output_path, tokens_array)

        return tokens_array

    def load_tokens(self, tokens_path):
        tokens = np.load(tokens_path, allow_pickle=True)
        return tokens