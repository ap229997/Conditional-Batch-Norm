# Source : https://github.com/GuessWhatGame/generic/tree/master/data_provider

import numpy as np
from utils.file_handlers import pickle_loader

class GloveEmbeddings(object):

    def __init__(self, file, glove_dim=300):
        self.glove = pickle_loader(file)
        self.glove_dim = glove_dim

    def get_embeddings(self, tokens):
        vectors = []
        for token in tokens:
            token = token.lower().replace("\'s", "")
            if token in self.glove:
                vectors.append(np.array(self.glove[token]))
            else:
                vectors.append(np.zeros((self.glove_dim,)))
        return vectors
