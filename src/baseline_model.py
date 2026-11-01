'''
Baseline model for the Enron email spam classification experiment
'''

import torch
from torch import nn
from src.text import PAD_ID

class BaselineModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, padding_idx=PAD_ID):
        super().__init__()

        self.padding_idx = padding_idx

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )

        self.classifier = nn.Linear(embedding_dim, 2)

    def forward(self, input_ids):
        # Convert token IDs into embeddings
        embeddings = self.embedding(input_ids)

        # Ignore padding tokens when calculating the average
        mask = (input_ids != self.padding_idx).unsqueeze(-1)
        embeddings = embeddings * mask

        lengths = mask.sum(dim=1).clamp(min=1)
        pooled = embeddings.sum(dim=1) / lengths

        # Return scores for the two classes
        return self.classifier(pooled)