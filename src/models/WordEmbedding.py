
from torch import nn
import numpy as np
import torch
import pickle
class WordEmbedding(nn.Module):
    def __init__(self,embedding):
        super(WordEmbedding,self).__init__()
        self.Embedding=nn.Embedding(len(embedding),len(embedding[0]))
        self.Embedding.weight.data=embedding.clone().detach()
        self.Embedding.requires_grad=True
        

    def forward(self,input_ids):
        return self.Embedding(input_ids)
        