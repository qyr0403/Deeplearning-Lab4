
import math
import torch
import torch.nn as nn
from models.LSTM import LstmBlock
from models.WordEmbedding import WordEmbedding

class ShoppingLSTM(nn.Module):
    def __init__(self,hidden_sz: int,output_sz:int,embedding):
        super().__init__()
        self.input_size = len(embedding[0])
        self.hidden_size = hidden_sz
        self.output_sz=output_sz
        self.embedding_layer=WordEmbedding(embedding)
        self.lstm_layer=nn.Sequential(
            LstmBlock(self.input_size,hidden_sz),
            *[LstmBlock(hidden_sz,hidden_sz)]*11
        )
        self.fc_layer=nn.Linear(hidden_sz,output_sz)

    def forward(self,x):
        """
        x.shape (batch_size, sequence_size)
        """
        embedding=self.embedding_layer(x)
        """
        embedding.shape (batch_size, sequence_size, embedding_dim)
        """
        init_ht=torch.zeros(
            embedding.shape[0],self.hidden_size
        ).to(x.device)
        init_ct=torch.zeros(
            embedding.shape[0],self.hidden_size
        ).to(x.device)
        inputs={
            'hidden_seq':embedding,
            'h_t':init_ht,
            'c_t':init_ct
        }
        final_hiddenstate=self.lstm_layer(inputs)['h_t']
        """
        final_hiddenstate.shape (batch_size,hidden_sz)
        """
        output=self.fc_layer(final_hiddenstate)
        """
        output.shape  (batch_size,output_sz)
        """
        return output



