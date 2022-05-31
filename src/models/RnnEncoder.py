

import torch
import torch.nn as nn
from src.models.RnnEncoderBlock import EncoderBlock
class Encoder(nn.Module):
    def __init__(self, input_sz: int, hidden_sz: int):
        super().__init__()
        self.input_size = input_sz
        self.hidden_size = hidden_sz
        self.EncoderLayer=nn.Sequential(
            EncoderBlock(input_sz,hidden_sz),
            EncoderBlock(hidden_sz,hidden_sz)
        )


    def forward(self,x):
        encoder_layer_output=self.EncoderLayer(
            x
        )
        encoder_output=torch.mean(encoder_layer_output,dim=1)
        return encoder_output
        # [batch_sz,hidden_sz]

