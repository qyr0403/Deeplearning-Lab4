
import torch.nn as nn
from src.models.RnnDecoderBlock import DecoderBlock

class Decoder(nn.Module):
    def __init__(self, input_sz: int, hidden_sz: int,output_sz,output_seq):
        super().__init__()
        self.input_size = input_sz
        self.hidden_size = hidden_sz
        self.output_sz=output_sz
        self.output_seq=output_seq
        self.DecoderLayer=nn.Sequential(
            DecoderBlock(input_sz,hidden_sz),
            DecoderBlock(hidden_sz,hidden_sz)
        )
        self.fc_layer=nn.Linear(
            hidden_sz,output_sz
        )


    def forward(self,inputs):

        #encoder_inputs,decoder_inputs=inputs[0],inputs[1]
        decoder_layer_output=self.DecoderLayer(
            inputs
            )
        # decoder_output :    [batch_sz,output_seq,hidden_sz]
        decoder_output=self.fc_layer(
            decoder_layer_output[1]
        )
        #output:    [batch_sz,output_seq,output_sz]
        
        return decoder_output

