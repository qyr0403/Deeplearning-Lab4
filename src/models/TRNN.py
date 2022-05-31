
import torch
import torch.nn as nn
from src.models.RnnEncoder import Encoder
from src.models.RnnDecoder import Decoder

class TRNN(nn.Module):
    def __init__(self, input_sz: int, hidden_sz: int,output_sz:int,output_seq:int):
        super().__init__()
        self.input_sz=input_sz
        self.hidden_sz=hidden_sz
        self.output_sz=output_sz
        self.output_seq=output_seq
        self.encoder=Encoder(input_sz,hidden_sz)
        self.decoder=Decoder(input_sz,hidden_sz,output_sz,output_seq)

    def forward(self,x):
        encoder_output=self.encoder(
            x
        )
        decoder_inputs=torch.randn(
            x.shape[0],self.output_seq,self.input_sz
        )
        decoder_output=self.decoder(
            (encoder_output,decoder_inputs)
        )

        return decoder_output


