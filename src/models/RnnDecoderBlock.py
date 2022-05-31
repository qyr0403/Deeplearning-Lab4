
import math
import torch
import torch.nn as nn

class DecoderBlock(nn.Module):
    def __init__(self, input_sz: int, hidden_sz: int):
        super().__init__()
        self.input_size = input_sz
        self.hidden_size = hidden_sz
        
        #i_t
        self.U_i = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.V_i = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i = nn.Parameter(torch.Tensor(hidden_sz))
        self.init_weights()


    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self,inputs):
            

            encoder_inputs,decoder_inputs=inputs[0],inputs[1]

            """
            encoder_inputs:   [bz,hidden_sz]
            decoder_inputs:   [bz,seq_sz,input_sz]
            """
            _,seq,_=decoder_inputs.shape
            h_t=encoder_inputs
            #h_t=torch.randn_like(input['h_t']).to(x.device)
            #h_t=torch.randn(bz,self.hidden_size)
            decoder_outputs = []
            
            # h_t (batch_sz,hidden_sz)
                
            for t in range(seq):
                x_t = decoder_inputs[:, t, :]
                
                h_t = torch.tanh(x_t @ self.U_i + h_t @ self.V_i + self.b_i)
                
                decoder_outputs.append(h_t.unsqueeze(0))
            
            decoder_outputs = torch.cat(decoder_outputs, dim=0)
            decoder_outputs = decoder_outputs.transpose(0, 1).contiguous()

            # hidden_seq: (bz,input_seq,hidden_sz)
            return (encoder_inputs,decoder_outputs)

