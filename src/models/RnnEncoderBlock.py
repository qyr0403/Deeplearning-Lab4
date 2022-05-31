
import math
import torch
import torch.nn as nn

class RnnEncoderBlock(nn.Module):
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

    def forward(self,x):
            
            """
            assumes x.shape represents (batch_size, input_seq,input_sz)
            for example :  [128,5,24*6]
            """
            bz,seq,_=x.shape
            #h_t=torch.randn_like(input['h_t']).to(x.device)
            h_t=torch.randn(bz,self.hidden_size)
            hidden_seq = []
            
            # h_t (batch_sz,hidden_sz)
                
            for t in range(seq):
                x_t = x[:, t, :]
                
                h_t = torch.tanh(x_t @ self.U_i + h_t @ self.V_i + self.b_i)
                
                hidden_seq.append(h_t.unsqueeze(0))
            
            hidden_seq = torch.cat(hidden_seq, dim=0)
            hidden_seq = hidden_seq.transpose(0, 1).contiguous()

            # hidden_seq: (bz,input_seq,hidden_sz)
            return hidden_seq

