
import math
import torch
import torch.nn as nn

class LstmBlock(nn.Module):
    def __init__(self, input_sz: int, hidden_sz: int):
        super().__init__()
        self.input_size = input_sz
        self.hidden_size = hidden_sz
        
        #i_t
        self.U_i = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.V_i = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i = nn.Parameter(torch.Tensor(hidden_sz))
        
        #f_t
        self.U_f = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.V_f = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_f = nn.Parameter(torch.Tensor(hidden_sz))
        
        #c_t
        self.U_c = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.V_c = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_c = nn.Parameter(torch.Tensor(hidden_sz))
        
        #o_t
        self.U_o = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.V_o = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_o = nn.Parameter(torch.Tensor(hidden_sz))
        
        self.init_weights()


    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self,
                    input
                    ):
            
            """
            assumes x.shape represents (batch_size, sequence_size, input_size)
            """
            x=input['hidden_seq']
            #h_t, c_t =input['h_t'],input['c_t']
            h_t,c_t=(
                torch.randn_like(input['h_t']),
                torch.randn_like(input['c_t'])
            )
            h_t,c_t=h_t.to(x.device),c_t.to(x.device)
            _, seq_sz, _ = x.size()
            hidden_seq = []
            
            #if init_states is None:
            #    h_t, c_t = (
            #        torch.zeros(bs, self.hidden_size).to(x.device),
            #        torch.zeros(bs, self.hidden_size).to(x.device),
            #    )
            #else:
            #h_t, c_t = init_states
                
            for t in range(seq_sz):
                x_t = x[:, t, :]
                
                i_t = torch.sigmoid(x_t @ self.U_i + h_t @ self.V_i + self.b_i)
                f_t = torch.sigmoid(x_t @ self.U_f + h_t @ self.V_f + self.b_f)
                g_t = torch.tanh(x_t @ self.U_c + h_t @ self.V_c + self.b_c)
                o_t = torch.sigmoid(x_t @ self.U_o + h_t @ self.V_o + self.b_o)
                c_t = f_t * c_t + i_t * g_t
                h_t = o_t * torch.tanh(c_t)
                
                hidden_seq.append(h_t.unsqueeze(0))
            
            #reshape hidden_seq p/ retornar
            hidden_seq = torch.cat(hidden_seq, dim=0)
            hidden_seq = hidden_seq.transpose(0, 1).contiguous()
            return {'hidden_seq':hidden_seq,'h_t':h_t,'c_t':c_t}
            #return hidden_seq, (h_t, c_t)


