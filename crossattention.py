from math import sqrt
import torch
import torch.nn as nn

#last dimension of q need to be the same as k/v


class CalculateAttention(nn.Module):
    def __init__(self):
        super().__init__()

#change
    # def forward(self, Q, K, V, mask):
    def forward(self, Q, K, V):

        attention = torch.matmul(Q,torch.transpose(K, -1, -2))
        # use mask
        #change
        # attention = attention.masked_fill_(mask, -1e9)
        
        attention = torch.softmax(attention / sqrt(Q.size(-1)), dim=-1)
        attention = torch.matmul(attention,V)
        return attention

class Multi_CrossAttention(nn.Module):

    def __init__(self,hidden_size,all_head_size,head_num):
        super().__init__()
        self.hidden_size    = hidden_size       # input_dim
        self.all_head_size  = all_head_size     # output_dim
        self.num_heads      = head_num         
        self.h_size         = all_head_size // head_num

        assert all_head_size % head_num == 0

        # W_Q,W_K,W_V (hidden_size,all_head_size)
        self.linear_q = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_output = nn.Linear(all_head_size, hidden_size)

        # normalization
        self.norm = sqrt(all_head_size)
        # self.layer_norm=nn.LayerNorm(all_head_size)

    # def print(self):
    #     print(self.hidden_size,self.all_head_size)
    #     print(self.linear_k,self.linear_q,self.linear_v)
    
    # def forward(self,x,y,attention_mask):
    #change
    def forward(self,x,y):

        batch_size = x.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        
        # q_s: [batch_size, num_heads, seq_length, h_size]
        q_s = self.linear_q(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)

        # k_s: [batch_size, num_heads, seq_length, h_size]
        k_s = self.linear_k(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)

        # v_s: [batch_size, num_heads, seq_length, h_size]
        v_s = self.linear_v(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)

        # attention_mask = attention_mask.eq(0)

        # attention = CalculateAttention()(q_s,k_s,v_s,attention_mask)
        #change
        attention = CalculateAttention()(q_s,k_s,v_s)
        # attention : [batch_size , seq_length , num_heads * h_size]
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.h_size)
        
        # output : [batch_size , seq_length , hidden_size]
        # i_feature=attention[:,1,:]
        output = self.linear_output(attention)
        # print(f'the size i_feature of i is {i_feature.shape}')
        i_feature=output[:,1:,:]
        # print(f'the size i_feature of i is {i_feature.shape}')
        icls_token=output[:,0,:]
        # head=nn.Linear(self.hidden_size,20)
        # i_lable=head(icls_token)
        head=nn.Linear(icls_token.size(1),50)
        icls_token=head(icls_token)
        # print(f'I_mge  is {icls_token}')


        return i_feature,icls_token
    
# layer = Multi_CrossAttention(768,768,8)
# layer.print()
# cross_output = layer(modeloutput1,modeloutput2)
# print(cross_output.shape)
# print(cross_output)
