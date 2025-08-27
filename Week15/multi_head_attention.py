import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert (self.head_dim * num_heads == embed_dim), "embed_dim must be divisible by num_heads"

        self.wq = nn.Linear(embed_dim, embed_dim)
        self.wk = nn.Linear(embed_dim, embed_dim)
        self.wv = nn.Linear(embed_dim, embed_dim)
        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, mask=None):
    
        batch_size = query.size(1)

        Q = self.wq(query).view(-1, batch_size, self.num_heads, self.head_dim).transpose(0, 1) # [B, T, H, Dk]
        K = self.wk(key).view(-1, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        V = self.wv(value).view(-1, batch_size, self.num_heads, self.head_dim).transpose(0, 1)

        # Scaled Dot-Product Attention
        energy = torch.einsum("bthd,bfhd->bhtf", [Q, K]) # [B, H, T_query, T_key]
        
        if mask is not None:
            # Perbaikan: Mask harus di-broadcast ke dimensi [B, H, T_query, T_key]
            # Pastikan mask memiliki dimensi yang benar untuk operasi ini.
            # Mask harus berdimensi [B, 1, Tq, Tk]
            energy = energy.masked_fill(mask == 0, float("-inf"))

        attention_weights = F.softmax(energy / math.sqrt(self.head_dim), dim=-1)
        
        x = torch.einsum("bhtf,bfhd->bthd", [attention_weights, V])
        x = x.transpose(0, 1).contiguous().view(-1, batch_size, self.embed_dim)

        out = self.fc_out(x)
        return out, attention_weights