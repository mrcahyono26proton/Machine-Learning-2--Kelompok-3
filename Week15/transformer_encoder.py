import torch
import torch.nn as nn
from multi_head_attention import MultiHeadAttention
from positional_encoding import PositionalEncoding

class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, dropout):
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, feed_forward_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feed_forward_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # self-attention
        attn_out, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))

        # feed forward
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, feed_forward_dim, dropout, max_len=5000):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, dropout, max_len)
        self.layers = nn.ModuleList([
            EncoderBlock(embed_dim, num_heads, feed_forward_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, src_mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x