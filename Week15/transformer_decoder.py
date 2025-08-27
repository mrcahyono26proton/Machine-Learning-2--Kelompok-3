import torch
import torch.nn as nn
from multi_head_attention import MultiHeadAttention
from positional_encoding import PositionalEncoding

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, dropout):
        super(DecoderBlock, self).__init__()
        self.self_attention = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.encoder_attention = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, feed_forward_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feed_forward_dim, embed_dim)
        )
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # masked self-attention
        trg_attn, _ = self.self_attention(trg, trg, trg, trg_mask)
        trg = self.norm1(trg + self.dropout(trg_attn))

        # encoder-decoder attention
        enc_attn, attention_weights = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        trg = self.norm2(trg + self.dropout(enc_attn))

        # feed forward
        ff_out = self.feed_forward(trg)
        trg = self.norm3(trg + self.dropout(ff_out))

        return trg, attention_weights

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, feed_forward_dim, dropout, max_len=5000):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, dropout, max_len)
        self.layers = nn.ModuleList([
            DecoderBlock(embed_dim, num_heads, feed_forward_dim, dropout)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        trg = self.embedding(trg)
        trg = self.pos_encoding(trg)
        
        for layer in self.layers:
            trg, attention_weights = layer(trg, enc_src, trg_mask, src_mask)
        
        output = self.fc_out(trg)
        return output, attention_weights