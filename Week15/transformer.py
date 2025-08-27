import torch
import torch.nn as nn
from transformer_encoder import TransformerEncoder
from transformer_decoder import TransformerDecoder
from util import PAD, BOS, EOS, UNK

class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_pad_id, trg_pad_id, bos_id, eos_id, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_id = src_pad_id
        self.trg_pad_id = trg_pad_id
        self.bos_id = bos_id 
        self.eos_id = eos_id
        self.device = device

    def make_src_mask(self, src):
        # src_mask = (src != self.src_pad_id).unsqueeze(1).unsqueeze(2) # [B, 1, 1, Tsrc]
        return (src != self.src_pad_id).transpose(0, 1).unsqueeze(1).unsqueeze(2)

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_id).transpose(0, 1).unsqueeze(1) # [B, 1, Ttrg]
        trg_len = trg.size(0)
        trg_sub_mask = torch.triu(torch.ones(trg_len, trg_len, device=self.device)).transpose(0, 1)
        trg_mask = (trg_sub_mask.bool() & trg_pad_mask).unsqueeze(1).to(self.device)
        return trg_mask

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        enc_src = self.encoder(src, src_mask)
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        
        return output, attention
    
    @torch.no_grad()
    def greedy_decode(self, src, max_len=50):
        """
        Greedy decoding for Transformer model.
        src: [src_len, batch_size]
        returns predicted tokens.
        """
        self.eval()
        batch_size = src.size(1)

        # Encoder pass
        src_mask = self.make_src_mask(src)
        enc_src = self.encoder(src, src_mask)

        # Decoder starts with <bos> token
        trg_ids = torch.full((1, batch_size), self.bos_id, dtype=torch.long, device=self.device) # Perbaikan: Gunakan self.bos_id
        
        for _ in range(max_len):
            trg_mask = self.make_trg_mask(trg_ids)
            output, _ = self.decoder(trg_ids, enc_src, trg_mask, src_mask)

            # Get the next token (last token in sequence)
            pred_token = output.argmax(2)[-1, :]

            # Append to the sequence
            trg_ids = torch.cat((trg_ids, pred_token.unsqueeze(0)), dim=0)

            # Stop if all batches predict <eos>
            if (pred_token == self.eos_id).all():
                break

        return trg_ids, None