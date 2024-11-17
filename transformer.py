"""
code by Tae Hwan Jung(Jeff Jung) @graykode, Derek Miller @dmmiller612, modify by shwei
Reference: https://github.com/jadore801120/attention-is-all-you-need-pytorch
           https://github.com/JayParks/transformer
           http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
"""
import math
import torch
import numpy as np
import torch.nn as nn

class PositonEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositonEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.range(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.range(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = torch.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        :param x: [seq_len, batch_size, d_model]
        :return:  x --> [seq_len, batch_size, d_model]
        """
        x += self.pe[:x.size(0), :]
        return self.dropout(x)

def get_attn_pad_mask(seq_q, seq_k):
    """
    :param seq_q: [batch_size, seq_len]
    :param seq_k: [batch_size, seq_len]
    :return: [batch_size, len_q, len_k]
    """
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, len_q, len_k)

def get_attn_subsequence_mask(seq):
    """
    :param seq: [batch_size, tgt_len]
    :return:    [batch_size, tgt_len, tgt_len]
    """
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        """
        :param Q: [batch_size, n_head, len_q, d_k]
        :param K: [batch_size, n_head, len_k, d_k]
        :param V: [batch_size, n_head, len_v(=lenk), d_v]
        :param attn_mask: [batch_size, n_head, seq_len, seq_len]
        :return: context: [batch_size, n_heads, len_q, d_v]
                 attn: [batch_size, n_head, len_q, len_k]
        """
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e-9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.W_K = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.W_V = nn.Linear(d_model, n_heads * d_v, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        """
        :param input_Q: [batch_size, len_q, d_model]
        :param input_K: [batch_size, len_k, d_model]
        :param input_V: [batch_size, len_v(=len_k), d_model]
        :param attn_mask: [batch_size, seq_len, seq_len]
        :return: output: [batch_size, seq_len, d_model]
                 attn:   [batch_size, n_heads, seq_len, seq_len]
        """
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        V = self.W_K(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)
        output = self.fc(context)
        return nn.LayerNorm(d_model).to(device)(residual + output), attn

class PosWiseFeedForward(nn.Module):
    def __init__(self):
        super(PosWiseFeedForward, self).__init__()
        self.fc = nn.ModuleList(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, inputs):
        """
        :param inputs: [batch_size, src_len, src_len]
        :return:
        """
        residual = inputs
        outputs = self.fc(inputs)
        return nn.LayerNorm(d_model).to(device)(residual + outputs)

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PositonEncoding()
    def forward(self, enc_inputs, enc_self_attn_mask):
        """
        :param enc_inputs: [batch_size, src_len, d_model]
        :param enc_self_attn_mask: [batch_size, src_len, src_len]
        :return: enc_outputs: [batch_size, src_len, d_model]
                 attn: [batch_size, n_heads, src_len, src_len]
        """
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                           enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PosWiseFeedForward()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs,
                                                        dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs,
                                                      dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.pos_emb = PositonEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        """
        :param enc_inputs: [batch_size, src_len, src_len]
        :return:  enc_outputs: [batch_size, src_len, d_model]
                  enc_self_attns: [batch_size, n_heads, src_len, src_len]
        """
        enc_outputs = self.src_emb(enc_inputs)
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attns = []
        for layers in self.layer:
            enc_outputs, enc_self_attn = layers(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = PositonEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)]).to(device)

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        """
        :param dec_inputs: [batch_size, tgt_len]
        :param enc_inputs: [batch_size, src_len]
        :param enc_outputs: [batch_size, src_len, d_model]
        :return: dec_outputs: [batch_size, tgt_len, d_model]
                 dec_self_attns: [batch_size, n_heads, tgt_len, tgt_len]
                 dec_enc_attns:  [batch_size, n_heads, tgr_len, src_len]
        """
        dec_outputs = self.tgt_emb(dec_inputs)
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1).to(device)
        dec_self_pad_attn_mask = get_attn_pad_mask(dec_inputs, dec_inputs).to(device)
        dec_self_subsequence_attn_mask = get_attn_subsequence_mask(dec_inputs)
        dec_self_attn_mask = torch.gt((dec_self_pad_attn_mask + dec_self_subsequence_attn_mask),
                                      0).to(device)
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)
        dec_self_attns, dec_enc_attns = [], []
        for layers in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layers(dec_outputs, enc_outputs, dec_self_attn_mask,
                                                              dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder().to(device)
        self.decoder = Decoder().to(device)
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False).to(device)

    def forward(self, enc_inputs, dec_inputs):
        """
        :param enc_inputs: [batch_size, src_len]
        :param dec_inputs: [batch_size, tgt_len]
        :return:  enc_self_attns: [batch_size, n_heads, src_len, src_len]
                  dec_self_attns: [batch_size, n_heads, tgt_len, tgt_len]
                  dec_enc_attns:  [batch_size, n_heads, tgt_len, src_len]
        """
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        dec_logtis = self.projection(dec_outputs)
        return dec_logtis.view(-1, dec_logtis.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns
