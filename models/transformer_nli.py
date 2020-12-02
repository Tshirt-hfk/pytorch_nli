import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models.utils import LayerNorm


class MultiHeadedAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, k_dim=None, v_dim=None, bias=True, dropout=0.0):
        super(MultiHeadedAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.k_dim = embed_dim // num_heads if k_dim is None else k_dim
        self.v_dim = embed_dim // num_heads if v_dim is None else v_dim

        self.bias = bias

        self.scaling = self.k_dim ** -0.5

        self.dropout = dropout

        self.q_proj = nn.Linear(embed_dim, self.num_heads * self.k_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, self.num_heads * self.k_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, self.num_heads * self.v_dim, bias=bias)

        self.out_proj = nn.Linear(self.num_heads * self.v_dim, embed_dim, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

        if self.bias:
            nn.init.constant_(self.q_proj.bias, 0.)
            nn.init.constant_(self.k_proj.bias, 0.)
            nn.init.constant_(self.v_proj.bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, value, mask=None):
        # len, bsz, embed_dim
        bsz = query.size(1)
        tgt_len = query.size(0)
        src_len = value.size(0)

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q *= self.scaling

        q = q.view(-1, bsz * self.num_heads, self.k_dim).transpose(0, 1)
        k = k.view(-1, bsz * self.num_heads, self.k_dim).transpose(0, 1)
        v = v.view(-1, bsz * self.num_heads, self.v_dim).transpose(0, 1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if mask is not None:
            mask = mask.unsqueeze(0)
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))

        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, self.num_heads * self.v_dim)

        attn = self.out_proj(attn)

        return attn, attn_weights


class PositionwiseFeedForward(nn.Module):

    def __init__(self, embed_dim, ffn_embed_dim, bias=True, dropout=0.0):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim, bias=bias)
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim, bias=bias)
        self.bias = bias
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        if self.bias:
            nn.init.constant_(self.fc1.bias, 0.)
            nn.init.constant_(self.fc2.bias, 0.)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        return x


class TransformerEncoderLayer(nn.Module):

    def __init__(self, args):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            k_dim=args.k_dim,
            v_dim=args.v_dim,
            dropout=args.attention_dropout
        )
        self.self_attn_layer_norm = LayerNorm(args.embed_dim)
        self.ffn = PositionwiseFeedForward(
            embed_dim=args.embed_dim,
            ffn_embed_dim=args.ffn_embed_dim,
            dropout=args.activation_dropout
        )
        self.ffn_layer_norm = LayerNorm(args.embed_dim)
        self.dropout = args.dropout

    def forward(self, x, mask=None):
        residual = x
        x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(query=x, key=x, value=x, mask=mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        residual = x
        x = self.ffn_layer_norm(x)
        x = self.ffn(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        return x


class TransformerDecoderLayer(nn.Module):

    def __init__(self, args):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            k_dim=args.k_dim,
            v_dim=args.v_dim,
            dropout=args.attention_dropout
        )
        self.self_attn_layer_norm = LayerNorm(args.embed_dim)
        self.encoder_attn = MultiHeadedAttention(
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            k_dim=args.k_dim,
            v_dim=args.v_dim,
            dropout=args.attention_dropout
        )
        self.encoder_attn_layer_norm = LayerNorm(args.embed_dim)
        self.ffn = PositionwiseFeedForward(
            embed_dim=args.embed_dim,
            ffn_embed_dim=args.ffn_embed_dim,
            dropout=args.activation_dropout
        )
        self.ffn_layer_norm = LayerNorm(args.embed_dim)
        self.dropout = args.dropout

    def forward(self, x,  encoder_out, mask=None):
        residual = x
        x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(query=x, key=x, value=x, mask=mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        residual = x
        x = self.encoder_attn_layer_norm(x)
        x, _ = self.encoder_attn(query=x, key=encoder_out, value=encoder_out, mask=mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        residual = x
        x = self.ffn_layer_norm(x)
        x = self.ffn(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        return x


class PositionalEncoding(nn.Module):

    def __init__(self, embed_dim, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., embed_dim, 2) *
                             -(math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


class Comparison(nn.Module):

    def __init__(self, embed_dim, num_units):
        super(Comparison, self).__init__()
        self.fc1 = nn.Linear(embed_dim*2, embed_dim)
        self.fc2 = nn.Linear(embed_dim, embed_dim)
        self.fc3 = nn.Linear(embed_dim*2, embed_dim)
        self.fc4 = nn.Linear(embed_dim, embed_dim)
        self.fc5 = nn.Linear(embed_dim*2, embed_dim)
        self.fc6 = nn.Linear(embed_dim, num_units)

    def forward(self, encoding_1, encoding_2, interaction_1, interaction_2):
        x_1 = torch.cat([encoding_1, interaction_1], dim=-1)
        x_1 = self.fc2(torch.relu(self.fc1(x_1)))
        x_1 = x_1.sum(dim=0) / (x_1.size(0) ** 2)

        x_2 = torch.cat([encoding_2, interaction_2], dim=-1)
        x_2 = self.fc4(torch.relu(self.fc3(x_2)))
        x_2 = x_2.sum(dim=0) / (x_2.size(0) ** 2)

        x = torch.cat([x_1, x_2], dim=-1)
        x = self.fc6(torch.relu(self.fc5(x)))
        return x


class TransformerNLI(nn.Module):

    def __init__(self, args):
        super(TransformerNLI, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(
            torch.load('.vector_cache/{}_vectors.pt'.format(args.dataset)))

        self.embedding_proj = nn.Linear(self.embedding.embedding_dim, args.embed_dim)

        self.pe = PositionalEncoding(args.embed_dim, args.dropout)

        self.N = args.N
        self.M = args.M
        self.encoder = nn.ModuleList()
        for i in range(self.N):
            self.encoder.append(TransformerEncoderLayer(args))
        self.interaction = nn.ModuleList()
        for i in range(self.M):
            self.interaction.append(TransformerDecoderLayer(args))

        self.comparison = Comparison(args.embed_dim, args.num_units)

    def forward(self, x):
        encoding_1 = self.pe(self.embedding_proj(self.embedding(x.premise))).transpose(0, 1)
        encoding_2 = self.pe(self.embedding_proj(self.embedding(x.hypothesis))).transpose(0, 1)
        for i in range(self.N):
            encoding_1 = self.encoder[i](encoding_1)
            encoding_2 = self.encoder[i](encoding_2)
        interaction_1 = encoding_1
        interaction_2 = encoding_2
        for i in range(self.N):
            interaction_1 = self.interaction[i](interaction_1, interaction_2)
            interaction_2 = self.interaction[i](interaction_2, interaction_1)
        x = self.comparison(encoding_1, encoding_2, interaction_1, interaction_2)
        return x





