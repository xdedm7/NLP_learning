from torch import nn
from transformer_Net import position_encoding, Muliti_head_attention
import torch
import math
#@save FFN
class PositionWiseFFN(nn.Module):
    """基于位置的前馈网络"""

    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
                 **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, x):
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        return x


#@save
class AddNorm(nn.Module):
    """残差连接后进行层规范化"""

    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


#@save
class EncoderBlock(nn.Module):
    """Transformer编码器块"""

    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = Muliti_head_attention.MultiHeadAttention(qurey_size=query_size,
                                                                  key_size=key_size,
                                                                  value_size=value_size,
                                                                  num_hidden=num_hiddens,
                                                                  num_heads=num_heads,
                                                                  dropout=dropout)
        self.add_norm = AddNorm(normalized_shape=norm_shape, dropout=dropout)
        #ffn_num_input=ffn_num_output=512
        self.fnn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.add_norm2 = AddNorm(normalized_shape=norm_shape, dropout=dropout)

    def forward(self, x, value_len):
        x = self.add_norm(x, self.attention(x, x, x, value_len))
        x = self.add_norm2(x, self.fnn(x))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, norm_shape, dropout, num_heads,
                 ffn_num_input, ffn_num_hidden, i, use_bias=False, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i

        self.attention_1 = Muliti_head_attention.MultiHeadAttention(qurey_size=query_size,
                                                                    key_size=key_size,
                                                                    value_size=value_size,
                                                                    num_hidden=num_hiddens,
                                                                    num_heads=num_heads,
                                                                    dropout=dropout)
        self.add_norm1 = AddNorm(normalized_shape=norm_shape, dropout=dropout)
        self.attention_2 = Muliti_head_attention.MultiHeadAttention(qurey_size=query_size,
                                                                    key_size=key_size,
                                                                    value_size=value_size,
                                                                    num_hidden=num_hiddens,
                                                                    num_heads=num_heads,
                                                                    dropout=dropout)
        self.add_norm2 = AddNorm(normalized_shape=norm_shape, dropout=dropout)
        self.fnn = PositionWiseFFN(ffn_num_input, ffn_num_hidden, num_hiddens)
        self.add_norm3 = AddNorm(normalized_shape=norm_shape, dropout=dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        # 训练阶段，输出序列的所有词元都在同一时间处理，
        # 因此state[2][self.i]初始化为None。
        # 预测阶段，输出序列是通过词元一个接着一个解码的，
        # 因此state[2][self.i]包含着直到当前时间步第i个块解码的输出表示
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values
        if self.training:
            batch_size, num_steps, _ = X.shape
            dec_valid_lens = torch.arange(
                1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None
        # 自注意力
        X = self.add_norm1(self.attention_1(X, key_values, key_values, dec_valid_lens), X)
        # 编码器－解码器注意力
        X = self.add_norm2(self.attention_2(X, enc_outputs, enc_outputs, enc_valid_lens), X)
        X = self.add_norm3(self.fnn(X), X)
        return X,state


class Transformer_Encoder(nn.Module):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(Transformer_Encoder, self).__init__()
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.PositionEnc= position_encoding.PositionalEncoding(self.num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(
                f'EncoderBlock_{i}',
                EncoderBlock(key_size, query_size, value_size, num_hiddens,
                                    norm_shape=norm_shape,ffn_num_input=ffn_num_input,
                                    ffn_num_hiddens=ffn_num_hiddens,num_heads=num_heads,
                                    dropout=dropout,use_bias=use_bias)
            )

    def forward(self, X, valid_lens, *args):
        X =self.PositionEnc(self.embedding(X)*math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i ,blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i]=blk.attention.attention.attention_weights
        return X
class Transformer_Decoder(nn.Module):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(Transformer_Decoder, self).__init__()
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.PositionEnc= position_encoding.PositionalEncoding(self.num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(
                f'DecoderBlock_{i}',DecoderBlock(key_size, query_size, value_size, num_hiddens,
                                                 norm_shape=norm_shape,ffn_num_input=ffn_num_input,i=i,
                                                 ffn_num_hidden=ffn_num_hiddens,num_heads=num_heads,
                                                 dropout=dropout,use_bias=use_bias)
            )
        self.dense = nn.Linear(num_hiddens, vocab_size)
    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]
    def forward(self, X,state):
        X =self.PositionEnc(self.embedding(X)*math.sqrt(self.num_hiddens))
        # self.attention_weights = [None] * len(self.blks)
        self._attention_weights = [[None] * len(self.blks) for _ in range(2)]
        for i ,blk in enumerate(self.blks):
            X,state = blk(X, state)
            # 解码器自注意力权重
            self._attention_weights[0][i]=blk.attention_1.attention.attention_weights
            # “编码器－解码器”自注意力权重
            self._attention_weights[1][i]=blk.attention_2.attention.attention_weights

        return self.dense(X), state
    @property
    def attention_weights(self):
        return self._attention_weights
