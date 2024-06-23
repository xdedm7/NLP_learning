import torch
from torch import nn
from d2l import torch as d2l
import encoder_decoder
import seq2seq
import Muliti_head_attention
class AttentionDecoder(encoder_decoder.Decoder):
    """带有注意力机制解码器的基本接口"""
    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError

class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        # self.attention = d2l.AdditiveAttention(
        #     num_hiddens, dropout)
        # self.attention = d2l.DotProductAttention(
        #     dropout=dropout)
        self.attention = Muliti_head_attention.MultiHeadAttention(num_hiddens,
                                                                  num_heads=32,
                                                                  dropout=dropout
                                                                  )
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(
            embed_size + num_hiddens, num_hiddens, num_layers,
            dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)
    # outputs的形状为(batch_size，num_steps，num_hiddens).
    # hidden_state的形状为(num_layers，batch_size，num_hiddens)
    def init_state(self, enc_outputs, enc_valid_lens, *args):
        outputs, hidden_state = enc_outputs
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)

    def forward(self, X, state):
        # enc_outputs的形状为(batch_size,num_steps,num_hiddens).
        # hidden_state的形状为(num_layers,batch_size, num_hiddens)
        enc_outputs, hidden_state, enc_valid_lens = state
        # 输出X的形状为(num_steps,batch_size,embed_size)
        X = self.embedding(X).permute(1, 0, 2)
        outputs, self._attention_weights = [], []
        for x in X:
            # query的形状为(batch_size,1,num_hiddens)
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            # context的形状为(batch_size,1,num_hiddens)
            #AdditiveAttention
            # context = self.attention(
            #     query, enc_outputs, enc_outputs, enc_valid_lens)
            #DotproductAttention
            context = self.attention(query,enc_outputs,enc_outputs,enc_valid_lens)
            # 在特征维度上连结
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
            # 将x变形为(1,batch_size,embed_size+num_hiddens)
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            outputs.append(out)
            # self._attention_weights.append(self.attention.attention_weights)
        # 全连接层变换后，outputs的形状为
        # (num_steps,batch_size,vocab_size)
        outputs = self.dense(torch.cat(outputs, dim=0))
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state,
                                          enc_valid_lens]
    @property
    def attention_weights(self):
        return self._attention_weights


embed_size, num_hiddens, num_layers, dropout = 201, 32, 2, 0.05
batch_size, num_steps = 64, 10
lr, num_epochs, device = 0.005, 250, d2l.try_gpu()

train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
encoder = seq2seq.Seq2SeqEncoder(
    len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
decoder = Seq2SeqAttentionDecoder(
    len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
net = encoder_decoder.EncoderDecoder(encoder, decoder).to(device)
#net.load_state_dict(torch.load('/home/zzq/Desktop/seq_model/weight/seq2seq_300ep_1719106065.1140518.pt'),strict=False)
seq2seq.train_seq2seq(net, train_iter,lr, num_epochs,tgt_vocab,device)
# engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
# fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
# for eng, fra in zip(engs, fras):
#     translation, attention_weight_seq = seq2seq.predict_seq2seq(
#         net, eng, src_vocab, tgt_vocab, num_steps, device)
#     print(f'{eng} => {translation}, bleu {seq2seq.bleu(translation, fra, k=2):.3f}')
#
