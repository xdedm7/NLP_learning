import torch
from torch import nn
from d2l import torch as d2l
import encoder_decoder
import seq2seq
from transformer_Net import Muliti_head_attention, transformer

embed_size, num_hiddens, num_layers, dropout = 128, 32, 3, 0.05
ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
norm_shape = [32]
batch_size, num_steps = 64, 20
lr, num_epochs, device = 0.005, 100, d2l.try_gpu()
train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
encoder = transformer.Transformer_Encoder(
    vocab_size=len(src_vocab), query_size=num_hiddens,
    key_size=num_hiddens, value_size=num_hiddens,
    norm_shape=norm_shape,num_heads=num_heads,
    num_layers=num_layers,ffn_num_input=ffn_num_input,
    ffn_num_hiddens=ffn_num_hiddens,dropout=dropout,
    num_hiddens=num_hiddens
)
decoder = transformer.Transformer_Decoder(
    vocab_size=len(tgt_vocab), query_size=num_hiddens,
    key_size=num_hiddens, value_size=num_hiddens,
    norm_shape=norm_shape,num_heads=num_heads,
    num_layers=num_layers,ffn_num_input=ffn_num_input,
    ffn_num_hiddens=ffn_num_hiddens,dropout=dropout,
    num_hiddens=num_hiddens
)
net = encoder_decoder.EncoderDecoder(encoder, decoder).to(device)

# net.load_state_dict(torch.load('/home/zzq/Desktop/seq_model/weight/Bahdanau_1719381742.917139.pt'),strict=False)
seq2seq.train_seq2seq(net, train_iter,lr, num_epochs,tgt_vocab,device)

engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .','Do you have a table on the patio ?']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .','Disposez-vous d\'une table sur le patio ?']
for eng, fra in zip(engs, fras):
    translation, attention_weight_seq = seq2seq.predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps, device)
    print(f'{eng} => {translation}, bleu {seq2seq.bleu(translation, fra, k=2):.3f}')