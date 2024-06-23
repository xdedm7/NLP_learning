import time

import encoder_decoder
import torch.nn as nn
import torch.nn.functional as F
import torch
from d2l import torch as d2l
from tqdm import tqdm
import math
import collections
class Seq2SeqEncoder(encoder_decoder.Encoder):
    """用于序列到序列学习的循环神经网络编码器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_size)#字符编码
        self.encoder = nn.GRU(embed_size, num_hiddens, num_layers,dropout=dropout)
    def forward(self, X, *args):
        # 输出'X'的形状：(batch_size,num_steps,embed_size)
        X = self.embedding(X)
        # 在循环神经网络模型中，第一个轴对应于时间步
        X = X.permute(1, 0, 2)
        # 如果未提及状态，则默认为0
        output, state = self.encoder(X)
        # output的形状:(num_steps,batch_size,num_hiddens)
        # state的形状:(num_layers,batch_size,num_hiddens)
        return output, state
class Seq2SeqDecoder(encoder_decoder.Decoder):
    """用于序列到序列学习的循环神经网络解码器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.decoder = nn.GRU(embed_size+ num_hiddens, num_hiddens, num_layers,dropout=dropout)
        self.dense = nn.Linear(num_hiddens,vocab_size)
    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]
    def forward(self, X, state):
        # 输出'X'的形状：(batch_size,num_steps,embed_size)
        X=X.long()
        X = self.embedding(X).permute(1, 0, 2)
        # 广播context，使其具有与X相同的num_steps
        context = state[-1].repeat(X.shape[0], 1,1)

        X_and_context = torch.cat((X, context), 2)
        output, state = self.decoder(X_and_context, state)
        output = self.dense(output).permute(1, 0, 2)
        return output, state
#@save
# #@save
def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X
class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带遮蔽的softmax交叉熵损失函数"""
    # pred的形状：(batch_size,num_steps,vocab_size)
    # label的形状：(batch_size,num_steps)
    # valid_len的形状：(batch_size,)
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction = 'none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss

#@save
def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """训练序列到序列模型"""
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                     xlim=[10, num_epochs])
    iter_all=0
    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)  # 训练损失总和，词元数量
        for iter,batch in enumerate(tqdm(data_iter,desc=f'Epoch {epoch}/{num_epochs},')):
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                          device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # 强制教学
            Y_hat, _ = net(X, dec_input, X_valid_len)
            # torch.permute()
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()      # 损失函数的标量进行“反向传播”
            d2l.grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
        animator.add(epoch, (metric[0] / metric[1],))
        print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
              f'tokens/sec on {str(device)}')
    d2l.plt.show()
    #torch.save(net.state_dict(), f'weight/seq2seq_300ep_{time.time()}.pt')

    # print(predict_seq2seq(net, 'who are you', src_vocab, tgt_vocab, num_steps=num_steps, device=device))
def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=False):
    """序列到序列模型的预测"""
    # 在预测时将net设置为评估模式
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [
        src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # 添加批量轴
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)

    # 添加批量轴
    dec_X = torch.unsqueeze(torch.tensor(
        [tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        # 我们使用具有预测最高可能性的词元，作为解码器在下一时间步的输入
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # 保存注意力权重（稍后讨论）
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # 一旦序列结束词元被预测，输出序列的生成就完成了
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq


# encoder = Seq2SeqEncoder(vocab_size=28, embed_size=3, num_hiddens=256,
#                          num_layers=2)
# decoder = Seq2SeqDecoder(vocab_size=28, embed_size=3, num_hiddens=256,
#                          num_layers=1)
# encoder.eval()
# decoder.eval()
#
# class Seq2SeqModel(encoder_decoder.EncoderDecoder):
#     def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
#                  dropout=0, **kwargs):
#         super(Seq2SeqModel, self).__init__(**kwargs)
#         self.encoder=Seq2SeqEncoder(vocab_size, embed_size, num_hiddens, num_layers,dropout=dropout)
#         self.decoder=Seq2SeqDecoder(vocab_size, embed_size, num_hiddens, num_layers,dropout=dropout)
#
#     def forward(self, X_enc,X_dec, *args):
#         Output_enc, state = self.encoder(X)
#         Output_dec, state = self.decoder(X_dec, state)
#         return Output_enc, state
#
#
# #batch_size=4,num_step=7
# X = torch.zeros((4, 7), dtype=torch.long)
# output, state = encoder(X)
# state = decoder.init_state(output, state).unsqueeze(dim=0)
# output, state = decoder(X, state)
# # print(output)
#
# X = torch.tensor([[1, 2, 3], [4, 5, 6]])
# sequence_mask(X, torch.tensor([1, 2]))
# def bleu(pred_seq, label_seq, k):  #@save
#     """计算BLEU"""
#     pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
#     len_pred, len_label = len(pred_tokens), len(label_tokens)
#     score = math.exp(min(0, 1 - len_label / len_pred))
#     for n in range(1, k + 1):
#         num_matches, label_subs = 0, collections.defaultdict(int)
#         for i in range(len_label - n + 1):
#             label_subs[' '.join(label_tokens[i: i + n])] += 1
#         for i in range(len_pred - n + 1):
#             if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
#                 num_matches += 1
#                 label_subs[' '.join(pred_tokens[i: i + n])] -= 1
#         score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
#     return score
# embed_size, num_hiddens, num_layers, dropout = 3, 32, 2, 0.1
# batch_size, num_steps = 64, 10
# lr, num_epochs, device = 0.005, 250, d2l.try_gpu()
# train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
# encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers,
#                         dropout=dropout)
# decoder = Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers,
#                         dropout)
# net = d2l.EncoderDecoder(encoder, decoder)
# # net.load_state_dict(torch.load(f'weight/seq2seq_300ep_1718950276.3611968.pt'), strict=False)
# net.to(device)
# train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

# engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
# fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
# for eng, fra in zip(engs, fras):
#     translation, attention_weight_seq = predict_seq2seq(
#         net, eng, src_vocab, tgt_vocab, num_steps, device)
#     print(f'{eng} => {translation}, bleu {bleu(translation, fra, k=2):.3f}')
#
