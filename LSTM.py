import torch
from torch import nn
from d2l import torch as d2l
from matplotlib import pyplot as plt
#
# def get_lstm_params(vocab_size, num_hiddens, device):
#     num_inputs = num_outputs= vocab_size
#     def normal(shape):
#         return torch.randn(size=shape, device=device)*0.01
#
#     def three():
#         return (normal((num_inputs, num_hiddens)),
#                 normal((num_hiddens, num_hiddens)),
#                 torch.zeros(num_hiddens, device=device))
#
#     W_xi, W_hi, b_i = three()  # 输入门参数
#     W_xf, W_hf, b_f = three()  # 遗忘门参数
#     W_xo, W_ho, b_o = three()  # 输出门参数
#     W_xc, W_hc, b_c = three()  # 候选记忆元参数
#     # 输出层参数
#     W_hq = normal((num_hiddens, num_outputs))
#     b_q = torch.zeros(num_outputs, device=device)
#     # 附加梯度
#     params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc,
#               b_c, W_hq, b_q]
#     for param in params:
#         param.requires_grad_(True)
#     return params
#
# def init_lstm_state(batch_size, num_hiddens, device):
#     return (torch.zeros((batch_size, num_hiddens), device=device),
#             torch.zeros((batch_size, num_hiddens), device=device))
# def lstm(inputs, state, params):
#     [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c,
#      W_hq, b_q] = params
#     (H, C) = state
#     outputs = []
#     for X in inputs:
#         I = torch.sigmoid((X @ W_xi) + (H @ W_hi) + b_i)
#         F = torch.sigmoid((X @ W_xf) + (H @ W_hf) + b_f)
#         O = torch.sigmoid((X @ W_xo) + (H @ W_ho) + b_o)
#         C_tilda = torch.tanh((X @ W_xc) + (H @ W_hc) + b_c)
#         C = F * C + I * C_tilda
#         H = O * torch.tanh(C)
#         Y = (H @ W_hq) + b_q
#         outputs.append(Y)
#     return torch.cat(outputs, dim=0), (H, C)
#
#
#
# batch_size, num_steps = 32, 28
# train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
#
# vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
# num_epochs, lr = 100, 1
# model = d2l.RNNModelScratch(len(vocab), num_hiddens,
#                             init_state=init_lstm_state,
#                             get_params= get_lstm_params,
#                              forward_fn=lstm)
# model =LSTM(input_size=vocab_size, hidden_size=num_hiddens,
#             num_layers=2, output_size=vocab_size).to(d2l.try_gpu())
# d2l.train_ch8(net=model, lr=lr,
#               num_epochs=num_epochs,
#               device=device,
#               train_iter=train_iter,
#               vocab=vocab,
#               figure_name='LSTM_1.png')
#

import torch

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):

        super().__init__()
        self.device =d2l.try_gpu()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        # 定义LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers).to(torch.device('cuda:0'))
        self.fc = nn.Linear(hidden_size, output_size)
    # 前向传播函数
    def forward(self, inputs,h0=None,c0=None):
        #h0,c0->(num_layers, batch_size, hidden_size)
        # 初始化LSTM的隐藏状态和记忆单元 intput->[step_num,batch_size,onehot_encode]
        if h0 is None and c0 is None:
            h0 = torch.zeros(self.num_layers,inputs.size(0), self.hidden_size).to(inputs.device)
            c0 = torch.zeros(self.num_layers,inputs.size(0), self.hidden_size).to(inputs.device)
        # 通过LSTM层进行前向传播
        inputs = torch.nn.functional.one_hot(inputs.T.long(), 28).type(torch.float32)
        output, (hn, cn) = self.lstm(inputs, (h0, c0))

        # # 取LSTM最后一个时间步的输出作为全连接层的输入
        # output = output[-1, :, :]
        # 将LSTM的输出传入全连接层进行前向传播
        output = self.fc(output)
        return output, hn, cn


batch_size, num_steps,epochs = 32, 35,200
model = LSTM(input_size=28, hidden_size=256, num_layers=2, output_size=28).to(d2l.try_gpu())
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss().to(d2l.try_gpu())
def predict(prefix, num_preds, net, vocab, device):
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    _, H, C = net(get_input())
    for y in prefix[1:]:  # 预热期
        _,H,C = net(get_input(),H,C)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # 预测num_preds步
        y, H,C = net(get_input(),H,C)
        outputs.append(int(y.argmax(dim=2).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])

iter_num=0
loss_all=[]
for epoch in range(epochs):
    for batch in train_iter:
        inputs, targets = batch
        inputs=inputs.float().to(d2l.try_gpu())
        targets=targets.float().to(d2l.try_gpu())
        outputs,H,C = model(inputs)
        loss = criterion(outputs, torch.nn.functional.one_hot(targets.T.long(), 28).type(torch.float32))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        iter_num+=1
        if iter_num%100 == 0:
            loss_all.append(loss.item())
            print(predict('time ',100,model,vocab,device=d2l.try_gpu()))
            print(f'Epoch {epoch+1}/{epochs}.. ')
            print(f'Loss: {loss.item()}')

plt.plot(loss_all, label='loss', color='blue', alpha=0.5)
plt.show()