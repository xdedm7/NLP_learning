# from d2l import torch as d2l
# import torch.nn as nn
# import torch
# from torch.nn import functional as F
# import utils
# import time
# class GRU_cell_my(nn.Module):
#     def __init__(self, input_size, hidden_size,device='cuda:0'):
#         super(GRU_cell_my, self).__init__()
#         self.input_size = input_size
#         self.ouput_size = input_size
#         self.hidden_size = hidden_size
#         self.W_xz = nn.Parameter(torch.Tensor(input_size, hidden_size))# 更新门参数
#         self.W_hz = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
#         self.b_z = nn.Parameter(torch.zeros(self.hidden_size, device=device))
#         self.W_xr = nn.Parameter(torch.Tensor(input_size, hidden_size))# 重置门参数
#         self.W_hr = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
#         self.b_r = nn.Parameter(torch.zeros(self.hidden_size, device=device))
#         self.W_xh = nn.Parameter(torch.Tensor(input_size, hidden_size))# 候选隐状态参数
#         self.W_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
#         self.b_h = nn.Parameter(torch.zeros(self.hidden_size, device=device))
#         self.W_hq =nn.Parameter(torch.Tensor(self.hidden_size,self.ouput_size))
#         self.b_hq = nn.Parameter(torch.zeros(self.ouput_size,device=device))
#         self.num_directions=1
#     def forward(self,inputs,state):
#         H = state
#         H=H.to('cuda:0')
#         output = []
#         for X in inputs:
#             Z = torch.sigmoid((X @ self.W_xz) + (H @ self.W_hz) + self.b_z)
#             R = torch.sigmoid((X @ self.W_xr) + (H @ self.W_hr) + self.b_r)
#             H_tilda = torch.tanh((X @ self.W_xh) + ((R * H) @ self.W_hh) + self.b_h)
#             H = Z * H + (1 - Z) * H_tilda
#             Y= H @ self.W_hq+self.b_hq
#             output.append(Y)
#         return torch.cat(output,dim=0),(H)
#     def begin_state(self, device, batch_size=1):
#         # nn.GRU以张量作为隐状态
#         return  torch.randn((num_layers,batch_size,self.hidden_size))
#
#
# class GRU_torch(nn.Module):
#     def __init__(self,  input_size, hidden_size, output_size, n_layers, drop_prob=0.2,device='cuda:0'):
#         super(GRU_torch, self).__init__()
#         self.hidden_size =hidden_size
#         self.n_layers = n_layers
#         self.device =device
#
#         self.gru = nn.GRU(input_size, self.hidden_size,n_layers,dropout=drop_prob)
#         self.fc =nn.Linear(self.hidden_size,output_size)
#         self.relu =nn.ReLU()
#
#     def forward(self,X,h):
#         X=torch.unsqueeze(X,dim=0)
#         out,h =self.gru(X,h)
#         out = self.fc(self.relu(out[-1,:]))
#         return out,h
#     def init_hidden(self,batch_size):
#         weight = next(self.parameters()).data
#         hidden = weight.new(self.n_layers,batch_size,self.hidden_size).zero_().to(self.device)
#         return hidden
#
# def train_model(model, criterion, optimizer, train_loader, epochs=10):
#     model.train()  # 设置模型为训练模式
#     start_time = time.time()
#     train_step=0
#     for epoch in range(epochs):
#         h = model.init_hidden(batch_size)
#         for i, (inputs, targets) in enumerate(train_loader):
#             optimizer.zero_grad()
#             model.zero_grad()
#             outputs,h = model(inputs.to('cuda:0').float(),h)
#             loss = criterion(outputs, targets.float().to('cuda:0'))
#             loss.backward()
#             optimizer.step()
#             train_step+=1
#             if (train_step + 1) % 200 == 0:
#                 print(f'Epoch [{epoch + 1}/{epochs}], Step [{train_step + 1}], Loss: {loss.item():.4f}')
#
#                 print(utils.predict_nn(' our chairs, being his patents ',net=model,num_preds=30,input_size=input_size,vocab=vocab,device='cuda:0'))
#             h = h.data
#     current_time = time.time()
#     print("Total Time Elapsed: {} seconds".format(str(current_time - start_time)))
#
#
# # 定义GRU层的参数
# input_size = 10  # 输入特征的数量
# hidden_size = 20 # 隐藏层的大小
# num_layers = 1   # GRU的层数
# batch_size= 32
#
# train_iter, vocab = d2l.load_data_time_machine(batch_size, input_size)
# num_epochs, lr = 60, 0.1
#
#
# # 实例化GRU模型
# #model = GRU(input_size, hidden_size, num_layers)
# # model= GRU_cell_my(input_size,hidden_size)
# model =GRU_torch(input_size,hidden_size,output_size=input_size,n_layers=1)
# model=model.to('cuda:0')
#
# # 定义损失函数和优化器
# criterion = nn.MSELoss()  # 均方误差损失，适用于回归任务
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam优化器
# train_model(model,criterion,optimizer,train_iter,epochs=100)


import time

import torch
from torch import nn
from d2l import torch as d2l
from matplotlib import pyplot as plt
import torch


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size,num_layers, output_size):
        super(GRU, self).__init__()
        self.device =d2l.try_gpu()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        # 定义LSTM层
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, inputs,h0=None):
        # 初始化LSTM的隐藏状态和记忆单元 intput->[step_num,batch_size,onehot_encode]
        if h0 is None:
            h0 = torch.zeros(self.num_layers,inputs.size(0), self.hidden_size).to(inputs.device)

        # 通过LSTM层进行前向传播
        inputs = torch.nn.functional.one_hot(inputs.T.long(), 28).type(torch.float32)
        output, H = self.gru(inputs, h0)
        output = self.fc(output)
        return output,H

def predict(prefix, num_preds, net, vocab, device):
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    _, H = net(get_input())
    for y in prefix[1:]:  # 预热期
        _,H = net(get_input(),H)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # 预测num_preds步
        y, H = net(get_input(),H)
        outputs.append(int(y.argmax(dim=2).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])

batch_size, num_steps,epochs = 32, 35,300
model = GRU(input_size=28, hidden_size=256, num_layers=2, output_size=28).to(d2l.try_gpu())
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss().to(d2l.try_gpu())
iter_num=0
loss_all=[]
strat_time= time.time()
for epoch in range(epochs):

    #tqdm = tqdm(train_iter, total=len(train_iter), position=0, leave=True,desc=f'Epoch {epoch+1}/{epochs},Loss: {loss.item()}')
    for batch in train_iter:
        inputs, targets = batch
        inputs=inputs.float().to(d2l.try_gpu())
        targets=targets.float().to(d2l.try_gpu())
        outputs,H = model(inputs)
        loss = criterion(outputs,torch.nn.functional.one_hot(targets.T.long(), 28).type(torch.float32))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        iter_num+=1
        if iter_num%100 == 0:
            loss_all.append(loss.item())
    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
    print(predict('time ',100,model,vocab,device=d2l.try_gpu()))

end_time = time.time()
print(f'Total time: {end_time-strat_time}')
plt.plot(loss_all, label='loss', color='blue', alpha=0.5)
plt.grid(True)
plt.xlabel('iter_num')
plt.ylabel('loss')
plt.show()



