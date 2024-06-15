from d2l import torch as d2l
import torch.nn as nn
import torch
from torch.nn import functional as F
import utils
import time
class GRU_cell_my(nn.Module):
    def __init__(self, input_size, hidden_size,device='cuda:0'):
        super(GRU_cell_my, self).__init__()
        self.input_size = input_size
        self.ouput_size = input_size
        self.hidden_size = hidden_size
        self.W_xz = nn.Parameter(torch.Tensor(input_size, hidden_size))# 更新门参数
        self.W_hz = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_z = nn.Parameter(torch.zeros(self.hidden_size, device=device))
        self.W_xr = nn.Parameter(torch.Tensor(input_size, hidden_size))# 重置门参数
        self.W_hr = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_r = nn.Parameter(torch.zeros(self.hidden_size, device=device))
        self.W_xh = nn.Parameter(torch.Tensor(input_size, hidden_size))# 候选隐状态参数
        self.W_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_h = nn.Parameter(torch.zeros(self.hidden_size, device=device))
        self.W_hq =nn.Parameter(torch.Tensor(self.hidden_size,self.ouput_size))
        self.b_hq = nn.Parameter(torch.zeros(self.ouput_size,device=device))
        self.num_directions=1
    def forward(self,inputs,state):
        H = state
        H=H.to('cuda:0')
        output = []
        for X in inputs:
            Z = torch.sigmoid((X @ self.W_xz) + (H @ self.W_hz) + self.b_z)
            R = torch.sigmoid((X @ self.W_xr) + (H @ self.W_hr) + self.b_r)
            H_tilda = torch.tanh((X @ self.W_xh) + ((R * H) @ self.W_hh) + self.b_h)
            H = Z * H + (1 - Z) * H_tilda
            Y= H @ self.W_hq+self.b_hq
            output.append(Y)
        return torch.cat(output,dim=0),(H)
    def begin_state(self, device, batch_size=1):
        # nn.GRU以张量作为隐状态
        return  torch.randn((num_layers,batch_size,self.hidden_size))


class GRU_torch(nn.Module):
    def __init__(self,  input_size, hidden_size, output_size, n_layers, drop_prob=0.2,device='cuda:0'):
        super(GRU_torch, self).__init__()
        self.hidden_size =hidden_size
        self.n_layers = n_layers
        self.device =device

        self.gru = nn.GRU(input_size, self.hidden_size,n_layers,dropout=drop_prob)
        self.fc =nn.Linear(self.hidden_size,output_size)
        self.relu =nn.ReLU()

    def forward(self,X,h):
        X=torch.unsqueeze(X,dim=0)
        out,h =self.gru(X,h)
        out = self.fc(self.relu(out[-1,:]))
        return out,h
    def init_hidden(self,batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers,batch_size,self.hidden_size).zero_().to(self.device)
        return hidden

def train_model(model, criterion, optimizer, train_loader, epochs=10):
    model.train()  # 设置模型为训练模式
    start_time = time.time()
    train_step=0
    for epoch in range(epochs):
        h = model.init_hidden(batch_size)
        for i, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            model.zero_grad()
            outputs,h = model(inputs.to('cuda:0').float(),h)
            loss = criterion(outputs, targets.float().to('cuda:0'))
            loss.backward()
            optimizer.step()
            train_step+=1
            if (train_step + 1) % 200 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Step [{train_step + 1}], Loss: {loss.item():.4f}')

                print(utils.predict_nn(' our chairs, being his patents ',net=model,num_preds=30,input_size=input_size,vocab=vocab,device='cuda:0'))
            h = h.data
    current_time = time.time()
    print("Total Time Elapsed: {} seconds".format(str(current_time - start_time)))


# 定义GRU层的参数
input_size = 10  # 输入特征的数量
hidden_size = 20 # 隐藏层的大小
num_layers = 1   # GRU的层数
batch_size= 32

train_iter, vocab = d2l.load_data_time_machine(batch_size, input_size)
num_epochs, lr = 60, 0.1


# 实例化GRU模型
#model = GRU(input_size, hidden_size, num_layers)
# model= GRU_cell_my(input_size,hidden_size)
model =GRU_torch(input_size,hidden_size,output_size=input_size,n_layers=1)
model=model.to('cuda:0')

# 定义损失函数和优化器
criterion = nn.MSELoss()  # 均方误差损失，适用于回归任务
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam优化器
train_model(model,criterion,optimizer,train_iter,epochs=100)
