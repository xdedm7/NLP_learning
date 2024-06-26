import time

import torch
from torch import nn
from d2l import torch as d2l
from matplotlib import pyplot as plt
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
        self.embedding = nn.Embedding(input_size, input_size)
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)
    # 前向传播函数
    def forward(self, inputs,h0=None,c0=None):
        #h0,c0->(num_layers, batch_size, hidden_size)
        # 初始化LSTM的隐藏状态和记忆单元 intput->[step_num,batch_size,onehot_encode]
        if h0 is None and c0 is None:
            h0 = torch.zeros(self.num_layers,inputs.size(0), self.hidden_size).to(inputs.device)
            c0 = torch.zeros(self.num_layers,inputs.size(0), self.hidden_size).to(inputs.device)
        # 通过LSTM层进行前向传播
        inputs=self.embedding(inputs.T.long())
        #inputs = torch.nn.functional.one_hot(inputs.T.long(), 28).type(torch.float32)
        output, (hn, cn) = self.lstm(inputs, (h0, c0))
        
        # # 取LSTM最后一个时间步的输出作为全连接层的输入
        # output = output[-1, :, :]
        # 将LSTM的输出传入全连接层进行前向传播
        output = self.fc(output)
        return output, hn, cn


batch_size, num_steps,epochs = 32, 35,300
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
strat_time= time.time()
for epoch in range(epochs):
    #tqdm = tqdm(train_iter, total=len(train_iter), position=0, leave=True,desc=f'Epoch {epoch+1}/{epochs},Loss: {loss.item()}')
    for batch in train_iter:
        inputs, targets = batch
        inputs=inputs.float().to(d2l.try_gpu())
        targets=targets.float().to(d2l.try_gpu())
        outputs,H,C = model(inputs)
        loss = criterion(outputs,torch.nn.functional.one_hot(targets.T.long(), 28).type(torch.float32))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        iter_num+=1
        if iter_num%100 == 0:
            loss_all.append(loss.item())
    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
    print(predict('time traveller',100,model,vocab,device=d2l.try_gpu()))

end_time = time.time()
print(f'Total time: {end_time-strat_time}')
plt.plot(loss_all, label='loss', color='blue', alpha=0.5)
plt.grid(True)
plt.xlabel('iter_num')
plt.ylabel('loss')
plt.show()