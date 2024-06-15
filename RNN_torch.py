import  utils
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import torch

var = nn.GRU()
class RNN_model(nn.Module):
    def __init__(self,rnn_layer,vocab_size,**kwargs):
        super(RNN_model, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size=vocab_size
        self.num_hiddens = self.rnn.hidden_size
        self.rnn.forward(torch.ones((1,1,vocab_size)))
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)
    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # nn.GRU以张量作为隐状态
            return  torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens),
                                device=device)
        else:
            # nn.LSTM以元组作为隐状态
            return (torch.zeros((
                self.num_directions * self.rnn.num_layers,
                batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                        self.num_directions * self.rnn.num_layers,
                        batch_size, self.num_hiddens), device=device))
batch_size, num_steps = 32, 16
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
num_epochs, lr = 150, 0.1

num_hiddens = 256

rnn_layer = nn.RNN(input_size=len(vocab), hidden_size=num_hiddens)


model = RNN_model(rnn_layer, len(vocab))
model=model.to('cuda:0')
utils.train(model, train_iter, vocab, lr, num_epochs, device='cuda:0')