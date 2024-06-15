from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import torch
import math


def train_one_epoch(net, train_iter, loss, optimer, device, use_random_iter):
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)
    for X, Y in train_iter:
        if state is None or use_random_iter:
            state = net.begin_state(device=device, batch_size=X.shape[0])
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                state.detach_()
            else:
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        if isinstance(optimer, torch.optim.Optimizer):
            optimer.zero_grad()
            l.sum().backward()
            d2l.grad_clipping(net, 1)
            optimer.step()
        else:
            l.sum().backward()
            d2l.grad_clipping(net, 1)
            optimer(batch_size=1)
        metric.add(l * y.numel(), y.numel())
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()


def train(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False,predict_Flag=True):
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[5, num_epochs])
    # 初始化
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    for epoch in range(num_epochs):
        ppl, speed = train_one_epoch(
            net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, [ppl])
            print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
            if predict_Flag :
                print(predict('time traveller', 20, net, vocab, device))
    d2l.plt.show()
def predict(prefix, num_preds, net, vocab, device):
    if isinstance(net,nn.Module):
        state = net.init_hidden(batch_size=1).float()
    else:
        state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1)).float()
    for y in prefix[1:]:
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):
        x= get_input()
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])


def predict_nn(prefix,input_size,num_preds,net,vocab,device):
    if isinstance(net,nn.Module):
        state = net.init_hidden(batch_size=1).float()
    else:
        state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[item][1] for item in enumerate(prefix)]
    get_input = lambda: torch.tensor([outputs[-input_size:]], device=device).float()
    for y in prefix[1:]:
        x_=get_input()
        _, state = net(x_, state)
        outputs.append(vocab[y])
    for _ in range(num_preds):
        x=get_input()
        y, state = net(x, state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])


