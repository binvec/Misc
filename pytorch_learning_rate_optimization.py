import matplotlib.pyplot as plt
from copy import deepcopy
import torch
import torch.nn as nn
from torch.optim import SGD
import torch.autograd as ag

torch.manual_seed(123)

x = torch.rand(4, 3)
y = torch.sqrt(x.sum(dim=1)).T.reshape(4, 1)

w = nn.Parameter(torch.randn(3, 1) * 0.05)
w1 = deepcopy(w)
w2 = deepcopy(w)


# training without learning rate optimization
LR = 0.01
loss1 = []
optimizer1_w = SGD([w1,], lr=LR,  momentum=0)
for i in range(50):
    optimizer1_w.zero_grad()
    output = torch.matmul(x, w1)
    loss = (output - y).pow(2).sum()
    loss.backward()
    optimizer1_w.step()
    loss1.append(loss.item())


# training with learning rate optimization
lr = nn.Parameter(torch.tensor([LR,]))
optimizer2_lr = SGD([lr], lr=0.0001, momentum=0)
loss2 = []
lr_list = []
loss = None

for i in range(10):
    optimizer2_lr.zero_grad()
    new_w2 = w2
    for j in range(5):
        output = torch.matmul(x, new_w2)
        loss = (output - y).pow(2).sum()
        new_w2 = new_w2 - lr * ag.grad(loss, new_w2, create_graph=True)[0]
        loss2.append(loss.item())
    w2.data = new_w2.data
    loss.backward(retain_graph=False)
    lr_list.append(lr.item())
    optimizer2_lr.step()

plt.plot(loss1)
plt.plot(loss2)
plt.ylabel('loss')
plt.legend(['w/o learning rate optimization', 'w/ learning rate optimization'])
plt.show()
plt.plot(lr_list)
plt.show()
