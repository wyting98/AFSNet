import torch
import torch.nn as nn
import math
import numpy as np


input = torch.tensor([[-1.0606,  1.5613,  1.2007, -0.2481],
                      [-1.9652, -0.4367, -0.0645, -0.5104],
                      [ 0.1011, -0.5904,  0.0243,  0.1002]])

target = torch.tensor([0, 2, 1])

e1 = math.exp(-1.0606) + math.exp(1.5613) + math.exp(1.2007) + math.exp(-0.2481)

e2 = math.exp(-1.9652) + math.exp(-0.4367) + math.exp(-0.0645) + math.exp(-0.5104)

e3 = math.exp( 0.1011) + math.exp(-0.5904) + math.exp(0.0243) + math.exp(0.1002)


Cross_loss = nn.CrossEntropyLoss()  # 对于batch个数据，nn.CrossEntropyLoss()算出来的是batch个loss的均值
print(Cross_loss(input, target))
print(-math.log(math.exp(-1.0606)/e1))
print(-math.log(math.exp(-0.0645)/e2))
print(-math.log(math.exp(-0.5904)/e3))

print(-(math.log(math.exp(-1.0606)/e1) + math.log(math.exp(-0.0645)/e2) + math.log(math.exp(-0.5904)/e3)))
print(-(math.log(math.exp(-1.0606)/e1) + math.log(math.exp(-0.0645)/e2) + math.log(math.exp(-0.5904)/e3))/3)

fea1 = torch.tensor([[[[-1.0606,  -1.0606,  -1.0606], [-1.0606,  -1.0606,  -1.0606], [-1.0606,  -1.0606,  -1.0606]],
                      [[1.5613,  1.5613,  1.5613], [1.5613,  1.5613,  1.5613], [1.5613,  1.5613,  1.5613]],
                      [[1.2007,  1.2007,  1.2007], [1.2007,  1.2007,  1.2007], [1.2007,  1.2007,  1.2007]],
                      [[-0.2481,  -0.2481,  -0.2481], [-0.2481,  -0.2481,  -0.2481], [-0.2481,  -0.2481,  -0.2481]]],
                     [[[1.0606, 1.0606, 1.0606], [1.0606, 1.0606, 1.0606], [1.0606, 1.0606, 1.0606]],
                      [[-1.5613, -1.5613, -1.5613], [-1.5613, -1.5613, -1.5613], [-1.5613, -1.5613, -1.5613]],
                      [[-1.2007, -1.2007, -1.2007], [-1.2007, -1.2007, -1.2007], [-1.2007, -1.2007, -1.2007]],
                      [[0.2481, 0.2481, 0.2481], [0.2481, 0.2481, 0.2481], [0.2481, 0.2481, 0.2481]]]
                     ])
mask = torch.tensor([[[3, 0, 1], [3, 3, 3], [3, 3, 3]],
                     [[3, 0, 1], [3, 3, 3], [3, 3, 3]]])

fea1 = fea1.view(2, 4, -1)
fea1 = fea1.transpose(1, 2).contiguous()
fea1 = fea1.view(-1, 4)
mask = mask.view(2, -1).contiguous()
mask = mask.view(-1)
print(Cross_loss(fea1, mask))

a = torch.rand((3, 4))
b = torch.rand((4, 4))
c = [a, b]
c = torch.cat(c, 0)
print(c)




































