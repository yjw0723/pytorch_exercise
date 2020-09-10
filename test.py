import torch
import numpy as np
from nets import *
from train_validation import WBCEloss
import torch.optim as optim

criterion = WBCEloss

g_model = GlobalNet(10).cuda()
g_optim = optim.Adam(g_model.parameters(), lr=0.005)

l_model = LocalNet(10).cuda()
l_optim = optim.Adam(l_model.parameters(), lr=0.005)

f_model = FusionNet(10).cuda()
f_optim = optim.Adam(f_model.parameters(), lr=0.005)

weight_p = torch.from_numpy(np.random.rand(128, 10)).float().cuda()
weight_n = torch.from_numpy(np.random.rand(128, 10)).float().cuda()

for i in range(10):
    labels = torch.from_numpy(np.random.randint(low=0, high=1, size=(128, 10))).float().cuda()
    weight_p = torch.from_numpy(np.random.rand(128, 10)).float().cuda()
    weight_n = torch.from_numpy(np.random.rand(128, 10)).float().cuda()

    g_inputs = torch.from_numpy(np.random.rand(128,3,224,224)).float().cuda()

    g_optim.zero_grad()
    outputs, poolings, heatmaps = g_model.gOutput(g_inputs)

    loss = criterion(labels, outputs, weight_p, weight_n)
    loss.backward()
    g_optim.step()

    inputs = torch.from_numpy(np.random.rand(128,2048)).float().cuda()
    test = poolings.detach().cpu().numpy().tolist()
    test = torch.from_numpy(np.array(test)).float().cuda()
    concated = torch.cat((test,test),dim=1)

    f_optim.zero_grad()

    output = f_model.forward(concated)
    loss = criterion(labels, output, weight_p, weight_n)
    loss.backward()
    f_optim.step()
    print(i, loss)