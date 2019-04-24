# Author : Wang Yongjie
# Email  : yongjie.wang@ntu.edu.sg
# Description: Network of DQN

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self, state_space, action_space):
        super(Net, self).__init__()
        self.state_space = state_space
        self.action_space = action_space

        #define layers
        self.conv1 = nn.Sequential(
                nn.Conv2d(
                    in_channels = 2,
                    out_channels = 32,
                    kernel_size = 8,
                    stride = 4,
                    padding = 0,
                    ),
                nn.ReLU(),
                )
        self.conv2 = nn.Sequential(
                nn.Conv2d(
                    in_channels = 32,
                    out_channels = 64,
                    kernel_size = 4,
                    stride = 2,
                    padding = 0,
                    ),
                nn.ReLU()
                )
        self.conv3 = nn.Sequential(
                nn.Conv2d(
                    in_channels = 64,
                    out_channels = 64,
                    kernel_size = 3,
                    stride = 1,
                    padding = 0,
                    ),
                nn.ReLU()
                )

        self.fc1 = nn.Linear(7 * 7 * 64 , 512)
        self.fc2 = nn.Linear(512, self.action_space)


    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        flatten = conv3.view(conv3.size(0), -1)
        fc1 = self.fc1(flatten)
        relu1 = F.relu(fc1)
        fc2 = self.fc2(relu1)

        return fc2

'''
if __name__ == "__main__":
    net = Net((84,84, 2), 18)
    test = torch.rand(1, 2, 84, 84)
    print(net(test))

'''
