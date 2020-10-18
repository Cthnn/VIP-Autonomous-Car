import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        #OUR VERSION OF NVIDIA'S SELF DRIVING NEURAL NETWORK
        self.conv1 = nn.Conv2d(3,24,5,2)
        self.conv2 = nn.Conv2d(24,36,5,2)
        self.conv3 = nn.Conv2d(36,48,5,2)
        self.conv4 = nn.Conv2d(48,64,3)
        self.conv5 = nn.Conv2d(64,64,3)
        self.dropout = nn.Dropout(0.25)
        self.dense1 = nn.Linear(64*2*33,100)
        self.dense2 = nn.Linear(100,50)
        self.dense3 = nn.Linear(50,10)
        self.dense4 = nn.Linear(10,1)
        

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.dropout(self.conv5(x))
        x = x.view(-1,64*2*33)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.dense3(x)
        x = self.dense4(x)
        return x