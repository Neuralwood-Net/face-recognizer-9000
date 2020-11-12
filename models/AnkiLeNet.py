import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class AnkiLeNetV1(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.conv3 = nn.Conv2d(64, 128, 5, padding=2)
        
        # Input neurons defined by
        # out_channels * (input_img_size / (pool_size**num_pooling)**2)
        self.fc1 = nn.Linear(int(128*(64 / 2**3)**2), 512)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        
        x = x.view(-1, int(128*(64 / 2**3)**2))
        
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        
        return x;