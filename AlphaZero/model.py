import torch
import torch.nn as nn
import torch.nn.functional as F

class AlphaZeroNet(nn.Module):
    def __init__(self, board_size=8, num_input_channels=7, num_actions=12):
        super(AlphaZeroNet, self).__init__()
        
        # --- Feature Extractor (The "Eyes") ---
        # A small ResNet-style block to understand the board
        self.conv1 = nn.Conv2d(num_input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # --- Policy Head (The "Actor") ---
        # Outputs probability for every possible move (4 dirs * 3 types = 12)
        self.policy_conv = nn.Conv2d(128, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        # 32 channels * 8 * 8 = 2048 -> Flattened
        self.policy_fc = nn.Linear(32 * board_size * board_size, num_actions)

        # --- Value Head (The "Critic") ---
        # Outputs a single number [-1, 1] (Lose/Win)
        self.value_conv = nn.Conv2d(128, 16, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(16)
        self.value_fc1 = nn.Linear(16 * board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # x shape: [Batch, Channels, 8, 8]
        
        # Backbone
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Policy Head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1) # Flatten
        p = self.policy_fc(p)
        # We use LogSoftmax for numerical stability during training, 
        # but Softmax for inference. Let's return raw logits here.
        
        # Value Head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1) # Flatten
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v)) # Output between -1 and 1
        
        return p, v