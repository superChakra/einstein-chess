import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channel=256, num_channel=256):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=num_channel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=num_channel, out_channels=num_channel, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channel)

    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.relu(y + x)

class AlphaGoModel(nn.Module):
    def __init__(self, residual_block, num_blocks=7):
        super(AlphaGoModel, self).__init__()

        self.relu = nn.ReLU()

        # 初始的卷积块，输入通道数改为10
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            self.relu
        )

        # 残差块的堆叠
        self.residual_blocks = nn.Sequential(
            *[residual_block() for _ in range(num_blocks)]
        )

        # 策略头
        self.policy_conv = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            self.relu,
            nn.Dropout(p=0.3)
        )

        self.flatten = nn.Flatten()
        self.linear_p_1 = nn.Linear(16*5*5, 18)

        # 价值头
        self.value_conv = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=8, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(8),
            self.relu,
            nn.Dropout(p=0.3)
        )

        self.linear_v_1 = nn.Linear(8*5*5, 64)
        self.linear_v_2 = nn.Linear(64, 1)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.block1(x)
        x = self.residual_blocks(x)

        # 策略头
        result_policy = self.policy_conv(x)
        result_policy = self.flatten(result_policy)
        result_policy = self.linear_p_1(result_policy)
        result_policy = F.softmax(result_policy, dim=1)

        # 价值头
        result_value = self.value_conv(x)
        result_value = self.flatten(result_value)
        result_value = self.relu(self.linear_v_1(result_value))
        result_value = self.linear_v_2(result_value)
        result_value = torch.tanh(result_value)

        return result_policy, result_value
