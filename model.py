# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channel=256, num_channel=256):
        super(ResidualBlock, self).__init__()
        # 第一个卷积层，保持通道数
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=num_channel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channel)
        self.relu = nn.ReLU()
        # 第二个卷积层，保持通道数
        self.conv2 = nn.Conv2d(in_channels=num_channel, out_channels=num_channel, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channel)

    def forward(self, x):
        # 通过第一个卷积层
        y = self.relu(self.bn1(self.conv1(x)))
        # 通过第二个卷积层并加上输入实现跳跃连接
        y = self.bn2(self.conv2(y))
        return self.relu(y + x)  # 残差连接

class AlphaGoModel(nn.Module):
    def __init__(self, residual_block, num_blocks=7):
        super(AlphaGoModel, self).__init__()

        # 激活函数
        self.relu = nn.ReLU()  # ReLU函数会在各个地方被重用

        # 初始的卷积块，用于提取棋盘状态的基础特征
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=256, kernel_size=3, stride=1, padding=1),  # 输入通道数4，输出256
            nn.BatchNorm2d(256),
            self.relu
        )

        # 残差块的堆叠，通过循环创建多个残差块
        self.residual_blocks = nn.Sequential(
            *[residual_block() for _ in range(num_blocks)]
        )

        # 策略头 (Policy Head)，预测下一步动作的概率分布
        self.policy_conv = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=16, kernel_size=5, stride=1, padding=2),  # 修改kernel_size和padding
            nn.BatchNorm2d(16),
            self.relu,
            nn.Dropout(p=0.3)  # 添加 Dropout
        )

        # 价值头 (Value Head)，评估当前局面的优劣
        self.value_conv = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=8, kernel_size=5, stride=1, padding=2),  # 修改kernel_size和padding
            nn.BatchNorm2d(8),
            self.relu,
            nn.Dropout(p=0.3)  # 添加 Dropout
        )
        # 扁平化，用于将特征从2D卷积转换为1D
        self.flatten = nn.Flatten()

        # 全连接层用于策略网络的输出
        self.linear_p_1 = nn.Linear(16*5*5, 18)  # 16输入，输出为18，表示5x5棋盘上的18种可能的移动

        # 全连接层用于价值网络的输出
        self.linear_v_1 = nn.Linear(8*5*5, 64)  # 从8通道输出64维
        self.linear_v_2 = nn.Linear(64, 1)  # 最后输出一个标量，表示局面价值

        # 初始化权重
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 基础卷积块，提取棋盘的初步特征
        x = self.block1(x)

        # 通过多个残差块提取更深层次的特征
        x = self.residual_blocks(x)

        # 策略头 (Policy Head) 流程
        result_policy = self.policy_conv(x)  # 卷积提取策略特征
        result_policy = self.flatten(result_policy)  # 扁平化
        result_policy = self.linear_p_1(result_policy)  # 全连接层
        result_policy = F.softmax(result_policy, dim=1)  # softmax 计算行动概率分布

        # 价值头 (Value Head) 流程
        result_value = self.value_conv(x)  # 卷积提取价值特征
        result_value = self.flatten(result_value)  # 扁平化
        result_value = self.relu(self.linear_v_1(result_value))  # 全连接层并使用ReLU
        result_value = self.linear_v_2(result_value)  # 输出一个标量表示局面价值
        result_value = torch.tanh(result_value)  # 用tanh将输出范围限制在 [-1, 1]

        return result_policy, result_value
