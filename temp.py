# train.py

import torch
import numpy as np
from model import AlphaGoModel, ResidualBlock
from game_state import GameState
from mcts import MCTS
import torch.nn.functional as F
import torch.multiprocessing as mp
import threading
import math
import random
import copy

# 自我博弈类（已修正）
class SelfPlay:
    def __init__(self, model, mcts, num_simulations=800, num_games=100):
        """
        :param model: 策略-价值模型
        :param mcts: MCTS实例
        :param num_simulations: 每次MCTS的模拟次数
        :param num_games: 自我博弈的游戏数量
        """
        self.model = model
        self.mcts = mcts  # 初始化MCTS
        self.num_games = num_games  # 自我博弈的游戏数量
        self.training_data = []  # 保存训练数据
        self.num_pieces = 6  # 每方棋子数量
        self.num_directions = 3  # 每个棋子的移动方向

        # 定义方向映射
        self.red_direction_map = {(1, 0): 0, (0, 1): 1, (1, 1): 2}
        self.blue_direction_map = {(-1, 0): 0, (0, -1): 1, (-1, -1): 2}

    def encode_action(self, action, current_player, game_state):
        """
        将动作编码为唯一的索引 (0-17)
        :param action: ((start_i, start_j), (end_i, end_j))
        :param current_player: 当前玩家，1代表红方，-1代表蓝方
        :param game_state: 当前游戏状态
        :return: action_index (0-17)
        """
        start, end = action
        di = end[0] - start[0]
        dj = end[1] - start[1]
        direction = (di, dj)

        # 获取方向索引
        if current_player == 1:
            direction_map = self.red_direction_map
            piece_num = game_state.board[start[0]][start[1]]
            try:
                piece_index = game_state.red_list.index(piece_num)
            except ValueError:
                raise ValueError(f"Red piece {piece_num} not found in red_list.")
        else:
            direction_map = self.blue_direction_map
            piece_num = game_state.board[start[0]][start[1]]
            try:
                piece_index = game_state.blue_list.index(piece_num)
            except ValueError:
                raise ValueError(f"Blue piece {piece_num} not found in blue_list.")

        direction_index = direction_map.get(direction, -1)
        if direction_index == -1:
            raise ValueError(f"Invalid direction {direction} for player {current_player}.")

        return piece_index * self.num_directions + direction_index  # 0-17

    def decode_action(self, action_index, current_player, game_state):
        """
        将动作索引解码为具体动作
        :param action_index: 0-17
        :param current_player: 当前玩家，1代表红方，-1代表蓝方
        :param game_state: 当前游戏状态
        :return: ((start_i, start_j), (end_i, end_j))
        """
        piece_index = action_index // self.num_directions
        direction_index = action_index % self.num_directions

        if current_player == 1:
            direction_map = {0: (1, 0), 1: (0, 1), 2: (1, 1)}
            if piece_index >= len(game_state.red_list):
                raise ValueError(f"Invalid piece index {piece_index} for red player.")
            piece_num = game_state.red_list[piece_index]
        else:
            direction_map = {0: (-1, 0), 1: (0, -1), 2: (-1, -1)}
            if piece_index >= len(game_state.blue_list):
                raise ValueError(f"Invalid piece index {piece_index} for blue player.")
            piece_num = game_state.blue_list[piece_index]

        direction = direction_map.get(direction_index, (0, 0))
        # Find the piece's current position
        positions = np.argwhere(game_state.board == piece_num)
        if len(positions) == 0:
            raise ValueError(f"Piece {piece_num} not found on the board.")
        start_i, start_j = positions[0]  # 假设第一个找到的棋子
        end_i = start_i + direction[0]
        end_j = start_j + direction[1]

        # 检查目标位置是否在棋盘范围内
        if not (0 <= end_i < 5 and 0 <= end_j < 5):
            raise ValueError(f"Decoded action leads to out-of-bounds position: ({end_i}, {end_j}).")

        return ((start_i, start_j), (end_i, end_j))

    def self_play_game(self):
        """
        执行一次完整的自我博弈
        :return: 返回该局游戏的状态、动作和最终结果
        """
        state = GameState()  # 初始化游戏状态
        game_data = []  # 记录该局游戏的状态、动作及奖励

        while not state.is_terminal():  # 当游戏没有结束时
            current_player = state.current_player
            # 使用MCTS搜索并选择动作
            mcts_root = self.mcts.run(state)  # 通过MCTS搜索
            if mcts_root is None:
                print("MCTS root is None.")
                break

            print("-----------开始遍历mcts_root---------------")
            print(f"MCTS root is leaf: {mcts_root.is_leaf()}")

            for action, child_node in mcts_root.children.items():
                print(f"动作: {action}")  # 动作
                print(f"访问次数: {child_node.visits}")  # 子节点的访问次数
                print(f"累计价值: {child_node.value_sum}")  # 子节点的累计价值
                print(f"平均价值: {child_node.value()}")  # 子节点的平均价值

            action_probs = self.get_mcts_action_probs(mcts_root, current_player, state)  # 获取MCTS给出的动作概率分布

            print(f"-------获取MCTS给出的动作概率分布---------action_probs:{action_probs}")

            # 将当前状态、MCTS动作概率保存，用于训练
            state_tensor = state.to_tensor()  # 不需要移动到设备，待训练时统一处理
            print(f"State tensor shape: {state_tensor.shape}")  # 添加打印语句
            game_data.append((state_tensor, action_probs))

            # 根据MCTS的搜索结果随机选择一个动作（带有探索性）
            action_index = self.select_action(action_probs)
            try:
                action = self.decode_action(action_index, current_player, state)
            except ValueError as e:
                print(f"Decoding action failed: {e}")
                # 选择随机合法动作
                legal_actions = state.get_legal_actions()
                if legal_actions:
                    action = random.choice(legal_actions)
                else:
                    break  # 无合法动作，结束游戏

            state.apply_move(action)  # 应用该动作

        # 获取胜者
        winner = state.get_winner()
        reward = 1 if winner == 1 else -1  # 红方胜利为1，蓝方胜利为-1

        # 为每一步分配最终奖励
        game_data = [(state_tensor, action_prob, reward) for (state_tensor, action_prob) in game_data]
        return game_data  # 返回游戏数据

    def get_mcts_action_probs(self, root, current_player, game_state):
        """
        获取MCTS的动作概率分布
        :param root: MCTS搜索的根节点
        :param current_player: 当前玩家，1代表红方，-1代表蓝方
        :param game_state: 当前游戏状态
        :return: 动作的概率分布，长度为18
        """
        print("--------------get_mcts_action_probs----------------")
        total_visits = sum(child.visits for child in root.children.values())
        print(f"total_visits:{total_visits}")
        action_probs = np.zeros(18)  # 初始化18个动作的概率值

        for action, child in root.children.items():
            try:
                action_index = self.encode_action(action, current_player, game_state)
                action_probs[action_index] += child.visits / total_visits
            except ValueError as e:
                print(f"Action encoding failed: {e}")
                continue  # 无效动作，跳过

        print(f"total_visits:{total_visits}\naction_probs:{action_probs}")
        return action_probs

    def select_action(self, action_probs, temperature=1.0):
        """
        根据MCTS的动作概率选择一个动作
        :param action_probs: 动作的概率分布，长度为18
        :param temperature: 控制探索的温度参数，越高探索性越强
        :return: 选择的动作索引
        """
        # 如果action_probs全为零，说明当前无合法动作或者MCTS未找到合适动作
        if np.sum(action_probs) == 0:
            print("Warning: action_probs sum is zero, choosing random action")
            action_probs = np.ones_like(action_probs) / len(action_probs)  # 如果没有合法动作，则均匀选择

        # 进行softmax计算，使用温度参数调节探索性
        action_probs = np.power(action_probs, 1 / temperature)
        action_probs /= np.sum(action_probs)  # 归一化，确保概率和为1

        # 选择动作索引
        action_index = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        return action_index

    def self_play(self):
        """
        执行自我博弈，生成训练数据
        """
        for game_num in range(self.num_games):
            print(f"开始第 {game_num + 1}/{self.num_games} 局游戏的自我博弈...")
            game_data = self.self_play_game()  # 执行一局游戏
            self.training_data.extend(game_data)  # 将游戏数据添加到训练集中

    def get_training_data(self):
        """
        获取自我博弈生成的训练数据
        :return: 返回训练数据
        """
        return self.training_data

# 训练数据的打包处理（已修正）
def prepare_training_data(training_data):
    """
    将游戏数据整理成神经网络训练所需的形式
    :param training_data: 自我博弈产生的原始数据
    :return: 整理后的训练输入和标签
    """
    states, policies, rewards = zip(*training_data)  # 解压游戏数据
    states = torch.stack(states, dim=0)  # 拼接所有的状态，形状 [N, 4, 5, 5]
    policies = torch.tensor(np.array(policies), dtype=torch.float32)  # 形状 [N, 18]
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)  # 形状 [N, 1]
    return states, policies, rewards

# 用于训练神经网络的训练循环（已修正）
def train_model(model, optimizer, training_data, batch_size=32, epochs=10):
    model.train()  # 切换为训练模式
    states, policies, rewards = prepare_training_data(training_data)  # 准备训练数据

    dataset = torch.utils.data.TensorDataset(states, policies, rewards)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0  # 累计loss
        for batch_states, batch_policies, batch_rewards in data_loader:
            print(f"Batch states shape: {batch_states.shape}")  # 添加打印语句
            batch_states = batch_states.to(next(model.parameters()).device)
            batch_policies = batch_policies.to(next(model.parameters()).device)
            batch_rewards = batch_rewards.to(next(model.parameters()).device)

            optimizer.zero_grad()  # 清除上一次的梯度

            # 前向传播，计算策略和价值
            policy_pred, value_pred = model(batch_states)

            # 策略损失: 使用交叉熵损失
            # 这里假设政策标签是概率分布，因此使用负对数似然
            policy_loss = -torch.sum(batch_policies * torch.log(policy_pred + 1e-8)) / batch_size

            # 价值损失: 使用均方误差损失
            value_loss = F.mse_loss(value_pred, batch_rewards)

            # 总损失
            loss = policy_loss + value_loss

            # 反向传播并更新模型参数
            loss.backward()
            optimizer.step()

            total_loss += loss.item()  # 记录损失值

        # 打印每个epoch的平均损失
        print(f"Epoch {epoch + 1}/{epochs} - Avg Loss: {total_loss / len(data_loader)}")
    torch.save(model.state_dict(), 'trained_model.pth')
    print("模型已保存为 'trained_model.pth'")

# 主程序
if __name__ == '__main__':
    mp.set_start_method('spawn')  # 推荐在Windows下使用'spawn'启动方法

    model = AlphaGoModel(ResidualBlock)  # 假设你已经定义了AlphaGoModel和ResidualBlock
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()  # 设置为评估模式

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 初始化MCTS
    mcts = MCTS(model, c_puct=1.0, num_simulations=800, num_threads=4)

    # 执行自我博弈，生成训练数据
    self_play_agent = SelfPlay(model, mcts, num_simulations=800, num_games=100)
    self_play_agent.self_play()

    # 获取自我博弈生成的训练数据
    training_data = self_play_agent.get_training_data()

    # 使用生成的数据训练模型
    train_model(model, optimizer, training_data, batch_size=32, epochs=10)
