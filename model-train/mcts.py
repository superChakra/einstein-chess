# mcts.py

import torch
import numpy as np
import torch.nn.functional as F
import threading
import math
import random

# ------------------------ MCTS 类 ------------------------

class MCTSNode:
    def __init__(self, parent=None, prior=1.0):
        self.parent = parent
        self.children = {}  # 动作为键，子节点为值
        self.visits = 0
        self.value_sum = 0
        self.prior = prior

    def is_leaf(self):
        return len(self.children) == 0

    def expand(self, action_priors):
        """
        扩展当前节点的子节点
        :param action_priors: 动作及其先验概率列表
        """
        for action, prior in action_priors:
            if action not in self.children:
                self.children[action] = MCTSNode(parent=self, prior=prior)

    def update(self, value):
        """
        更新节点的访问次数和累计价值
        :param value: 模拟结果的价值
        """
        self.visits += 1
        self.value_sum += value

    def value(self):
        """
        计算节点的平均价值
        :return: 平均价值
        """
        return self.value_sum / self.visits if self.visits > 0 else 0

    def ucb_score(self, parent_visits, c_puct=1.0):
        """
        计算节点的UCB评分
        :param parent_visits: 父节点的访问次数
        :param c_puct: 探索系数
        :return: UCB评分
        """
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visits)
        return self.value() + exploration


class MCTS:
    def __init__(self, model, encode_action, c_puct=1.0, num_simulations=1600, num_threads=4):
        """
        初始化MCTS
        :param model: 策略-价值模型
        :param encode_action: 动作编码函数
        :param c_puct: 探索系数
        :param num_simulations: 总模拟次数
        :param num_threads: 并行线程数
        """
        self.model = model
        self.encode_action = encode_action  # 存储传入的动作编码函数
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.num_threads = num_threads
        self.lock = threading.Lock()
        self.root = MCTSNode()

    def run(self, state):
        """
        运行MCTS，并返回根节点
        :param state: 当前的游戏状态
        :return: 根节点
        """
        self.root = MCTSNode()
        threads = []
        for _ in range(self.num_threads):
            thread = threading.Thread(target=self.simulate, args=(state.copy(),))
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
        return self.root

    def simulate(self, state):
        """
        单次MCTS模拟，涉及选择、扩展、模拟和反向传播四个阶段
        :param state: 当前游戏状态的副本
        """
        for _ in range(self.num_simulations // self.num_threads):
            node = self.root
            state_copy = state.copy()

            # 选择阶段
            while not node.is_leaf():
                max_ucb = -float('inf')
                best_action = None
                for action, child in node.children.items():
                    score = child.ucb_score(node.visits, self.c_puct)
                    if score > max_ucb:
                        max_ucb = score
                        best_action = action
                if best_action is None:
                    break
                state_copy.apply_move(best_action)
                node = node.children[best_action]

            # 扩展阶段
            legal_actions = state_copy.get_legal_actions()
            if not legal_actions:
                winner = state_copy.get_winner()
                value = 1 if winner == 1 else -1
                with self.lock:
                    node.update(value)
                continue

            # 神经网络预测
            state_tensor = state_copy.to_tensor().to(next(self.model.parameters()).device)
            # print(f"Before unsqueeze: {state_tensor.shape}")  # 应输出 torch.Size([4, 5, 5])
            state_tensor = state_tensor.unsqueeze(0)  # 添加批次维度，形状: [1, 4, 5, 5]
            # print(f"After unsqueeze: {state_tensor.shape}")  # 应输出 torch.Size([1, 4, 5, 5])
            with torch.no_grad():
                policy, value = self.model(state_tensor)
            # print(f"Policy shape: {policy.shape}")  # 应输出 torch.Size([1, 18])
            # print(f"Value shape: {value.shape}")    # 应输出 torch.Size([1, 1])
            policy_np = policy.cpu().numpy().squeeze()  # 移除批次维度，形状: [18]
            value = value.cpu().numpy().squeeze()  # 移除批次和通道维度，形状: []

            # print(f"Policy numpy shape: {policy_np.shape}")  # 应输出 (18,)
            # print(f"Value: {value}")  # 应输出标量

            # 映射策略概率到合法动作
            action_probs = self.get_action_probs(policy_np, state_copy.current_player, state_copy)
            legal_actions = list(action_probs.keys())
            policy_probs = np.array([action_probs[action] for action in legal_actions])

            if len(policy_probs) == 0 or np.sum(policy_probs) == 0:
                with self.lock:
                    node.update(value)
                continue

            # softmax 计算
            policy_probs = F.softmax(torch.tensor(policy_probs), dim=0).cpu().numpy()
            # print(f"Policy probabilities after softmax: {policy_probs}")

            # 选择前20%的动作进行扩展
            top_k = max(1, int(len(policy_probs) * 0.2))
            sorted_indices = np.argsort(policy_probs)[::-1][:top_k]
            top_actions = [legal_actions[i] for i in sorted_indices]
            top_probs = policy_probs[sorted_indices]

            # print(f"Top actions: {top_actions}")
            # print(f"Top probabilities: {top_probs}")

            # 扩展子节点
            with self.lock:
                node.expand(zip(top_actions, top_probs))

            # 模拟阶段
            for action, prob in zip(top_actions[:3], top_probs[:3]):
                state_sim = state_copy.copy()
                state_sim.apply_move(action)
                rollout_value = self.rollout(state_sim)
                combined_value = (rollout_value + value) / 2  # 根据需要调整权重
                with self.lock:
                    node.children[action].update(combined_value)

    def rollout(self, state):
        """
        执行rollout（即模拟游戏直到终局），以获取模拟的局面价值
        :param state: 当前游戏状态
        :return: 模拟结束后的局面价值，1代表红方胜利，-1代表蓝方胜利
        """
        while not state.is_terminal():
            legal_actions = state.get_legal_actions()
            if len(legal_actions) == 0:
                break
            action = random.choice(legal_actions)
            state.apply_move(action)
        winner = state.get_winner()
        return 1 if winner == 1 else -1

    def get_action_probs(self, policy_np, current_player, game_state):
        """
        根据当前玩家和游戏状态，动态映射神经网络输出的策略概率到具体的动作
        :param policy_np: 神经网络输出的策略概率数组，形状: [18]
        :param current_player: 当前玩家，1代表红方，-1代表蓝方
        :param game_state: 当前游戏状态
        :return: 动作概率的字典，键为动作，值为对应的概率
        """
        legal_actions = game_state.get_legal_actions()
        action_probs = {}
        for action in legal_actions:
            try:
                # 使用传递的 encode_action 方法获取动作的索引
                action_index = self.encode_action(action, current_player, game_state)
                if 0 <= action_index < len(policy_np):
                    action_probs[action] = policy_np[action_index]
            except ValueError as e:
                print(f"Action encoding failed: {e}")
                continue  # 无效动作，跳过
        return action_probs
