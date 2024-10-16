import copy
from collections import deque
import numpy as np
import torch
import random

class GameState:
    def __init__(self, history_length=3):
        self.winner = None
        self.board = np.zeros((5, 5), dtype=int)
        self.red_list = [1, 2, 3, 4, 5, 6]
        self.blue_list = [-1, -2, -3, -4, -5, -6]
        self.move_count = 0
        self.current_player = 1  # 当前玩家: 1代表红方, -1代表蓝方
        self.history_length = history_length
        self.history = deque(maxlen=self.history_length)  # 保存最近几轮的棋盘状态
        self.current_dice = None  # 新增：当前骰子结果
        self.initialize_pieces()  # 初始化棋子位置
        self.roll_dice()  # 第一次掷骰子
        self.update_history()  # 保存初始棋盘状态

    def initialize_pieces(self):
        # 固定初始位置，放置红方和蓝方棋子在棋盘的角落
        self.board = np.zeros((5, 5), dtype=int)
        # 红方棋子放在左上角
        self.board[0][0:3] = self.red_list[:3]
        self.board[1][0:2] = self.red_list[3:5]
        self.board[2][0] = self.red_list[5]
        # 蓝方棋子放在右下角
        self.board[4][4:1:-1] = self.blue_list[:3]
        self.board[3][4:2:-1] = self.blue_list[3:5]
        self.board[2][4] = self.blue_list[5]

    def roll_dice(self):
        self.current_dice = random.randint(1, 6)

    def get_legal_actions(self):
        """
        获取当前玩家的所有合法移动，基于当前骰子结果
        :return: 列表 of ((start_i, start_j), (end_i, end_j))
        """
        dice_result = self.current_dice
        # 获取当前玩家的可用棋子
        if self.current_player == 1:
            player_pieces = [piece for piece in self.red_list if piece in self.board.flatten()]
            directions = [(1, 0), (0, 1), (1, 1)]
        else:
            player_pieces = [abs(piece) for piece in self.blue_list if -piece in self.board.flatten()]
            directions = [(-1, 0), (0, -1), (-1, -1)]

        # 找到与骰子结果最接近的棋子
        min_diff = float('inf')
        closest_pieces = []
        for piece in player_pieces:
            diff = abs(piece - dice_result)
            if diff < min_diff:
                min_diff = diff
                closest_pieces = [piece]
            elif diff == min_diff:
                closest_pieces.append(piece)

        if not closest_pieces:
            return []  # 没有可移动的棋子

        legal_actions = []
        for piece_num in closest_pieces:
            piece_value = piece_num if self.current_player == 1 else -piece_num
            # 找到所有符合 piece_value 的棋子位置
            positions = np.argwhere(self.board == piece_value)
            for pos in positions:
                i, j = pos
                for direction in directions:
                    ni, nj = i + direction[0], j + direction[1]
                    if 0 <= ni < 5 and 0 <= nj < 5:
                        legal_actions.append(((i, j), (ni, nj)))
        return legal_actions

    def apply_move(self, move):
        """
        执行一次移动并切换玩家
        :param move: ((start_i, start_j), (end_i, end_j))
        """
        (start, end) = move
        moving_piece = self.board[start[0]][start[1]]
        target_piece = self.board[end[0]][end[1]]

        # 如果目标位置有棋子，移出棋盘
        if target_piece != 0:
            if target_piece > 0:
                if target_piece in self.red_list:
                    self.red_list.remove(target_piece)
            else:
                if target_piece in self.blue_list:
                    self.blue_list.remove(target_piece)

        # 移动棋子
        self.board[end[0]][end[1]] = moving_piece
        self.board[start[0]][start[1]] = 0
        self.move_count += 1
        self.current_player *= -1  # 切换玩家
        self.roll_dice()  # 新增：切换玩家后掷骰子
        self.update_history()  # 更新历史记录

    def is_terminal(self):
        """
        判断游戏是否结束
        :return: True if game is over, else False
        """
        if self.get_winner() is not None:
            return True
        return False

    def get_winner(self):
        """
        获取胜者
        :return: 1代表红方获胜，-1代表蓝方获胜，None代表游戏未结束
        """
        if self.winner is not None:
            return self.winner
        # 优先检查是否占据对方起始点
        if self.board[0][0] < 0:  # 蓝方占据红方起始点
            self.winner = -1
        elif self.board[4][4] > 0:  # 红方占据蓝方起始点
            self.winner = 1
        else:
            # 检查是否所有棋子被吃掉
            red_has_pieces = np.any(self.board > 0)
            blue_has_pieces = np.any(self.board < 0)
            if not red_has_pieces:
                self.winner = -1
            elif not blue_has_pieces:
                self.winner = 1
        return self.winner

    def update_history(self):
        """
        更新历史记录，保存当前棋盘状态
        """
        self.history.append(self.board.copy())

    def to_tensor(self):
        """
        将棋盘状态转换为神经网络输入张量
        :return: torch.Tensor of shape (10, 5, 5)
        """
        red_pieces = (self.board > 0).astype(np.float32)
        blue_pieces = (self.board < 0).astype(np.float32)
        piece_numbers = np.abs(self.board).astype(np.float32) / 6.0  # 归一化
        history_tensor = np.zeros_like(self.board, dtype=np.float32)
        if len(self.history) > 1:
            history_tensor = np.abs(self.history[-2]).astype(np.float32) / 6.0

        # 骰子结果的独热编码
        dice_one_hot = np.zeros((6, 5, 5), dtype=np.float32)
        dice_index = self.current_dice - 1
        dice_one_hot[dice_index, :, :] = 1.0

        # 堆叠成10个通道的张量
        state_tensor = np.stack([red_pieces, blue_pieces, piece_numbers, history_tensor] + list(dice_one_hot), axis=0)
        return torch.tensor(state_tensor, dtype=torch.float32)

    def copy(self):
        """
        深度复制游戏状态
        :return: 新的 GameState 实例
        """
        game_state = GameState(history_length=self.history_length)
        game_state.board = self.board.copy()
        game_state.winner = self.winner
        game_state.current_player = self.current_player
        game_state.move_count = self.move_count
        game_state.history = copy.deepcopy(self.history)
        game_state.red_list = self.red_list.copy()
        game_state.blue_list = self.blue_list.copy()
        game_state.current_dice = self.current_dice  # 复制骰子结果
        return game_state
