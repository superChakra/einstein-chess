import sys
import os
import numpy as np
import torch
import random  # 新增：用于掷骰子
from PySide6 import QtCore
from PySide6.QtCore import QTimer, Qt, QSize, QThread, Signal
from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QGridLayout, QWidget, QLabel,
    QVBoxLayout, QHBoxLayout, QMessageBox, QSizePolicy, QComboBox, QLineEdit
)

from game_state import GameState
from mcts import MCTS
from model import AlphaGoModel, ResidualBlock


class AIMoveThread(QThread):
    """
    QThread 子类，用于在后台线程中执行 AI 移动（MCTS 搜索）。
    """
    move_completed = Signal(object)  # 发送 AI 选择的动作

    def __init__(self, mcts, game_state):
        super().__init__()
        self.mcts = mcts
        self.game_state = game_state

    def run(self):
        """
        执行 MCTS 搜索，选择最佳动作，并通过信号发送回主线程。
        """
        print("AI 开始进行 MCTS 搜索...")
        mcts_root = self.mcts.run(self.game_state)
        if not mcts_root.children:
            print("AI 找不到可行的动作。")
            self.move_completed.emit(None)
            return

        # 根据访问次数选择最优动作
        visit_counts = {action: child.visits for action, child in mcts_root.children.items()}
        best_action = max(visit_counts, key=visit_counts.get)
        print(f"AI 选择的最佳动作: {best_action}")
        self.move_completed.emit(best_action)


class EinsteinChess(QMainWindow):
    """
    主游戏窗口类，继承自 QMainWindow。
    负责初始化 UI、处理用户交互、管理游戏状态以及集成 AI 移动。
    """

    def __init__(self):
        super().__init__()

        # 游戏相关属性初始化
        self.num_directions = 3
        self.board_size = 5
        self.sign_op_step = True
        self.current_col = None
        self.old_button = None
        self.current_row = None
        self.ai_role = "红方"  # 默认 AI 为红方
        self.blue_time = 240
        self.move_history = []
        self.red_time = 240
        self.current_turn = "红方"
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.ai_numbers = list(range(1, 7))
        self.person_numbers = list(range(1, 7))
        self.undo_button = None
        self.ai_move_button = None
        self.gen_board_button = None
        self.switch_user_button = None
        self.tool_layout = None
        self.blue_label = None
        self.red_label = None
        self.user_label = None
        self.right_layout = None
        self.blue_board_numbers_input = None
        self.red_board_numbers_input = None
        self.blue_time_input = None
        self.red_time_input = None
        self.data_input_layout = None
        self.left_grid_layout = None
        self.left_bottom_layout = None
        self.left_layout = None

        # 新增：骰子结果
        self.current_dice = None

        # 初始化定时器
        self.red_timer = QTimer()
        self.red_timer.timeout.connect(self.update_red_timer)
        self.blue_timer = QTimer()
        self.blue_timer.timeout.connect(self.update_blue_timer)

        # 初始化模型和 MCTS
        self.model = AlphaGoModel(ResidualBlock)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trained_model.pth')
            print(f"加载模型路径: {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            print("模型加载成功。")
        except Exception as e:
            QMessageBox.critical(self, "加载模型失败", f"无法加载模型: {e}")
            sys.exit(1)
        self.model.to(device)
        self.model.eval()  # 设置为评估模式

        # 初始化 MCTS，encode_action 方法将在后面定义
        self.mcts = MCTS(
            model=self.model,
            encode_action=self.encode_action,
            c_puct=1.0,
            num_simulations=1600,
            num_threads=4
        )

        # 初始化 UI
        self.init_ui()
        self.reset_game()

    def init_ui(self):
        """
        初始化用户界面布局和组件。
        """
        # 主布局
        main_layout = QHBoxLayout()
        main_layout.setSpacing(20)  # 增加 左右布局之间的距离

        # 左侧布局：棋盘和相关输入
        self.left_layout = QVBoxLayout()
        self.left_layout.setSpacing(5)
        self.left_grid_layout = QGridLayout()
        self.left_grid_layout.setSpacing(10)  # 增加网格按钮之间的间距

        self.left_bottom_layout = QHBoxLayout()

        # 设置参数输入布局
        self.data_input_layout = QVBoxLayout()

        self.red_time_input = QLineEdit()
        self.red_time_input.setFixedSize(280, 40)
        self.red_time_input.setPlaceholderText("请输入红方时间（秒）")
        self.data_input_layout.addWidget(self.red_time_input)

        self.blue_time_input = QLineEdit()
        self.blue_time_input.setFixedSize(280, 40)
        self.blue_time_input.setPlaceholderText("请输入蓝方时间（秒）")
        self.data_input_layout.addWidget(self.blue_time_input)

        self.red_board_numbers_input = QLineEdit()
        self.red_board_numbers_input.setFixedSize(280, 40)
        self.red_board_numbers_input.setPlaceholderText("请输入红方棋子布局（用逗号分隔）")
        self.data_input_layout.addWidget(self.red_board_numbers_input)

        self.blue_board_numbers_input = QLineEdit()
        self.blue_board_numbers_input.setFixedSize(280, 40)
        self.blue_board_numbers_input.setPlaceholderText("请输入蓝方棋子布局（用逗号分隔）")
        self.data_input_layout.addWidget(self.blue_board_numbers_input)

        self.left_bottom_layout.addLayout(self.data_input_layout)
        self.left_layout.addLayout(self.left_grid_layout)
        self.left_layout.addLayout(self.left_bottom_layout)
        main_layout.addLayout(self.left_layout)

        # 右侧布局：玩家信息、计时器和控制按钮
        self.right_layout = QVBoxLayout()
        self.right_layout.setSpacing(15)  # 减少右边内布局之间的间距

        # 玩家和计时器区域
        player_and_timer_layout = QVBoxLayout()

        # 当前 AI 角色显示
        self.user_label = QLabel(f"当前 AI 为: {self.ai_role}")
        self.user_label.setFont(QFont('Arial', 18))
        self.user_label.setAlignment(Qt.AlignCenter)  # 设置文本居中
        self.user_label.setFixedHeight(50)  # 设置固定高度，防止文本过长时影响布局
        player_and_timer_layout.addWidget(self.user_label)

        # 计时器区域
        timer_layout = QVBoxLayout()  # 改为垂直布局，保持简洁
        self.red_label = QLabel(f"红方计时器: {self.red_time}")
        self.red_label.setFont(QFont('Arial', 18))
        self.red_label.setAlignment(Qt.AlignCenter)
        timer_layout.addWidget(self.red_label)

        self.blue_label = QLabel(f"蓝方计时器: {self.blue_time}")
        self.blue_label.setFont(QFont('Arial', 18))
        self.blue_label.setAlignment(Qt.AlignCenter)
        timer_layout.addWidget(self.blue_label)

        # 骰子结果显示
        self.dice_label = QLabel("骰子结果: -")
        self.dice_label.setFont(QFont('Arial', 16, QFont.Bold))
        self.dice_label.setAlignment(Qt.AlignCenter)
        timer_layout.addWidget(self.dice_label)

        player_and_timer_layout.addLayout(timer_layout)
        self.right_layout.addLayout(player_and_timer_layout)

        # 骰子选择控件
        self.dice_select_box = QComboBox()
        self.dice_select_box.setFixedSize(280, 40)
        self.dice_select_box.addItem("选择骰子数")
        self.dice_select_box.addItems([str(i) for i in range(1, 7)])  # 1到6
        self.dice_select_box.currentIndexChanged.connect(self.select_dice)
        self.right_layout.addWidget(self.dice_select_box)

        # 中间添加空白区域
        self.right_layout.addStretch(1)

        # 工具按钮区域
        self.tool_layout = QVBoxLayout()

        # 定义按钮大小
        button_size = QSize(280, 40)

        # 添加切换红蓝方按钮
        self.switch_user_button = QPushButton("切换红蓝方")
        self.switch_user_button.setFixedSize(button_size)
        self.switch_user_button.clicked.connect(self.switch_user)
        self.tool_layout.addWidget(self.switch_user_button)

        # 添加重新开局按钮
        self.gen_board_button = QPushButton("重新开局")
        self.gen_board_button.setFixedSize(button_size)
        self.gen_board_button.clicked.connect(self.reset_game)
        self.tool_layout.addWidget(self.gen_board_button)

        # AI移动按钮
        self.ai_move_button = QPushButton("AI移动")
        self.ai_move_button.setFixedSize(button_size)
        self.ai_move_button.clicked.connect(self.ai_move)
        self.tool_layout.addWidget(self.ai_move_button)

        # 悔棋按钮
        self.undo_button = QPushButton("悔棋")
        self.undo_button.setFixedSize(button_size)
        self.undo_button.clicked.connect(self.undo_move)
        self.tool_layout.addWidget(self.undo_button)

        # 设置工具按钮的大小策略，保持自适应
        for widget in [self.gen_board_button, self.ai_move_button, self.undo_button, self.switch_user_button]:
            widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        # 将工具布局添加到右侧布局
        self.right_layout.addLayout(self.tool_layout)

        # 添加空白区域，使得控件不挤在一起
        self.right_layout.addStretch(2)

        # 设置右侧布局到主布局
        main_layout.addLayout(self.right_layout)

        # 设置布局到窗口
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def select_dice(self, index):
        """
        处理骰子选择事件。
        :param index: 选中的索引
        """
        if index == 0:
            self.current_dice = None
            self.dice_label.setText("骰子结果: -")
            print("骰子未选择。")
        else:
            self.current_dice = int(self.dice_select_box.currentText())
            self.dice_label.setText(f"骰子结果: {self.current_dice}")
            print(f"骰子选择: {self.current_dice}")

    def reset_game(self):
        """
        重置棋盘和游戏数据，根据输入框的内容或使用默认值。
        """
        print("重置游戏...")

        # 停止任何正在运行的AI线程，防止多线程冲突
        if hasattr(self, 'ai_thread') and self.ai_thread.isRunning():
            self.ai_thread.terminate()
            self.ai_thread.wait()
            print("停止了正在运行的AI线程。")

        # 获取输入框中的文本内容
        red_input_text = self.red_board_numbers_input.text().strip()
        blue_input_text = self.blue_board_numbers_input.text().strip()

        # 如果输入为空，则使用默认的数字列表
        if not red_input_text:
            self.person_numbers = list(range(1, 7))
            print("使用默认红方棋子布局。")
        else:
            try:
                self.person_numbers = [int(char) for char in red_input_text.split(',') if char.strip().isdigit()]
                if len(self.person_numbers) != 6:
                    raise ValueError("红方棋子数量不正确。")
                print(f"红方棋子布局: {self.person_numbers}")
            except ValueError:
                QMessageBox.warning(self, "输入错误", "红方棋子布局输入有误。请使用逗号分隔6个数字。使用默认布局。")
                self.person_numbers = list(range(1, 7))
                print("红方棋子布局输入有误，使用默认布局。")

        if not blue_input_text:
            self.ai_numbers = list(range(1, 7))
            print("使用默认蓝方棋子布局。")
        else:
            try:
                self.ai_numbers = [int(char) for char in blue_input_text.split(',') if char.strip().isdigit()]
                if len(self.ai_numbers) != 6:
                    raise ValueError("蓝方棋子数量不正确。")
                print(f"蓝方棋子布局: {self.ai_numbers}")
            except ValueError:
                QMessageBox.warning(self, "输入错误", "蓝方棋子布局输入有误。请使用逗号分隔6个数字。使用默认布局。")
                self.ai_numbers = list(range(1, 7))
                print("蓝方棋子布局输入有误，使用默认布局。")

        # 初始化棋盘
        self.board = self.reset_board()

        # 设置初始游戏状态
        self.current_turn = "红方"
        # 读取时间输入
        try:
            self.red_time = int(self.red_time_input.text()) if self.red_time_input.text() else 240
        except ValueError:
            self.red_time = 240
            QMessageBox.warning(self, "输入错误", "红方时间输入有误，使用默认值240秒。")
        try:
            self.blue_time = int(self.blue_time_input.text()) if self.blue_time_input.text() else 240
        except ValueError:
            self.blue_time = 240
            QMessageBox.warning(self, "输入错误", "蓝方时间输入有误，使用默认值240秒。")

        self.red_label.setText(f"红方计时器: {self.red_time}")
        self.blue_label.setText(f"蓝方计时器: {self.blue_time}")

        # 停止计时器
        self.red_timer.stop()
        self.blue_timer.stop()
        print("计时器已停止。")

        # 清空移动历史
        self.move_history = []
        print("移动历史已清空。")

        # 重置选择步骤标志
        self.sign_op_step = True
        print("选择步骤标志已重置。")

        # 重置选择的棋子信息
        self.current_row = None
        self.current_col = None
        self.old_button = None
        print("选择的棋子信息已重置。")

        # 重置骰子结果显示
        self.current_dice = None
        self.dice_label.setText("骰子结果: -")
        self.dice_select_box.setCurrentIndex(0)
        print("骰子结果已重置。")

        # 更新棋盘显示
        self.update_board()

    def reset_board(self):
        """
        根据玩家输入或默认值重置棋盘。
        :return: 新的棋盘数组。
        """
        print("重置棋盘...")
        board = np.zeros((self.board_size, self.board_size), dtype=int)

        # 定义填充位置和对应值的列表
        # 红方（人类）棋子为正数，蓝方（AI）棋子为负数
        positions_values = [
            ((0, 0), self.person_numbers[0]),
            ((0, 1), self.person_numbers[1]),
            ((0, 2), self.person_numbers[2]),
            ((1, 0), self.person_numbers[3]),
            ((1, 1), self.person_numbers[4]),
            ((2, 0), self.person_numbers[5]),
            ((2, 4), -self.ai_numbers[0]),
            ((3, 3), -self.ai_numbers[1]),
            ((3, 4), -self.ai_numbers[2]),
            ((4, 2), -self.ai_numbers[3]),
            ((4, 3), -self.ai_numbers[4]),
            ((4, 4), -self.ai_numbers[5])
        ]

        # 使用for循环填充矩阵
        for position, value in positions_values:
            row, col = position
            if 0 <= row < self.board_size and 0 <= col < self.board_size:
                board[row][col] = value
                print(f"设置位置 ({row}, {col}) = {value}")
            else:
                QMessageBox.warning(self, "布局错误", f"位置 {position} 超出棋盘范围。")
                print(f"警告: 位置 {position} 超出棋盘范围。")

        print("棋盘初始化完成：")
        print(board)
        return board

    def undo_move(self):
        """
        悔棋功能，恢复到上一个棋盘状态。
        """
        print("尝试悔棋...")
        if len(self.move_history) > 0:
            self.board = self.move_history.pop()
            print("已恢复上一个棋盘状态：")
            print(self.board)
            self.update_board()
            self.switch_turn()
            self.sign_op_step = True
        else:
            QMessageBox.information(self, "提示", "无法悔棋，已达到游戏开始状态。")
            print("无法悔棋，移动历史为空。")

    def ai_move(self):
        """
        使用训练好的模型和 MCTS 选择最佳动作并执行。
        """
        print("AI 移动按钮被点击。")

        # 检查骰子是否已选择
        if self.current_dice is None:
            QMessageBox.warning(self, "提示", "请先选择骰子数。")
            print("AI 未选择骰子数。")
            return

        state = self.convert_board_to_game_state()
        if state.is_terminal():
            QMessageBox.information(self, "游戏结束", "当前游戏已经结束。")
            print("游戏已经结束。")
            return

        # 创建并启动 AI 移动线程
        self.ai_thread = AIMoveThread(self.mcts, state)
        self.ai_thread.move_completed.connect(self.handle_ai_move)
        self.ai_thread.start()

    def handle_ai_move(self, best_action):
        """
        处理 AI 选择的动作，更新游戏状态和 UI。
        :param best_action: AI 选择的动作索引或动作元组。
        """
        print(f"处理 AI 选择的动作: {best_action}")
        if best_action is None:
            QMessageBox.warning(self, "警告", "AI 无法找到可行的动作。")
            print("AI 无法找到可行的动作。")
            return

        if isinstance(best_action, tuple) and len(best_action) == 2 and isinstance(best_action[0], tuple) and isinstance(best_action[1], tuple):
            # 假设它是一个移动元组
            action = best_action
            print(f"AI 选择的动作是一个移动元组: {action}")
        else:
            # 假设它是一个动作索引
            try:
                action = self.decode_action(best_action, 1 if self.current_turn == "红方" else -1, self.convert_board_to_game_state())
                print(f"AI 解码后的动作: {action}")
            except ValueError as e:
                print(f"解码动作失败: {e}")
                QMessageBox.warning(self, "错误", "AI 选择了一个无效的动作。")
                return

        # 保存当前棋盘状态到历史
        self.save_board_state()

        # 执行动作
        start_pos, end_pos = action
        start_i, start_j = start_pos
        end_i, end_j = end_pos

        piece_num = self.board[start_i][start_j]
        target_num = self.board[end_i][end_j]

        self.board[start_i][start_j] = 0
        if target_num != 0:
            print(f"棋子在目标位置被吃掉: {target_num}")
        self.board[end_i][end_j] = piece_num
        print(f"AI 已移动棋子到 ({end_i}, {end_j})")
        self.update_board()

        # 检查游戏是否结束
        state = self.convert_board_to_game_state()
        winner = state.get_winner()
        if winner is not None:
            if winner == 1:
                QMessageBox.information(self, "游戏结束", "红方获胜！")
                print("红方获胜！")
            elif winner == -1:
                QMessageBox.information(self, "游戏结束", "蓝方获胜！")
                print("蓝方获胜！")
            # 结束游戏
            self.reset_game()
        else:
            # 切换玩家并启动计时器
            self.switch_turn()
            self.start_timer()
            # 重置骰子选择
            self.current_dice = None
            self.dice_label.setText("骰子结果: -")
            self.dice_select_box.setCurrentIndex(0)

    def switch_turn(self):
        """
        切换当前玩家。
        """
        print(f"切换玩家：从 {self.current_turn} 到 ", end="")
        self.current_turn = "蓝方" if self.current_turn == "红方" else "红方"
        print(f"{self.current_turn}")
        if self.current_turn == self.ai_role:
            self.user_label.setText(f"当前 AI 为: {self.ai_role}")
            print(f"当前 AI 为: {self.ai_role}")
        else:
            self.user_label.setText("当前玩家为: 人类")
            print("当前玩家为: 人类")

    def start_timer(self):
        """
        启动对应玩家的计时器。
        """
        print(f"启动 {self.current_turn} 的计时器。")
        if self.current_turn == "红方":
            self.red_timer.start(1000)
            self.blue_timer.stop()
            print("红方计时器启动，蓝方计时器停止。")
        else:
            self.blue_timer.start(1000)
            self.red_timer.stop()
            print("蓝方计时器启动，红方计时器停止。")

    def update_red_timer(self):
        """
        更新红方计时器，每秒减少一次。
        """
        self.red_time -= 1
        self.red_label.setText(f"红方计时器: {self.red_time}")
        print(f"红方计时器: {self.red_time}")
        if self.red_time <= 0:
            self.end_game("蓝方获胜！")

    def update_blue_timer(self):
        """
        更新蓝方计时器，每秒减少一次。
        """
        self.blue_time -= 1
        self.blue_label.setText(f"蓝方计时器: {self.blue_time}")
        print(f"蓝方计时器: {self.blue_time}")
        if self.blue_time <= 0:
            self.end_game("红方获胜！")

    def end_game(self, message):
        """
        结束游戏，显示结果并重置游戏。
        :param message: 游戏结束的提示信息。
        """
        print(f"游戏结束: {message}")
        self.red_timer.stop()
        self.blue_timer.stop()
        QMessageBox.information(self, "对局结束", message)
        self.reset_game()

    # 更新后的 update_board 方法，修复棋子颜色显示问题
    def update_board(self):
        """
        更新棋盘显示，根据当前棋盘状态生成按钮。
        """
        print("更新棋盘显示...")
        # 清空现有按钮
        for i in reversed(range(self.left_grid_layout.count())):
            widget = self.left_grid_layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)

        # 获取项目目录路径，确保图片路径正确
        base_path = os.path.dirname(os.path.abspath(__file__))
        resource_path = os.path.join(base_path, "resource", "img")

        # 创建并添加按钮
        for row in range(self.board_size):
            for col in range(self.board_size):
                value = self.board[row][col]
                button = QPushButton()
                button.setFixedSize(100, 100)  # 使每个按钮有相同大小

                if value > 0:
                    icon_filename = f"{value}.jpg"
                elif value < 0:
                    icon_filename = f"{value}.jpg"  # 保留负号，加载负数文件名
                else:
                    icon_filename = "0.jpg"

                icon_path = os.path.join(resource_path, icon_filename)

                # 检查图片路径是否存在
                if not os.path.exists(icon_path):
                    QMessageBox.warning(self, "图片加载错误", f"图片路径不存在: {icon_path}")
                    print(f"图片路径不存在: {icon_path}")
                    icon_path = os.path.join(resource_path, "0.jpg")  # 使用默认空白图片

                button.setIcon(QIcon(icon_path))
                button.setIconSize(QtCore.QSize(100, 100))
                button.clicked.connect(lambda checked, b=button, r=row, c=col: self.handle_move(b, r, c))
                self.left_grid_layout.addWidget(button, row, col)
                print(f"添加按钮到棋盘位置 ({row}, {col})")

    def save_board_state(self):
        """
        保存当前棋盘状态到移动历史，用于悔棋功能。
        """
        self.move_history.append(self.board.copy())
        print("保存当前棋盘状态到历史。")

    def switch_user(self):
        """
        切换 AI 角色（红方或蓝方）。
        """
        self.ai_role = "蓝方" if self.ai_role == "红方" else "红方"
        self.user_label.setText(f"当前 AI 为: {self.ai_role}")
        QMessageBox.information(self, "切换成功", f"AI 已切换为 {self.ai_role}。")
        print(f"AI 已切换为 {self.ai_role}。")

    def handle_move(self, button, row, col):
        """
        处理用户点击棋盘按钮进行移动。
        :param button: 被点击的按钮。
        :param row: 按钮所在行。
        :param col: 按钮所在列。
        """
        print(f"用户点击了棋盘位置 ({row}, {col})")
        current_player = self.current_turn
        if self.ai_role == current_player:
            QMessageBox.warning(self, "提示", "现在是 AI 的回合，请等待 AI 移动。")
            print("当前是 AI 的回合，用户不能移动。")
            return

        if self.sign_op_step:
            # 用户选择棋子前，必须选择骰子数
            if self.current_dice is None:
                QMessageBox.warning(self, "提示", "请先选择骰子数。")
                print("用户未选择骰子数。")
                return

            # 选择棋子
            piece_num = self.board[row][col]
            print(f"用户尝试选择棋子: {piece_num}")
            if (current_player == "红方" and piece_num > 0) or (current_player == "蓝方" and piece_num < 0):
                # 检查是否选择的棋子符合骰子要求
                required_piece = self.current_dice  # 人类需要选择与骰子数相同编号的棋子
                if abs(piece_num) != required_piece:
                    QMessageBox.warning(self, "错误", f"请选择编号为 {required_piece} 的棋子！")
                    print(f"用户选择的棋子编号不符合骰子要求，需要移动编号为 {required_piece} 的棋子。")
                    return

                self.current_row = row
                self.current_col = col
                self.old_button = button
                selected_icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resource", "img", "f.jpg")
                if not os.path.exists(selected_icon_path):
                    QMessageBox.warning(self, "图片加载错误", f"图片路径不存在: {selected_icon_path}")
                    print(f"图片路径不存在: {selected_icon_path}")
                    selected_icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resource", "img", "0.jpg")
                button.setIcon(QIcon(selected_icon_path))  # 选中标记
                button.setIconSize(QSize(100, 100))
                self.sign_op_step = False
                print(f"棋子 ({row}, {col}) 已被选中。")
            else:
                QMessageBox.warning(self, "错误", "请选择您自己的棋子！")
                print("用户选择了无效的棋子。")
        else:
            # 移动棋子到目标位置
            target_num = self.board[row][col]
            print(f"用户尝试移动到位置 ({row}, {col})，目标棋子编号: {target_num}")

            # 检查是否选择了有效的移动
            piece_num = self.board[self.current_row][self.current_col]
            action = ((self.current_row, self.current_col), (row, col))
            print(f"用户移动动作: {action}")

            # 验证动作是否合法
            if not self.is_valid_move(action):
                QMessageBox.warning(self, "错误", "无效的移动！")
                print("用户尝试了无效的移动。")
                self.sign_op_step = True
                self.update_board()
                return

            # 保存当前棋盘状态到历史
            self.save_board_state()

            # 执行动作
            self.board[self.current_row][self.current_col] = 0
            if target_num != 0:
                print(f"棋子在目标位置被吃掉: {target_num}")
            self.board[row][col] = piece_num
            print(f"用户已移动棋子到 ({row}, {col})")
            self.update_board()

            # 检查游戏是否结束
            state = self.convert_board_to_game_state()
            winner = state.get_winner()
            if winner is not None:
                if winner == 1:
                    QMessageBox.information(self, "游戏结束", "红方获胜！")
                    print("红方获胜！")
                elif winner == -1:
                    QMessageBox.information(self, "游戏结束", "蓝方获胜！")
                    print("蓝方获胜！")
                # 结束游戏
                self.reset_game()
                return

            # 切换玩家并启动计时器
            self.switch_turn()
            self.start_timer()
            self.sign_op_step = True
            # 重置骰子选择
            self.current_dice = None
            self.dice_label.setText("骰子结果: -")
            self.dice_select_box.setCurrentIndex(0)

    def is_valid_move(self, action):
        """
        检查移动是否符合游戏规则。
        :param action: ((start_i, start_j), (end_i, end_j))
        :return: True 如果移动合法，否则 False。
        """
        start, end = action
        start_i, start_j = start
        end_i, end_j = end

        # 确保起始位置有棋子
        piece_num = self.board[start_i][start_j]
        print(f"验证移动：起始位置 ({start_i}, {start_j}) 的棋子编号: {piece_num}")
        if piece_num == 0:
            print("起始位置没有棋子。")
            return False

        # 确保目标位置在棋盘范围内
        if not (0 <= end_i < self.board_size and 0 <= end_j < self.board_size):
            print("目标位置超出棋盘范围。")
            return False

        # 计算移动方向
        di = end_i - start_i
        dj = end_j - start_j
        direction = (di, dj)
        print(f"移动方向: {direction}")

        # 定义合法的移动方向
        if (self.current_turn == "红方" and direction not in [(1, 0), (0, 1), (1, 1)]) \
                or (self.current_turn == "蓝方" and direction not in [(-1, 0), (0, -1), (-1, -1)]):
            print("移动方向不合法。")
            return False

        # 检查移动是否符合骰子规则
        required_piece = self.current_dice
        if required_piece is None:
            print("没有可移动的棋子符合骰子要求。")
            return False

        if abs(piece_num) != required_piece:
            print(f"移动的棋子编号 {abs(piece_num)} 不符合骰子要求 {required_piece}。")
            return False

        print("移动合法。")
        return True

    def get_required_piece(self, player):
        """
        AI 移动时，根据骰子结果，确定需要移动的棋子编号。
        :param player: "红方" 或 "蓝方"
        :return: 需要移动的棋子编号，或 None 如果没有可移动的棋子
        """
        dice = self.current_dice
        print(f"根据骰子 {dice} 确定需要移动的棋子。")

        # 获取玩家的棋子列表
        if player == "红方":
            pieces = [p for p in self.person_numbers if p in self.board]
        else:
            pieces = [p for p in self.ai_numbers if -p in self.board]

        # 检查骰子对应的棋子是否在棋盘上
        if dice in pieces:
            return dice
        else:
            # 找到最接近的棋子编号
            if not pieces:
                return None
            closest_piece = min(pieces, key=lambda x: (abs(x - dice), x))
            print(f"骰子对应的棋子已被移出，选择最接近的棋子编号: {closest_piece}")
            return closest_piece

    def convert_board_to_game_state(self):
        """
        将当前 UI 的棋盘转换为 GameState 对象。
        :return: GameState 实例。
        """
        game_state = GameState()
        game_state.board = self.board.copy()
        game_state.current_player = 1 if self.current_turn == "红方" else -1

        # 更新 red_list 和 blue_list
        red_pieces = []
        blue_pieces = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                value = self.board[i][j]
                if value > 0:
                    red_pieces.append(value)
                elif value < 0:
                    blue_pieces.append(-value)
        game_state.red_list = red_pieces
        game_state.blue_list = blue_pieces

        # 新增：传递当前骰子结果
        game_state.current_dice = self.current_dice

        print(f"转换后的游戏状态：玩家 {self.current_turn}，红方棋子 {game_state.red_list}，蓝方棋子 {game_state.blue_list}")
        return game_state

    def decode_action(self, action_index, current_player, game_state):
        """
        将动作索引解码为具体动作。
        :param action_index: 0-17
        :param current_player: 当前玩家，1代表红方，-1代表蓝方
        :param game_state: 当前游戏状态
        :return: ((start_i, start_j), (end_i, end_j))
        """
        print(f"解码动作索引: {action_index}")
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
        print(f"当前玩家: {'红方' if current_player == 1 else '蓝方'}, 方向: {direction}, 棋子编号: {piece_num}")

        # 查找棋子的当前位置
        positions = np.argwhere(game_state.board == (piece_num if current_player == 1 else -piece_num))
        if len(positions) == 0:
            raise ValueError(f"Piece {piece_num} not found on the board.")
        start_i, start_j = positions[0]  # 假设第一个找到的棋子
        end_i = start_i + direction[0]
        end_j = start_j + direction[1]

        # 检查目标位置是否在棋盘范围内
        if not (0 <= end_i < self.board_size and 0 <= end_j < self.board_size):
            raise ValueError(f"Decoded action leads to out-of-bounds position: ({end_i}, {end_j}).")

        print(f"解码后的动作: 从 ({start_i}, {start_j}) 到 ({end_i}, {end_j})")
        return (start_i, start_j), (end_i, end_j)

    def encode_action(self, action, current_player, game_state):
        """
        将动作编码为唯一的索引 (0-17)。
        :param action: ((start_i, start_j), (end_i, end_j))
        :param current_player: 当前玩家，1代表红方，-1代表蓝方
        :param game_state: 当前游戏状态
        :return: action_index (0-17)
        """
        start, end = action
        start_i, start_j = start
        end_i, end_j = end

        di = end_i - start_i
        dj = end_j - start_j
        direction = (di, dj)
        print(f"编码动作: 从 ({start_i}, {start_j}) 到 ({end_i}, {end_j})，方向: {direction}")

        # 获取方向索引
        if current_player == 1:
            direction_map = {(1, 0): 0, (0, 1): 1, (1, 1): 2}
            piece_num = game_state.board[start_i][start_j]
            try:
                piece_index = game_state.red_list.index(piece_num)
                print(f"红方棋子编号 {piece_num} 的索引: {piece_index}")
            except ValueError:
                raise ValueError(f"Red piece {piece_num} not found in red_list.")
        else:
            direction_map = {(-1, 0): 0, (0, -1): 1, (-1, -1): 2}
            piece_num = -game_state.board[start_i][start_j]
            try:
                piece_index = game_state.blue_list.index(piece_num)
                print(f"蓝方棋子编号 {piece_num} 的索引: {piece_index}")
            except ValueError:
                raise ValueError(f"Blue piece {piece_num} not found in blue_list.")

        direction_index = direction_map.get(direction, -1)
        if direction_index == -1:
            raise ValueError(f"Invalid direction {direction} for player {current_player}.")

        action_index_result = piece_index * self.num_directions + direction_index  # 0-17
        print(f"编码后的动作索引: {action_index_result}")
        return action_index_result


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EinsteinChess()
    window.setWindowTitle("Einstein Chess")
    window.show()
    sys.exit(app.exec())
