import torch
import numpy as np

# 对于输入特征信息的格式N*4*5*5
# N -> 批次大小,即一次需要处理的样本数量
# 4 -> 输入的通道数(自己棋子的位置,对手棋子的位置，棋子的数字标记，历史博弈走法)
# 5*5 -> 棋盘的大小


def train_valid_data_process():
    chess_board = np.zeros((5,5))
    print(np.shape(chess_board))
    print(chess_board)
    return None



def test_data_process():
    return None

if __name__ == '__main__':
    train_valid_data_process()
