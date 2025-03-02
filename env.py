import numpy as np
import random

class TicTacToe:
    
    def __init__(self):
        self.board = np.zeros((3, 3))
        self.player = 1
        self.done = False
        self.winner = None
        self.truncated = False
        
    def reset(self, first_player=1):
        """重置游戏状态，支持指定先手玩家"""
        self.board = np.zeros((3, 3))
        self.player = first_player  # 使用传入的参数而不是随机值
        self.done = False
        self.winner = None
        self.truncated = False
        return self.board.copy()
        
    def step(self, action):  #action is a tuple of (i, j)
        if self.done == True:
            raise Exception('Game is over')
            
        # 检查动作是否有效
        if action not in self.get_valid_action():
            # 返回错误但不终止游戏
            return self.board.copy(), -10, False, self.truncated, {'Error': 'invalid move'}
            
        # 执行动作
        self.board[action[0], action[1]] = self.player
        
        # 检查游戏是否结束
        winner = self.check_winner()
        if winner is not None:
            self.winner = winner
            self.done = True
            return self.board.copy(), 3 * self.player, True, self.truncated, {}
            
        # 检查是否还有空位
        if len(self.get_valid_action()) == 0:
            self.winner = 0  # 平局
            self.done = True
            return self.board.copy(), 0, True, self.truncated, {}
        
        # 游戏继续
        self.player *= -1
        return self.board.copy(), 0, False, self.truncated, {}
        
    def get_valid_action(self):
        """获取所有有效的动作"""
        valid_actions = []
        for i in range(3):
            for j in range(3):
                if self.board[i, j] == 0:
                    valid_actions.append((i, j))
        return valid_actions
    
    def check_winner(self):
        # 检查行
        for i in range(3):
            if abs(np.sum(self.board[i])) == 3:
                return self.board[i, 0]
        
        # 检查列
        for j in range(3):
            if abs(np.sum(self.board[:, j])) == 3:
                return self.board[0, j]
        
        # 检查对角线
        if abs(self.board[0, 0] + self.board[1, 1] + self.board[2, 2]) == 3:
            return self.board[0, 0]
            
        if abs(self.board[0, 2] + self.board[1, 1] + self.board[2, 0]) == 3:
            return self.board[0, 2]
            
        # 没有获胜者
        return None
    
    # 移除不再使用的方法，它们可能导致了胜负判定的问题
    def get_reward_done(self):
        # 这个方法不再使用
        pass
    
    def check_is_done(self):
        # 这个方法不再使用
        pass
    
    def get_board(self):
        return self.board.copy()


