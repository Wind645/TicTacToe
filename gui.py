import tkinter as tk
from tkinter import messagebox, filedialog
import numpy as np
import torch
import os
import sys
import random

# 添加项目路径到sys.path
project_path = os.path.dirname(os.path.abspath(__file__))
if project_path not in sys.path:
    sys.path.append(project_path)

from dqn_agent import dqn_agent
from env import TicTacToe

class TicTacToeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("井字棋 - AI对弈")
        self.root.geometry("400x500")
        self.root.resizable(False, False)
        
        # 初始化设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化游戏环境
        self.env = TicTacToe()
        
        # 初始化AI代理（暂未加载模型）
        self.agent = None
        
        # 玩家角色 (1: X, -1: O)
        self.player_role = 1
        
        # 创建UI组件
        self.create_widgets()
    
    def create_widgets(self):
        # 创建标题
        title_label = tk.Label(self.root, text="井字棋 - AI对弈", font=("Arial", 20, "bold"))
        title_label.pack(pady=10)
        
        # 创建框架容纳棋盘
        board_frame = tk.Frame(self.root)
        board_frame.pack(pady=10)
        
        # 创建井字棋按钮
        self.buttons = []
        for i in range(3):
            row_buttons = []
            for j in range(3):
                button = tk.Button(
                    board_frame, 
                    text="",
                    font=("Arial", 24),
                    width=3,
                    height=1,
                    command=lambda i=i, j=j: self.make_move(i, j)
                )
                button.grid(row=i, column=j, padx=5, pady=5)
                row_buttons.append(button)
            self.buttons.append(row_buttons)
        
        # 创建控制框架
        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=10)
        
        # 加载模型按钮
        load_button = tk.Button(
            control_frame,
            text="加载AI模型",
            font=("Arial", 12),
            command=self.load_model
        )
        load_button.grid(row=0, column=0, padx=10)
        
        # 重置游戏按钮
        reset_button = tk.Button(
            control_frame,
            text="重新开始",
            font=("Arial", 12),
            command=self.reset_game
        )
        reset_button.grid(row=0, column=1, padx=10)
        
        # 玩家角色选择
        role_frame = tk.Frame(self.root)
        role_frame.pack(pady=10)
        
        role_label = tk.Label(role_frame, text="选择您的角色:", font=("Arial", 12))
        role_label.grid(row=0, column=0, padx=10)
        
        self.role_var = tk.StringVar(value="X")
        
        x_radio = tk.Radiobutton(role_frame, text="X (先手)", variable=self.role_var, value="X", 
                                 command=self.change_role, font=("Arial", 12))
        o_radio = tk.Radiobutton(role_frame, text="O (后手)", variable=self.role_var, value="O", 
                                 command=self.change_role, font=("Arial", 12))
        
        x_radio.grid(row=0, column=1, padx=5)
        o_radio.grid(row=0, column=2, padx=5)
        
        # 状态标签
        self.status_label = tk.Label(self.root, text="请先加载AI模型", font=("Arial", 12))
        self.status_label.pack(pady=10)
    
    def load_model(self):
        file_path = filedialog.askopenfilename(
            title="选择AI模型文件",
            filetypes=[("PyTorch模型", "*.pth"), ("所有文件", "*.*")]
        )
        
        if not file_path:
            return
            
        try:
            # 初始化代理并加载模型
            self.agent = dqn_agent(self.device)
            self.agent.load(file_path)
            self.status_label.config(text=f"模型已加载: {os.path.basename(file_path)}")
            self.reset_game()
        except Exception as e:
            messagebox.showerror("加载错误", f"无法加载模型: {str(e)}")
    
    def change_role(self):
        if self.role_var.get() == "X":
            self.player_role = 1
        else:
            self.player_role = -1
        self.reset_game()
    
    def reset_game(self):
        # 如果没有加载模型，不能开始游戏
        if self.agent is None:
            self.status_label.config(text="请先加载AI模型")
            return
            
        # 重置环境，根据玩家角色确定先手
        first_player = 1  # 默认玩家1先手
        self.env.reset(first_player=first_player)
        
        # 清空按钮
        for i in range(3):
            for j in range(3):
                self.buttons[i][j]["text"] = ""
                self.buttons[i][j]["state"] = "normal"
                self.buttons[i][j]["bg"] = "SystemButtonFace"
        
        # 更新状态
        self.status_label.config(text="游戏已开始")
        
        # 如果AI先手
        if self.env.player != self.player_role:
            self.ai_move()
    
    def make_move(self, i, j):
        if self.agent is None:
            messagebox.showinfo("提示", "请先加载AI模型")
            return
            
        if self.env.board[i, j] != 0:  # 检查位置是否已有棋子
            return
            
        # 玩家走棋
        action = (i, j)
        board, reward, done, _, info = self.env.step(action)
        
        # 检查是否有错误信息
        if info and isinstance(info, dict) and 'Error' in info:
            print(f"玩家动作错误: {info}")
            return
        
        # 更新UI
        symbol = "X" if self.player_role == 1 else "O"
        self.buttons[i][j]["text"] = symbol
        
        # 添加调试信息
        print(f"玩家放置棋子在 ({i},{j}), 游戏结束?: {done}")
        print(f"当前棋盘状态:\n{self.env.board}")
        
        # 检查游戏是否结束
        if done:
            self.end_game()
            return
            
        # AI走棋
        self.ai_move()
    
    def ai_move(self):
        # 获取有效动作
        valid_actions = self.env.get_valid_action()
        if not valid_actions:
            # 没有有效动作，游戏结束
            self.end_game()
            return
        
        # 获取AI动作
        state = self.env.get_board()
        action = self.agent.select_action(state)
        
        # 确保AI选择有效动作
        if action not in valid_actions:
            print(f"AI选择了无效动作 {action}，从有效动作中随机选择")
            action = random.choice(valid_actions)
        
        # 执行AI动作
        i, j = action
        board, reward, done, _, info = self.env.step(action)
        
        # 检查是否有错误信息
        if info and isinstance(info, dict) and 'Error' in info:
            print(f"AI动作错误: {info}")
            # 从有效动作中随机选择
            action = random.choice(valid_actions)
            i, j = action
            board, reward, done, _, _ = self.env.step(action)
        
        # 更新UI
        symbol = "O" if self.player_role == 1 else "X"
        self.buttons[i][j]["text"] = symbol
        
        # 添加调试信息
        print(f"AI放置棋子在 ({i},{j}), 游戏结束?: {done}")
        print(f"当前棋盘状态:\n{self.env.board}")
        
        # 检查游戏是否结束
        if done:
            self.end_game()
    
    def end_game(self):
        # 禁用所有按钮
        for i in range(3):
            for j in range(3):
                self.buttons[i][j]["state"] = "disabled"
        
        # 显示游戏结果
        winner = self.env.winner
        
        if winner == self.player_role:
            messagebox.showinfo("游戏结束", "恭喜，你赢了！")
            self.status_label.config(text="游戏结束 - 你赢了！")
        elif winner == -self.player_role:
            messagebox.showinfo("游戏结束", "AI赢了！")
            self.status_label.config(text="游戏结束 - AI赢了！")
        else:
            messagebox.showinfo("游戏结束", "平局！")
            self.status_label.config(text="游戏结束 - 平局！")
            
        # 高亮显示获胜路线
        self.highlight_winner()
    
    def highlight_winner(self):
        """高亮显示获胜路线"""
        board = self.env.board
        winner = self.env.winner
        
        if winner == 0:  # 平局
            return
            
        # 检查行
        for i in range(3):
            if abs(sum(board[i])) == 3:
                for j in range(3):
                    self.buttons[i][j]["bg"] = "light green"
                return
                
        # 检查列
        for j in range(3):
            if abs(sum(board[:, j])) == 3:
                for i in range(3):
                    self.buttons[i][j]["bg"] = "light green"
                return
                
        # 检查对角线
        if abs(board[0, 0] + board[1, 1] + board[2, 2]) == 3:
            self.buttons[0][0]["bg"] = "light green"
            self.buttons[1][1]["bg"] = "light green"
            self.buttons[2][2]["bg"] = "light green"
            return
            
        if abs(board[0, 2] + board[1, 1] + board[2, 0]) == 3:
            self.buttons[0][2]["bg"] = "light green"
            self.buttons[1][1]["bg"] = "light green"
            self.buttons[2][0]["bg"] = "light green"
            return

def main():
    root = tk.Tk()
    app = TicTacToeGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
