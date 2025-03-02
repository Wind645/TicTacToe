from dqn_agent import *
from env import TicTacToe
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time

'''
 I will introduce two agents here to represent two opponents.
 One represent the player 1 and another represent the player -1.
 They compete with each other in the env, and then I will pick 
 the -1 player to play the game with me
'''

'''
 But there is one thing that I have to point out, this kind of design actually
 wastes half of the neural network trained, as two agents each have their own
 Qnet, but I didn't leverage them both, maybe they can be both used if I play with
 them both. Ahh, but I won't do that.
'''


episodes = 100000 # They will play with each other for 100000 times.

if torch.cuda.is_available:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
print(f'Using device: {device}')

env = TicTacToe()
agent_1 = dqn_agent(device).to(device)
agent_2 = dqn_agent(device).to(device) # The one that gonna play with me later on

def train():
    # 创建保存模型的目录
    os.makedirs("models", exist_ok=True)
    
    start_time = time.time()
    # 记录胜率统计
    stats = {
        'agent_1_wins': 0,
        'agent_2_wins': 0,
        'draws': 0
    }
    
    for i in range(episodes):
        if i % 100 == 0:
            print(f'Fighting in episode {i+1}/{episodes}')
            
            # 每1000回合保存一次模型
            if i > 0 and i % 1000 == 0:
                # 保存两个模型，以便后续可以选择更好的那个
                agent_1.save(f"models/agent_1_episode_{i}.pth")
                agent_2.save(f"models/agent_2_episode_{i}.pth")
                
                # 显示时间和胜率统计
                elapsed_time = time.time() - start_time
                print(f"Time elapsed: {elapsed_time:.2f} seconds")
                total_games = i
                if total_games > 0:
                    print(f"Agent 1 win rate: {stats['agent_1_wins']/total_games*100:.2f}%")
                    print(f"Agent 2 win rate: {stats['agent_2_wins']/total_games*100:.2f}%")
                    print(f"Draw rate: {stats['draws']/total_games*100:.2f}%")
        
        env.reset()
        board = env.get_board()
        done = False
        while not done:
            if env.player == 1:
                action = agent_1.select_action(board)
                agent_moved = agent_1
                agent_not_moved = agent_2
            else:
                action = agent_2.select_action(board)
                agent_moved = agent_2
                agent_not_moved = agent_1
            next_board, reward, done, _, _ = env.step(action)
            info1 = (board, action, next_board, reward, done)
            info2 = (board, action, next_board, -reward, done)
            agent_moved.update(info1)
            agent_not_moved.update(info2)
            # This is where I split the information and assign them to each agent,
            # as you can see, the difference lies in the sign of the reward.
            board = next_board
            
        # 更新胜率统计
        if env.winner == 1:
            stats['agent_1_wins'] += 1
        elif env.winner == -1:
            stats['agent_2_wins'] += 1
        else:
            stats['draws'] += 1
    
    # 训练结束，保存最终模型
    print("Training completed. Saving final models...")
    agent_1.save("models/agent_1_final.pth")
    agent_2.save("models/agent_2_final.pth")
    
    # 显示最终统计信息
    total_games = episodes
    print(f"\nFinal Statistics:")
    print(f"Agent 1 win rate: {stats['agent_1_wins']/total_games*100:.2f}%")
    print(f"Agent 2 win rate: {stats['agent_2_wins']/total_games*100:.2f}%")
    print(f"Draw rate: {stats['draws']/total_games*100:.2f}%")
    print(f"Total training time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    train()

