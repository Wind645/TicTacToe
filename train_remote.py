"""
远程训练脚本 - 用于在无GUI环境中训练井字棋AI模型
此脚本适合在远程服务器或无显示环境中运行
"""

import torch
import numpy as np
import os
import time
import argparse
import sys
from pathlib import Path

# 确保可以导入项目中的模块
current_file = Path(__file__).resolve()
project_root = current_file.parent
sys.path.append(str(project_root))

try:
    from dqn_agent import dqn_agent
    from env import TicTacToe
except ImportError:
    print("错误：无法导入必要模块。请确保您在正确的目录中运行此脚本。")
    sys.exit(1)

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='远程训练井字棋AI模型')
    parser.add_argument('--episodes', type=int, default=100000,
                        help='训练回合数 (默认: 100000)')
    parser.add_argument('--save-interval', type=int, default=1000,
                        help='保存模型的间隔回合数 (默认: 1000)')
    parser.add_argument('--output-dir', type=str, default='models',
                        help='保存模型的目录 (默认: models)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='学习率 (默认: 0.01)')
    return parser.parse_args()

def train_remote():
    """在远程环境中训练井字棋AI模型"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 记录训练配置
    with open(os.path.join(args.output_dir, "training_config.txt"), "w") as f:
        f.write(f"Episodes: {args.episodes}\n")
        f.write(f"Save interval: {args.save_interval}\n")
        f.write(f"Learning rate: {args.lr}\n")
        f.write(f"Training start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 初始化环境和智能体
    env = TicTacToe()
    agent_1 = dqn_agent(device, lr=args.lr)
    agent_2 = dqn_agent(device, lr=args.lr)
    
    # 记录训练统计信息
    stats = {
        'agent_1_wins': 0,
        'agent_2_wins': 0,
        'draws': 0,
        'episode_history': []
    }
    
    start_time = time.time()
    last_save_time = start_time
    
    print(f"开始训练 {args.episodes} 个回合...")
    
    try:
        for i in range(args.episodes):
            if i % 100 == 0:
                current_time = time.time()
                elapsed = current_time - start_time
                print(f"回合 {i+1}/{args.episodes} ({(i+1)/args.episodes*100:.1f}%)")
                print(f"经过时间: {elapsed:.2f} 秒")
                
                if i > 0:
                    games_per_second = i / elapsed
                    estimated_remaining = (args.episodes - i) / games_per_second
                    print(f"速度: {games_per_second:.2f} 游戏/秒")
                    print(f"估计剩余时间: {estimated_remaining/60:.1f} 分钟")
                    
                    # 显示胜率统计
                    total_games = i
                    if total_games > 0:
                        print(f"智能体1胜率: {stats['agent_1_wins']/total_games*100:.2f}%")
                        print(f"智能体2胜率: {stats['agent_2_wins']/total_games*100:.2f}%")
                        print(f"平局率: {stats['draws']/total_games*100:.2f}%")
                print("-" * 40)
            
            # 定期保存模型
            if (i+1) % args.save_interval == 0:
                save_path_1 = os.path.join(args.output_dir, f"agent_1_episode_{i+1}.pth")
                save_path_2 = os.path.join(args.output_dir, f"agent_2_episode_{i+1}.pth")
                
                agent_1.save(save_path_1)
                agent_2.save(save_path_2)
                
                current_save_time = time.time()
                print(f"已保存模型到 {save_path_1}")
                print(f"已保存模型到 {save_path_2}")
                print(f"保存耗时: {current_save_time - last_save_time:.2f} 秒")
                last_save_time = current_save_time
            
            # 执行一个回合
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
                
                board = next_board
            
            # 记录结果
            if env.winner == 1:
                stats['agent_1_wins'] += 1
            elif env.winner == -1:
                stats['agent_2_wins'] += 1
            else:
                stats['draws'] += 1
                
            # 每1000回合记录一次历史
            if (i+1) % 1000 == 0:
                stats['episode_history'].append({
                    'episode': i+1,
                    'agent_1_wins': stats['agent_1_wins'],
                    'agent_2_wins': stats['agent_2_wins'],
                    'draws': stats['draws'],
                    'time': time.time() - start_time
                })
            
    except KeyboardInterrupt:
        print("\n训练被用户中断。保存当前模型...")
    except Exception as e:
        print(f"\n训练出错: {e}")
        print("保存当前模型...")
    finally:
        # 保存最终模型
        final_path_1 = os.path.join(args.output_dir, "agent_1_final.pth")
        final_path_2 = os.path.join(args.output_dir, "agent_2_final.pth")
        
        try:
            agent_1.save(final_path_1)
            agent_2.save(final_path_2)
            print(f"已保存最终模型到 {final_path_1}")
            print(f"已保存最终模型到 {final_path_2}")
        except Exception as e:
            print(f"保存最终模型时出错: {e}")
        
        # 保存训练统计信息
        try:
            import json
            stats_path = os.path.join(args.output_dir, "training_stats.json")
            with open(stats_path, 'w') as f:
                json.dump(stats, f)
            print(f"已保存训练统计信息到 {stats_path}")
            
            # 记录训练结束时间
            with open(os.path.join(args.output_dir, "training_config.txt"), "a") as f:
                f.write(f"\nTraining end time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                total_time = time.time() - start_time
                f.write(f"Total training time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)\n")
        except Exception as e:
            print(f"保存统计信息时出错: {e}")
    
    total_time = time.time() - start_time
    print(f"\n训练完成！")
    print(f"总训练时间: {total_time:.2f} 秒 ({total_time/3600:.2f} 小时)")
    print(f"智能体1胜率: {stats['agent_1_wins']/args.episodes*100:.2f}%")
    print(f"智能体2胜率: {stats['agent_2_wins']/args.episodes*100:.2f}%")
    print(f"平局率: {stats['draws']/args.episodes*100:.2f}%")

if __name__ == "__main__":
    train_remote()
