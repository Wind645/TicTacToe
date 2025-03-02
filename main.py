import os
import sys
import tkinter as tk
from tkinter import messagebox

# 添加项目路径到sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

def run_training():
    """运行训练过程"""
    try:
        from train import train
        print("开始训练模型...")
        train()
        messagebox.showinfo("训练完成", "模型训练完成！模型已保存到'models'目录。")
    except Exception as e:
        messagebox.showerror("训练错误", f"训练过程中出错: {str(e)}")

def run_gui():
    """运行GUI界面"""
    try:
        from gui import main as gui_main
        print("启动游戏界面...")
        gui_main()
    except Exception as e:
        messagebox.showerror("GUI错误", f"启动游戏界面时出错: {str(e)}")

class MainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("井字棋AI - 主菜单")
        self.root.geometry("400x300")
        self.root.resizable(False, False)
        
        # 创建UI组件
        self.create_widgets()
    
    def create_widgets(self):
        # 标题
        title_label = tk.Label(self.root, text="井字棋 AI 系统", font=("Arial", 24, "bold"))
        title_label.pack(pady=30)
        
        # 按钮框架
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=20)
        
        # 训练模型按钮
        train_button = tk.Button(
            button_frame,
            text="训练模型",
            font=("Arial", 14),
            width=15,
            height=2,
            command=self.start_training
        )
        train_button.pack(pady=10)
        
        # 开始游戏按钮
        play_button = tk.Button(
            button_frame,
            text="开始游戏",
            font=("Arial", 14),
            width=15,
            height=2,
            command=self.start_gui
        )
        play_button.pack(pady=10)
        
        # 退出按钮
        exit_button = tk.Button(
            button_frame,
            text="退出",
            font=("Arial", 14),
            width=15,
            height=1,
            command=self.root.destroy
        )
        exit_button.pack(pady=20)
    
    def start_training(self):
        """确认并开始训练"""
        response = messagebox.askyesno("确认", "训练可能需要较长时间，是否继续？")
        if response:
            self.root.destroy()  # 关闭当前窗口
            run_training()
    
    def start_gui(self):
        """启动游戏界面"""
        self.root.destroy()  # 关闭当前窗口
        run_gui()

def main():
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()