import numpy as np
import random
import tkinter as tk
from tkinter import messagebox
from collections import defaultdict
from threading import Thread


class EnhancedTicTacToeAI:
    def __init__(self):
        self.board = [' '] * 9
        self.q_table = defaultdict(lambda: [0.0]*9)
        self.alpha = 0.7
        self.gamma = 0.95
        self.epsilon = 0.54
        self.epsilon_decay = 0.9999
        self.min_epsilon = 0.01
        self.win_patterns = [(0,1,2), (3,4,5), (6,7,8),
                            (0,3,6), (1,4,7), (2,5,8),
                            (0,4,8), (2,4,6)]
        self.priority_positions = {4: 0.3, 0:0.1, 2:0.1, 6:0.1, 8:0.1}

    def reset(self):
        self.board = [' '] * 9

    def get_state(self):
        return tuple(self.board)

    def available_actions(self):
        return [i for i, c in enumerate(self.board) if c == ' ']

    def check_win(self, player):
        for p in self.win_patterns:
            if self.board[p[0]] == self.board[p[1]] == self.board[p[2]] == player:
                return True
        return False

    def is_draw(self):
        return ' ' not in self.board

    def expert_move(self, player):
        """改进的专家规则系统"""
        opponent = 'O' if player == 'X' else 'X'

        # 立即获胜
        for a in self.available_actions():
            temp = self.board.copy()
            temp[a] = player
            if self.check_win(player):
                return a

        # 阻止对手获胜
        for a in self.available_actions():
            temp = self.board.copy()
            temp[a] = opponent
            if self.check_win(opponent):
                return a

        # 优先中心位置
        if 4 in self.available_actions():
            return 4

        # 选择角落
        corners = [0, 2, 6, 8]
        available_corners = [c for c in corners if c in self.available_actions()]
        if available_corners:
            return random.choice(available_corners)

        # 随机选择剩余位置
        return random.choice(self.available_actions())

    def choose_action(self, state):
        """改进的动作选择策略"""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        if random.random() < self.epsilon:
            return self.expert_move('O')  # 使用专家规则探索
        else:
            state_actions = self.q_table[state]
            available = self.available_actions()
            q_values = [state_actions[a] + np.random.normal(0, 0.01) for a in available]  # 添加噪声
            max_q = max(q_values)
            best_actions = [available[i] for i, q in enumerate(q_values) if q == max_q]
            return random.choice(best_actions)

    def get_reward(self, player):
        """改进的奖励函数"""
        if self.check_win(player):
            return 10
        opponent = 'O' if player == 'X' else 'X'
        if self.check_win(opponent):
            return -10
        if self.is_draw():
            return 3
        # 位置奖励
        return sum(self.priority_positions.get(i, 0) for i in self.available_actions())

    def update_q_table(self, state, action, next_state, reward):
        """改进的Q值更新"""
        max_next_q = max(self.q_table[next_state]) if next_state in self.q_table else 0
        self.q_table[state][action] += self.alpha * (
                reward + self.gamma * max_next_q - self.q_table[state][action]
        )

    def train(self, episodes=30000):
        """改进的训练逻辑"""
        for ep in range(episodes):
            self.reset()
            current_player = 'X'
            state = self.get_state()

            while True:
                # AI玩家'O'的回合
                action = self.choose_action(state)
                self.board[action] = 'O'
                next_state = self.get_state()
                reward = self.get_reward('O')
                self.update_q_table(state, action, next_state, reward)

                if self.check_win('O') or self.is_draw():
                    break

                # 对手'X'使用专家规则
                opp_action = self.expert_move('X')
                self.board[opp_action] = 'X'
                next_state = self.get_state()

                if self.check_win('X') or self.is_draw():
                    # 惩罚对手获胜的情况
                    self.update_q_table(state, action, next_state, -5)
                    break

                state = next_state

            # 每1000局降低学习率
            if ep % 1000 == 0:
                self.alpha = max(0.1, self.alpha * 0.999)


class TicTacToeGUI:
    def __init__(self, master):
        self.master = master
        self.ai = EnhancedTicTacToeAI()
        self.init_gui()
        self.start_training()
        self.game_active = False

    def init_gui(self):
        self.master.title("增强型井字棋AI")

        # 游戏棋盘
        self.canvas = tk.Canvas(self.master, width=300, height=300, bg='white')
        self.canvas.pack(pady=20)
        self.draw_board()

        # 控制面板
        control_frame = tk.Frame(self.master)
        control_frame.pack(pady=10)

        self.train_btn = tk.Button(control_frame, text="开始训练", command=self.start_training)
        self.train_btn.pack(side=tk.LEFT, padx=5)

        self.play_btn = tk.Button(control_frame, text="开始游戏", state=tk.DISABLED,
                                  command=self.start_game)  # 改为start_game
        self.play_btn.pack(side=tk.LEFT, padx=5)

        self.status_var = tk.StringVar()
        self.status_label = tk.Label(self.master, textvariable=self.status_var, font=('Arial', 12))
        self.status_label.pack()

        # 绑定事件（修复1：只在游戏激活时处理点击）
        self.canvas.bind("<Button-1>", self.handle_click)

    def start_game(self):  # 新增方法
        self.game_active = True
        self.ai.reset()
        self.canvas.delete("all")
        self.draw_board()
        self.status_var.set("你的回合（X）")
        self.play_btn.config(state=tk.DISABLED)
        self.train_btn.config(state=tk.DISABLED)

    def draw_board(self):
        for i in range(1, 3):
            self.canvas.create_line(i * 100, 0, i * 100, 300, width=2)
            self.canvas.create_line(0, i * 100, 300, i * 100, width=2)

    def start_training(self):
        self.train_btn.config(state=tk.DISABLED)
        self.status_var.set("训练中...")

        def training_thread():
            self.ai.train(episodes=100000)
            self.master.after(0, lambda: [
                self.play_btn.config(state=tk.NORMAL),
                self.status_var.set("训练完成！点击开始游戏")
            ])

        Thread(target=training_thread).start()

    def new_game(self):
        self.ai.reset()
        self.canvas.delete("all")
        self.draw_board()
        self.status_var.set("你的回合（X）")

    def handle_click(self, event):
        if not self.game_active:  # 修复2：检查游戏状态
            return

        # 修复3：使用更可靠的回合判断逻辑
        x_count = self.ai.board.count('X')
        o_count = self.ai.board.count('O')
        if x_count > o_count:
            return

        x = event.x // 100
        y = event.y // 100
        pos = y * 3 + x

        if 0 <= pos < 9 and self.ai.board[pos] == ' ':
            self.make_move(pos, 'X')
            if not self.check_game_over():
                self.master.after(500, self.ai_move)

    def ai_move(self):
        state = self.ai.get_state()
        action = self.ai.choose_action(state)
        if action is not None and 0 <= action < 9:
            self.make_move(action, 'O')
            self.check_game_over()

    def make_move(self, pos, player):
        if self.ai.board[pos] == ' ':
            self.ai.board[pos] = player
            self.draw_symbol(pos, player)
            self.update_status()


    def draw_symbol(self, pos, player):
        x = (pos % 3) * 100 + 50
        y = (pos // 3) * 100 + 50
        color = 'blue' if player == 'X' else 'red'
        if player == 'X':
            self.canvas.create_line(x - 30, y - 30, x + 30, y + 30, width=3, fill=color)
            self.canvas.create_line(x + 30, y - 30, x - 30, y + 30, width=3, fill=color)
        else:
            self.canvas.create_oval(x - 30, y - 30, x + 30, y + 30, outline=color, width=3)

    def ai_move(self):
        state = self.ai.get_state()
        action = self.ai.choose_action(state)
        self.make_move(action, 'O')
        self.check_game_over()

    def check_game_over(self):
        if self.ai.check_win('X'):
            self.show_result("玩家获胜！")
            self.game_active = False  # 修复4：结束游戏时更新状态
            return True
        elif self.ai.check_win('O'):
            self.show_result("AI获胜！")
            self.game_active = False
            return True
        elif self.ai.is_draw():
            self.show_result("平局！")
            self.game_active = False
            return True
        return False

    def show_result(self, msg):
        messagebox.showinfo("游戏结束", msg)
        self.play_btn.config(state=tk.NORMAL)
        self.train_btn.config(state=tk.NORMAL)
        self.game_active = False  # 修复5：重置游戏状态

    def update_status(self):
        x_count = self.ai.board.count('X')
        o_count = self.ai.board.count('O')
        status = "玩家回合（X）" if x_count == o_count else "AI回合（O）"
        self.status_var.set(f"{status} | X:{x_count} O:{o_count}")


if __name__ == "__main__":
    root = tk.Tk()
    game = TicTacToeGUI(root)
    root.mainloop()