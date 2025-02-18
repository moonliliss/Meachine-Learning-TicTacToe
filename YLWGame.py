import numpy as np
import random
import tkinter as tk
from tkinter import messagebox


class TicTacToeAI:
    def __init__(self):
        self.board = [' '] * 9
        self.current_player = 'X'
        self.q_table = {}
        self.alpha = 0.5
        self.gamma = 0.9
        self.epsilon = 0.1

    def reset(self):
        self.board = [' '] * 9
        self.current_player = 'X'

    def get_state(self):
        return tuple(self.board)

    def available_actions(self):
        return [i for i, cell in enumerate(self.board) if cell == ' ']

    def check_win(self, player):
        win_patterns = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
                        (0, 3, 6), (1, 4, 7), (2, 5, 8),
                        (0, 4, 8), (2, 4, 6)]
        for pattern in win_patterns:
            if self.board[pattern[0]] == self.board[pattern[1]] == self.board[pattern[2]] == player:
                return True
        return False

    def is_draw(self):
        return ' ' not in self.board

    def make_move(self, action, player):
        if self.board[action] == ' ':
            self.board[action] = player
            return True
        return False

    def get_reward(self, player):
        if self.check_win(player):
            return 1
        elif self.check_win('O' if player == 'X' else 'X'):
            return -1
        elif self.is_draw():
            return 0.5
        else:
            return 0

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.available_actions())
        else:
            state_actions = self.q_table.get(state, [0] * 9)
            available_actions = self.available_actions()
            q_values = [state_actions[action] for action in available_actions]
            max_q = max(q_values)
            best_actions = [available_actions[i] for i, q in enumerate(q_values) if q == max_q]
            return random.choice(best_actions)

    def update_q_table(self, state, action, next_state, reward):
        old_q = self.q_table.get(state, [0] * 9)[action]
        max_next_q = max(self.q_table.get(next_state, [0] * 9))
        new_q = old_q + self.alpha * (reward + self.gamma * max_next_q - old_q)

        if state not in self.q_table:
            self.q_table[state] = [0] * 9
        self.q_table[state][action] = new_q

    def train(self, episodes=10000):
        for _ in range(episodes):
            self.reset()
            state = self.get_state()
            while True:
                action = self.choose_action(state)
                self.make_move(action, self.current_player)
                next_state = self.get_state()
                reward = self.get_reward(self.current_player)

                self.update_q_table(state, action, next_state, reward)

                if self.check_win(self.current_player) or self.is_draw():
                    break

                self.current_player = 'O' if self.current_player == 'X' else 'X'
                state = next_state


class TicTacToeGUI:
    def __init__(self, master):
        self.master = master
        self.ai = TicTacToeAI()
        self.init_gui()
        self.ai.train(10000)
        self.reset_game()

    def init_gui(self):
        self.canvas = tk.Canvas(self.master, width=300, height=300, bg='white')
        self.canvas.pack()
        self.draw_board()

        self.status_label = tk.Label(self.master, text="当前玩家: X", font=('Arial', 12))
        self.status_label.pack()

        self.restart_btn = tk.Button(self.master, text="重新开始", command=self.reset_game)
        self.restart_btn.pack()

        self.canvas.bind("<Button-1>", self.handle_click)

    def draw_board(self):
        # 绘制棋盘线
        for i in range(1, 3):
            self.canvas.create_line(i * 100, 0, i * 100, 300, width=2)
            self.canvas.create_line(0, i * 100, 300, i * 100, width=2)

    def draw_symbol(self, position, player):
        row = position // 3
        col = position % 3
        x = col * 100 + 50
        y = row * 100 + 50
        color = 'blue' if player == 'X' else 'red'
        self.canvas.create_text(x, y, text=player, font=('Arial', 40), fill=color)

    def handle_click(self, event):
        if self.ai.current_player == 'O':
            return

        col = event.x // 100
        row = event.y // 100
        position = row * 3 + col

        if self.ai.board[position] == ' ':
            self.make_move(position, 'X')
            if not self.check_game_over():
                self.master.after(500, self.ai_move)

    def ai_move(self):
        state = self.ai.get_state()
        action = self.ai.choose_action(state)
        self.make_move(action, 'O')
        self.check_game_over()

    def make_move(self, position, player):
        if self.ai.make_move(position, player):
            self.draw_symbol(position, player)
            self.ai.current_player = 'O' if player == 'X' else 'X'
            self.status_label.config(text=f"当前玩家: {self.ai.current_player}")

    def check_game_over(self):
        if self.ai.check_win('X'):
            self.show_result("玩家获胜！")
            return True
        elif self.ai.check_win('O'):
            self.show_result("AI获胜！")
            return True
        elif self.ai.is_draw():
            self.show_result("平局！")
            return True
        return False

    def show_result(self, message):
        messagebox.showinfo("游戏结束", message)
        self.reset_game()

    def reset_game(self):
        self.ai.reset()
        self.canvas.delete("all")
        self.draw_board()
        self.status_label.config(text="当前玩家: X")


if __name__ == "__main__":
    root = tk.Tk()
    root.title("井字棋 - Q学习AI")
    game = TicTacToeGUI(root)
    root.mainloop()