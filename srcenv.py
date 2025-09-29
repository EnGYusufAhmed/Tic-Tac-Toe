from typing import Tuple, List, Optional
import numpy as np

WINNING_LINES = [
    (0,1,2),(3,4,5),(6,7,8),
    (0,3,6),(1,4,7),(2,5,8),
    (0,4,8),(2,4,6)
]

class TicTacToe:
    def __init__(self):
        self.reset()

    def reset(self) -> Tuple[Tuple[int,...], int]:
        # 0 empty, 1 X, -1 O
        self.board = [0]*9
        self.current_player = 1  # agent by default
        self.winner = None
        self.done = False
        return self._get_state(), self.current_player

    def _get_state(self) -> Tuple[int,...]:
        return tuple(self.board)

    def available_actions(self) -> List[int]:
        return [i for i,v in enumerate(self.board) if v==0]

    def step(self, action: int, player: int) -> Tuple[Tuple[int,...], float, bool, dict]:
        if self.done:
            raise ValueError("Game already finished")
        if self.board[action] != 0:
            raise ValueError("Invalid action")
        self.board[action] = player
        reward = 0.0
        self.winner = self._check_winner()
        if self.winner is not None:
            self.done = True
            if self.winner == 0:
                reward = 0.5  # draw
            elif self.winner == 1:
                reward = 1.0
            else:
                reward = -1.0
        elif not self.available_actions():
            self.done = True
            self.winner = 0
            reward = 0.5
        else:
            self.done = False

        return self._get_state(), reward, self.done, {"winner": self.winner}

    def _check_winner(self) -> Optional[int]:
        for a,b,c in WINNING_LINES:
            s = self.board[a] + self.board[b] + self.board[c]
            if s == 3:
                return 1
            if s == -3:
                return -1
        return None

    def render(self):
        chars = {1: 'X', -1: 'O', 0: ' '}
        board_lines = [f" {chars[self.board[i]]} | {chars[self.board[i+1]]} | {chars[self.board[i+2]]} " for i in range(0,9,3)]
        sep = '\n---+---+---\n'
        print(sep.join(board_lines))

    def clone(self):
        new = TicTacToe()
        new.board = self.board.copy()
        new.current_player = self.current_player
        new.winner = self.winner
        new.done = self.done
        return new