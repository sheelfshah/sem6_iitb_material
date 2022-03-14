import numpy as np
from scipy.signal import convolve2d

configuration = {'episodeSteps': 1000, 'actTimeout': 2, 'runTimeout': 1200,
    'columns': 7, 'rows': 6, 'inarow': 4, 'agentTimeout': 60, 'timeout': 2}

game_board = np.array([[0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0],
       [0, 0, 2, 0, 0, 0],
       [0, 0, 0, 2, 0, 0],
       [0, 0, 0, 0, 2, 0],
       [1, 2, 1, 2, 1, 2]])

class ConnectXState():
    """
    Class representing a state of the game
    """
    def __init__(self, board, inarow):
        # board: 2d np array with 1st index column, 2nd index row
        # inarow: number of inarow tokens for winning
        self.board = board.copy()
        self.inarow = inarow

    @property
    def to_str(self):
        ans = ""
        # for c in range(self.board.shape()[0]):
        #     for r in range
        return ans

    @property
    def p1_board(self):
        return (self.board==1)

    @property
    def p2_board(self):
        return (self.board==2)

    @property
    def winning(self):
        # needs optimization
        cols, rows = self.board.shape
        for p in [1, 2]:
            horizontal_kernel = np.ones((1, self.inarow))*p
            vertical_kernel = horizontal_kernel.T
            diag1_kernel = np.eye(self.inarow, dtype=np.uint8)
            diag2_kernel = np.fliplr(diag1_kernel)
            detection_kernels = [horizontal_kernel, vertical_kernel, diag1_kernel, diag2_kernel]
            for kernel in detection_kernels:
                if (convolve2d(self.board==p, kernel, mode="valid") == self.inarow).any():
                    return p
        return 0

    def display(self):
        return np.flip(self.board.T, axis=0)

    def play(self, column, player):
        row = np.where(self.board[column]==0)[0]
        if not len(row):
            raise Exception('Column %d is full' % column)
        row = row[0]
        new_board = self.board.copy()
        new_board[column, row] = player
        return new_board
