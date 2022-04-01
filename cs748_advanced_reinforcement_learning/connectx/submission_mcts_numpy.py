def mcts_agent_numpy(observation, configuration):
    global current_state

    import numpy as np
    import random
    from scipy.signal import convolve2d
    import time

    start_time = time.time()

    termination_time = configuration.timeout - 0.1
    
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
          # later for caching
          ans = ""
          # for c in range(self.board.shape()[0]):
          #     for r in range
          return ans

        @property
        def p1_board(self):
          return (self.board == 1)

        @property
        def p2_board(self):
          return (self.board == 2)

        @property
        def winning(self):
          # use bitmaps?
          cols, rows = self.board.shape
          for p in [1, 2]:
            horizontal_kernel = np.ones((1, self.inarow))
            vertical_kernel = horizontal_kernel.T
            diag1_kernel = np.eye(self.inarow, dtype=np.uint8)
            diag2_kernel = np.fliplr(diag1_kernel)
            detection_kernels = [
                horizontal_kernel, vertical_kernel, diag1_kernel, diag2_kernel
            ]
            for kernel in detection_kernels:
              if (convolve2d(self.board == p, kernel,
                             mode="valid") == self.inarow).any():
                return p
          return 0

        @property
        def draw(self):
          return (self.board == 0).sum() == 0

        @property
        def next_player(self):
          if self.p1_board.sum() == self.p2_board.sum():
            return 1
          return 2

        def display(self):
          return np.flip(self.board.T, axis=0)

        def play(self, column, player=None):
          if not player:
            player = self.next_player
          row = np.where(self.board[column] == 0)[0]
          if not len(row):
            raise Exception('Column %d is full' % column)
          row = row[0]
          new_board = self.board.copy()
          new_board[column, row] = player
          return new_board

        def playable_column(self, col):
          return self.board[col, -1]==0


    class MCTSNode():
        """
          Class for a node in the Monte Carlo Tree
          """

        def __init__(self, state_object, parent=None, parent_action=None,
                     is_terminal=None, terminal_value=None):
          # state_object: object representing the state
          # parent: node that created this node
          # parent_action: action that the parent took to create this node
          # is_terminal: whether node is a terminal node
          # terminal_value: terminal_value of node(and hence state)
          self.state = state_object
          self.children = []
          self.parent = parent
          self.parent_action = parent_action

          if is_terminal:
            self.is_terminal = is_terminal
            self.terminal_value = terminal_value
          else:
            winner = self.state.winning
            is_draw = self.state.draw
            self.is_terminal = ((winner != 0) or is_draw)
            if self.is_terminal:
              self.terminal_value = 0.5 if is_draw else 1
            else:
              self.terminal_value = None

          self.total_value = 0
          self.total_visits = 0
          self.unexpanded_moves = [c for c in range(self.state.board.shape[0])\
          if self.state.playable_column(c)]

        @property
        def is_leaf(self):
          # no selection if either terminal state, or atleast one unsimulated child
          return (len(self.unexpanded_moves) != 0) or (self.is_terminal)

        @property
        def ucb_value(self):
          C = np.sqrt(2)  # tweakable
          if self.total_visits == 0:
            return 1e10
          return (self.total_value / self.total_visits) + C * np.sqrt(
              np.log(self.parent.total_visits) / self.total_visits)

        def selection(self):
          # possible vectorization of ucb score calculation
          if not self.is_leaf:
            ucb_vals = [child.ucb_value for child in self.children]
            return self.children[np.argmax(ucb_vals)].selection()
          return self

        def expansion(self):
          if self.is_terminal or len(self.unexpanded_moves) == 0:
            return
          c = random.choice(self.unexpanded_moves)
          self.unexpanded_moves.remove(c)
          child_board = self.state.play(c)
          child_state = ConnectXState(child_board, self.state.inarow)
          self.children.append(MCTSNode(
            child_state, parent=self, parent_action=c)
          ) # is_terminal and terminal_value are autoset for the time being

        def simulate(self):
          # random playout with score that of the player
          # who sees this state before playing
          if self.is_terminal:
            return self.terminal_value

          child_state = self.state
          is_terminal = False
          while not is_terminal:
            playable_cols = [c for c in range(child_state.board.shape[0])\
              if child_state.playable_column(c)]
            c = random.choice(playable_cols)
            child_state = ConnectXState(child_state.play(c), self.state.inarow)
            winner = child_state.winning
            is_draw = child_state.draw
            is_terminal = ((winner != 0) or is_draw)
          if is_draw:
            return 0.5
          if winner == self.state.next_player:
            return 1
          return 0

        def backpropagate(self, score):
          self.total_visits += 1
          self.total_value += score
          if self.parent:
            self.parent.backpropagate(1 - score)

        def iteration(self):
          selected = self.selection()
          selected.expansion()
          score = selected.simulate()
          selected.backpropagate(score)

        def best_move(self):
          total_visits = [child.total_visits for child in self.children]
          return self.children[np.argmax(total_visits)].parent_action

        def child_of_action(self, new_board):
          old_board = self.state.board
          for c in range(old_board.shape[0]):
            if not (old_board[c]==new_board[c]).all():
              last_action = c
              break
          for child in self.children:
            if child.parent_action == c:
              return child
          return None

    board = np.flip(np.array(observation.board).reshape(
    (configuration.rows, configuration.columns)).T, axis=1)

    try:
        current_state = current_state.child_of_action(board)
    except:
        state = ConnectXState(board, configuration.inarow)
        current_state = MCTSNode(state)

    num_sims = 0
    while time.time() - start_time <= termination_time:
        current_state.iteration()
        num_sims += 1
     
    # print(num_sims)
    return current_state.best_move()