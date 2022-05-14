def mcts_minimax_hybrid_agent(observation, configuration):
    """
    Connect X agent based on MCTS.
    """
    import random
    import math
    import time
    import numpy as np
    global current_state  # so tree can be recycled

    init_time = time.time()
    EMPTY = 0
    T_max = 5 - 0.2  # time per move, left some overhead
    Cp_default = 0.5
    
    def minimax_subagent(board, config, mark):
        # Gets board at next step if agent drops piece in selected column
        def drop_piece(grid, col, mark, config):
            next_grid = grid.copy()
            for row in range(config.rows-1, -1, -1):
                if next_grid[row][col] == 0:
                    break
            next_grid[row][col] = mark
            return next_grid

        # Helper function for get_heuristic: checks if window satisfies heuristic conditions
        def check_window(window, pieces, config):
            num_windows = np.zeros(pieces.shape[0])
            mark = [window.count(pieces[0, 1]), window.count(pieces[-1, 1])] 
            if mark[0] > 0 and mark[1] > 0 or mark[0] < 2 and mark[1] < 2:
                pass
            elif mark[0]:
                num_windows[mark[0] - 2] = 1
            else:
                num_windows[mark[1] + 1] = 1
            return num_windows

        # Helper function for get_heuristic: counts number of windows satisfying specified heuristic conditions
        def count_windows(grid, pieces, config):
            num_windows = np.zeros(pieces.shape[0])
            # horizontal
            for row in range(config.rows):
                for col in range(config.columns-(config.inarow-1)):
                    window = list(grid[row, col:col+config.inarow])
                    num_windows += check_window(window, pieces, config)
            # vertical
            for row in range(config.rows-(config.inarow-1)):
                for col in range(config.columns):
                    window = list(grid[row:row+config.inarow, col])
                    num_windows += check_window(window, pieces, config)
            # positive diagonal
            for row in range(config.rows-(config.inarow-1)):
                for col in range(config.columns-(config.inarow-1)):
                    window = list(grid[range(row, row+config.inarow), range(col, col+config.inarow)])
                    num_windows += check_window(window, pieces, config)
            # negative diagonal
            for row in range(config.inarow-1, config.rows):
                for col in range(config.columns-(config.inarow-1)):
                    window = list(grid[range(row, row-config.inarow, -1), range(col, col+config.inarow)])
                    num_windows += check_window(window, pieces, config)
            return num_windows


        def get_heuristic(grid, mark, config):
            op_mark = mark % 2 + 1
            x = config.inarow
            l = []
            for i in range(2, x+1):
                l.append([i, mark])
            for i in range(2, x+1):
                l.append([i, op_mark])
            pieces = np.array(l)
            num_windows = count_windows(grid, pieces, config)
            c = []
            for i in range(2, x+1):
                c.append(1000**(i-1))
            for i in range(2, x+1):
                c.append(-(2000**(i-1)))
            cost = np.array(c)
            score = np.sum(num_windows * cost)
            return score


        # Uses minimax to calculate value of dropping piece in selected column
        def score_move(grid, col, mark, config, nsteps):
            next_grid = drop_piece(grid, col, mark, config)
            score = minimax(next_grid, nsteps-1, False, mark, config, -np.Inf)
            return score

        # Helper function for minimax: checks if agent or opponent has four in a row in the window
        def is_terminal_window(window, config):
            return window.count(1) == config.inarow or window.count(2) == config.inarow

        # Helper function for minimax: checks if game has ended
        def is_terminal_node(grid, config):
            # Check for draw 
            if list(grid[0, :]).count(0) == 0:
                return True
            # Check for win: horizontal, vertical, or diagonal
            # horizontal 
            for row in range(config.rows):
                for col in range(config.columns-(config.inarow-1)):
                    window = list(grid[row, col:col+config.inarow])
                    if is_terminal_window(window, config):
                        return True
            # vertical
            for row in range(config.rows-(config.inarow-1)):
                for col in range(config.columns):
                    window = list(grid[row:row+config.inarow, col])
                    if is_terminal_window(window, config):
                        return True
            # positive diagonal
            for row in range(config.rows-(config.inarow-1)):
                for col in range(config.columns-(config.inarow-1)):
                    window = list(grid[range(row, row+config.inarow), range(col, col+config.inarow)])
                    if is_terminal_window(window, config):
                        return True
            # negative diagonal
            for row in range(config.inarow-1, config.rows):
                for col in range(config.columns-(config.inarow-1)):
                    window = list(grid[range(row, row-config.inarow, -1), range(col, col+config.inarow)])
                    if is_terminal_window(window, config):
                        return True
            return False

        # Minimax implementation
        def minimax(node, depth, maximizingPlayer, mark, config, up_value):
            is_terminal = is_terminal_node(node, config)
            valid_moves = [c for c in range(config.columns) if node[0][c] == 0]
            if depth == 0 or is_terminal:
                return get_heuristic(node, mark, config)
            if maximizingPlayer:
                value = -np.Inf
                for col in valid_moves:
                    child = drop_piece(node, col, mark, config)
                    value = max(value, minimax(child, depth-1, False, mark, config, value))
                    if value >= up_value:
                        break
                return value
            else:
                value = np.Inf
                for col in valid_moves:
                    child = drop_piece(node, col, mark%2+1, config)
                    value = min(value, minimax(child, depth-1, True, mark, config, value))
                    if value <= up_value:
                        break
                return value


        N_STEPS = 2

        # Get list of valid moves
        valid_moves = [c for c in range(config.columns) if board[c] == 0]
        # Convert the board to a 2D grid
        grid = np.asarray(board).reshape(config.rows, config.columns)
        # Use the heuristic to assign a score to each possible board in the next step
        scores = dict(zip(valid_moves, [score_move(grid, col, mark, config, N_STEPS) for col in valid_moves]))
        # Get a list of columns (moves) that maximize the heuristic
        max_cols = [key for key in scores.keys() if scores[key] == max(scores.values())]
        # Select at random from the maximizing columns
        return random.choice(max_cols)

    def play(board, column, mark, config):
        """ Plays a move. Taken from the Kaggle environment. """
        columns = config.columns
        rows = config.rows
        row = max([r for r in range(rows) if board[column + (r * columns)] == EMPTY])
        board[column + (row * columns)] = mark

    def is_win(board, column, mark, config):
        """ Checks for a win. Taken from the Kaggle environment. """
        columns = config.columns
        rows = config.rows
        inarow = config.inarow - 1
        row = min([r for r in range(rows) if board[column + (r * columns)] == mark])

        def count(offset_row, offset_column):
            for i in range(1, inarow + 1):
                r = row + offset_row * i
                c = column + offset_column * i
                if (
                        r < 0
                        or r >= rows
                        or c < 0
                        or c >= columns
                        or board[c + (r * columns)] != mark
                ):
                    return i - 1
            return inarow

        return (
                count(1, 0) >= inarow  # vertical.
                or (count(0, 1) + count(0, -1)) >= inarow  # horizontal.
                or (count(-1, -1) + count(1, 1)) >= inarow  # top left diagonal.
                or (count(-1, 1) + count(1, -1)) >= inarow  # top right diagonal.
        )

    def is_tie(board):
        """ Checks if a tie occured. """
        return not(any(mark == EMPTY for mark in board))

    def check_finish_and_score(board, column, mark, config):
        """ Returns a tuple where the first argument states whether game is finished and second argument returns score if game has finished. """
        if is_win(board, column, mark, config):
            return (True, 1)
        if is_tie(board):
            return (True, 0.5)
        else:
            return (False, None)

    def uct_score(node_total_score, node_total_visits, parent_total_visits, Cp=Cp_default):
        """ UCB1 calculation. """
        if node_total_visits == 0:
            return math.inf
        return node_total_score / node_total_visits + Cp * math.sqrt(
            2 * math.log(parent_total_visits) / node_total_visits)

    def opponent_mark(mark):
        """ The mark indicates which player is active - player 1 or player 2. """
        return 3 - mark

    def opponent_score(score):
        """ To backpropagate scores on the tree. """
        return 1 - score

    def random_action(board, config):
        """ Returns a random legal action (from the open columns). """
        return random.choice([c for c in range(config.columns) if board[c] == EMPTY])
    
    def informed_action(board, config, mark):
        """ Uses shallow minimax to guide rollout. """
        return minimax_subagent(board, config, mark)

    def default_policy_simulation(board, mark, config):
        """
        Run a random play simulation. Starting state is assumed to be a non-terminal state.
        Returns score of the game for the player with the given mark.
        """
        start_time = time.time()
        original_mark = mark
        board = board.copy()
        column = informed_action(board, config, mark)
        play(board, column, mark, config)
        is_finish, score = check_finish_and_score(board, column, mark, config)
        num_informed_actions = 2
        while not is_finish:
            mark = opponent_mark(mark)
            if num_informed_actions > 0:
                column = informed_action(board, config, mark)
                num_informed_actions -= 1
            else:
                column = random_action(board, config)
            play(board, column, mark, config)
            is_finish, score = check_finish_and_score(board, column, mark, config)
        # print(time.time()-start_time)
        if mark == original_mark:
            return score
        return opponent_score(score)
    
    def find_action_taken_by_opponent(new_board, old_board, config):
        """ Given a new board state and a previous one, finds which move was taken. Used for recycling tree between moves. """
        for i, piece in enumerate(new_board):
            if piece != old_board[i]:
                return i % config.columns
        return -1  # shouldn't get here

    class State():
        """ 
        A class that represents nodes in the game tree.
        
        """
        def __init__(self, board, mark, config, parent=None, is_terminal=False, terminal_score=None, action_taken=None):
            self.board = board.copy()
            self.mark = mark
            self.config = config
            self.children = []
            self.parent = parent
            self.node_total_score = 0
            self.node_total_visits = 0
            self.available_moves = [c for c in range(config.columns) if board[c] == EMPTY]
            self.expandable_moves = self.available_moves.copy()
            self.is_terminal = is_terminal
            self.terminal_score = terminal_score
            self.action_taken = action_taken

        def is_expandable(self):
            """ Checks if the node has unexplored children. """
            return (not self.is_terminal) and (len(self.expandable_moves) > 0)

        def expand_and_simulate_child(self):
            """ Expands a random move from the legal unexplored moves, and runs a simulation of it 
            (Expansion + Simulation + Backpropagation stages in the MCTS algorithm description). """
            column = random.choice(self.expandable_moves)
            child_board = self.board.copy()
            play(child_board, column, self.mark, self.config)
            is_terminal, terminal_score = check_finish_and_score(child_board, column, self.mark, self.config)
            self.children.append(State(child_board, opponent_mark(self.mark),
                                       self.config, parent=self,
                                       is_terminal=is_terminal,
                                       terminal_score=terminal_score,
                                       action_taken=column
                                       ))
            simulation_score = self.children[-1].simulate()
            self.children[-1].backpropagate(simulation_score)
            self.expandable_moves.remove(column)

        def choose_strongest_child(self, Cp):
            """
            Chooses child that maximizes UCB1 score (Selection stage in the MCTS algorithm description).
            """
            children_scores = [uct_score(child.node_total_score,
                                         child.node_total_visits,
                                         self.node_total_visits,
                                         Cp) for child in self.children]
            max_score = max(children_scores)
            best_child_index = children_scores.index(max_score)
            return self.children[best_child_index]
            
        def choose_play_child(self):
            """ Choose child with maximum total score."""
            children_scores = [child.node_total_score for child in self.children]
            max_score = max(children_scores)
            best_child_index = children_scores.index(max_score)
            return self.children[best_child_index]

        def tree_single_run(self):
            """
            A single iteration of the 4 stages of the MCTS algorithm.
            """
            if self.is_terminal:
                self.backpropagate(self.terminal_score)
                return
            if self.is_expandable():
                self.expand_and_simulate_child()
                return
            self.choose_strongest_child(Cp_default).tree_single_run()

        def simulate(self):
            """
            Runs a simulation from the current state. 
            This method is used to simulate a game after move of current player, so if a terminal state was reached,
            the score would belong to the current player who made the move.
            But otherwise the score received from the simulation run is the opponent's score and thus needs to be flipped with the function opponent_score().            
            """
            if self.is_terminal:
                return self.terminal_score
            return opponent_score(default_policy_simulation(self.board, self.mark, self.config))

        def backpropagate(self, simulation_score):
            """
            Backpropagates score and visit count to parents.
            """
            self.node_total_score += simulation_score
            self.node_total_visits += 1
            if self.parent is not None:
                self.parent.backpropagate(opponent_score(simulation_score))
                
        def choose_child_via_action(self, action):
            """ Choose child given the action taken from the state. Used for recycling of tree. """
            for child in self.children:
                if child.action_taken == action:
                    return child
            return None

    board = observation.board
    mark = observation.mark
    
    # If current_state already exists, recycle it based on action taken by opponent
    try:  
        current_state = current_state.choose_child_via_action(
            find_action_taken_by_opponent(board, current_state.board, configuration))
        current_state.parent = None  # make current_state the root node, dereference parents and siblings
        
    except:  # new game or other error in recycling attempt due to Kaggle mechanism
        current_state = State(board, mark,  # This state is considered after the opponent's move
                              configuration, parent=None, is_terminal=False, terminal_score=None, action_taken=None)
   
    # Run MCTS iterations until time limit is reached.
    num_sims = 0
    while time.time() - init_time <= T_max:
        current_state.tree_single_run()
        num_sims += 1
    # print(num_sims)
        
    current_state = current_state.choose_play_child()
    return current_state.action_taken
