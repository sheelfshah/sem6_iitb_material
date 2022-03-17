import numpy as np
import random

from connectx import *


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
    return (len(self.unexpanded_moves) != 0) and (not self.is_terminal)

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
    if self.is_terminal:
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

    state = self.state
    playable_cols = [c for c in range(state.board.shape[0])\
      if state.playable_column(c)]
    c = random.choice(playable_cols)
    child_state = ConnectXState(state.play(c), self.state.inarow)
    winner = state.winning
    is_draw = state.draw
    is_terminal = ((winner != 0) or is_draw)
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
    if winner == state.next_player:
      return 1
    return 0

  def backpropagate(self, score):
    self.total_visits += 1
    self.total_value += score
    if self.parent:
      self.parent.backpropagate(1 - score)