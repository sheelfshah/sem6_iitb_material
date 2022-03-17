from mcts import *
# from connectx import *

default_state = ConnectXState(np.zeros((7, 6)), 4)
root = MCTSNode(default_state)

for _ in range(10000):
  selected = root.selection()
  selected.expansion()
  score = selected.simulate()
  selected.backpropagate(score)

for c in root.children:
  print(c.ucb_value)