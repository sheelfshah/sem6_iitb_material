from mcts import *
import time
from kaggle_environments import evaluate, make, utils

env = make("connectx", debug=True)
configuration = env.configuration
print(configuration)

def mcts_agent_numpy(observation, configuration):
  global current_state
  start_time = time.time()
  termination_time = configuration.timeout - 0.1

  board = np.flip(np.array(observation.board).reshape(
    (configuration.columns, configuration.rows)), axis=1)
  mark = observation.mark

  try:
    raise Exception()
    current_state = current_state.child_of_action
  except:
    state = ConnectXState(board, configuration.inarow)
    current_state = MCTSNode(state)

  while time.time() - start_time <= termination_time:
    current_state.iteration()

  return current_state.best_move()