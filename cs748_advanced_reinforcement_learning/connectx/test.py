from mcts import *
import time
from kaggle_environments import evaluate, make, utils

env = make("connectx", debug=True)
configuration = env.configuration
print(configuration)

def mcts_agent_numpy(observation, configuration):
  global current_state
  start_time = time.time()
  termination_time = 5 - 0.1

  board = np.flip(np.array(observation.board).reshape(
    (configuration.rows, configuration.columns)).T, axis=1)

  try:
    current_state = current_state.child_of_action(board)
    current_state.parent = None
  except:
    state = ConnectXState(board, configuration.inarow)
    current_state = MCTSNode(state)

  while time.time() - start_time <= termination_time:
    current_state.iteration()

  return current_state.best_move()

def mean_reward(rewards):
  return sum(r[0] for r in rewards) / sum(1 for r in rewards)

# print("My Agent vs Random Agent:", mean_reward(evaluate("connectx", [mcts_agent_numpy, "random"], num_episodes=20)))
# print("Random Agent vs My Agent:", mean_reward(evaluate("connectx", ["random", mcts_agent_numpy], num_episodes=20)))
# print(evaluate("connectx", [mcts_agent_numpy, "negamax"], num_episodes=5))
# print(evaluate("connectx", ["negamax", mcts_agent_numpy], num_episodes=5))

board = np.array([[0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0]])

def play_from_a_state(board, inarow):
  global current_state
  start_time = time.time()
  termination_time = 2 - 0.1
  state = ConnectXState(board, inarow)
  current_state = MCTSNode(state)
  while time.time() - start_time <= termination_time:
    current_state.iteration()
  return current_state.best_move()

def game():
  board = np.array([[0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0]])
  while True:
    col = play_from_a_state(board, 4)
    board = current_state.state.play(col)
    print(board)
    if current_state.is_terminal:
      return

# print(evaluate("connectx", [mcts_agent_numpy, mcts_agent_numpy], num_episodes=5))