from submission_greedy import *
from submission_mcts_normal_5 import *
from submission_minimax import *
from submission_mcts_minimax_hybrid import *
# from submission_alphazero import *

from kaggle_environments import evaluate, make, utils

print(evaluate("connectx", [MCTS_agent, greedy_agent], num_episodes=10))

print(evaluate("connectx", [minimax_agent, greedy_agent], num_episodes=10))
print(evaluate("connectx", [minimax_agent, MCTS_agent], num_episodes=10))

print(evaluate("connectx", [mcts_minimax_hybrid_agent, greedy_agent], num_episodes=10))
print(evaluate("connectx", [mcts_minimax_hybrid_agent, MCTS_agent], num_episodes=10))
print(evaluate("connectx", [mcts_minimax_hybrid_agent, minimax_agent], num_episodes=10))

# todo: run comparisions for alphazero
