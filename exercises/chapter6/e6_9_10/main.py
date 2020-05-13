import numpy as np
import random
from tqdm.notebook import tqdm
from typing import Tuple, Any, Optional
import __future__

"""
Abstract class representing a learning environment
"""
class Environment():
  def __init__(self):
    pass

  """
  returns a list all possible actions at state s
  """
  def actions(self, s: Any) -> list:
    raise NotImplementedError()
  
  """
    takes action a at state s returning new state s'
  """
  def next_state(self, s:Any, a: Any) -> Tuple[float, Optional[Any]]:
    raise NotImplementedError()
  
  """
  returns whichever value is to be used as the terminal state
  """
  def terminal_state(self) -> Any:
    raise NotImplementedError()

  """
  returns the start states
  """
  def start_states(self) -> [Any]:
    raise NotImplementedError()

  def gamma(self) -> float:
    raise NotImplementedError()

"""
dict that returns default value for uninitialized data
"""
class MapWithDefault():
  default_value: Any
  table: dict
  def __init__(self, default_value):
    self.default_value = default_value
    self.table = dict()
  
  def __getitem__(self, key):
    if key in self.table:
      return self.table[key]
    else:
      return self.default_value
  
  def __setitem__(self, key, value):
    self.table[key] = value

"""
dict that returns default value for uninitialized data
"""
class EpsilonGreedy():
  def __init__(self, env:Environment, action_value: dict, epsilon:float=0.1):
    self.env = env
    self.epsilon = epsilon
    self.action_value = action_value
  
  def __getitem__(self, state):
    actions = self.env.actions(state)
    if np.random.uniform() < self.epsilon:
      action = random.choice(actions)
      return action
    else:
      index = np.argmax(
        [self.action_value[(state, a)] for a in actions]
      )
      return actions[index]

"""
Takes an env uses epsilon greedy one step sarsa to approximate the optimal
epsilon greedy policy.
alpha: [0,1]
epsilon: [0, 1] should generally be small. epsilon = 1 means totally random actions
num_ep: positive integer representing number of episodes to run.
init_value: initial action value of all unvisited action.

returns tuple containing action value function, policy, and an array containing
the episode given the time step.
"""
def epsilon_greedy_sarsa(
  env: Environment,
  alpha:float=0.1,
  epsilon:float=0.1,
  num_ep:int=100,
  init_value:float=0.0) -> Tuple:
  q = MapWithDefault(init_value)
  policy = EpsilonGreedy(env, q, epsilon=epsilon)
  step_to_episode = []
  terminal = env.terminal_state()
  for a in env.actions(terminal):
    q[(terminal, a)] = 0.0

  for i in tqdm(range(num_ep)):
    s = random.choice(env.start_states())
    a = policy[s]
    while s != terminal:
      step_to_episode.append(i)
      reward, next_s = env.next_state(s, a)
      next_a = policy[next_s]
      q[(s, a)] = q[(s,a)] + alpha*(reward + env.gamma()*q[(next_s, next_a)] - q[(s, a)])
      s, a = next_s, next_a
  return (policy, q, step_to_episode)


"""
  useful helper for ensuring the agent stays in bounds
"""
def in_dims(xy, dims):
    return (max(min(dims[0] - 1, xy[0]), 0), max(min(dims[1] - 1, xy[1]), 0))

"""
The windy grid world as described in e6.9 from RL2018
"""
class WindyGridWorldKingsMove(Environment):
  def __init__(self):
    self._actions = [(x, y) for x in (1,0,-1) for y in (1,0,-1)]
    self._actions.remove((0,0))
    self.start_state = (0,3)
    self.goal_state = (7, 3)
    self._cols_windyness = [0,0,0,1,1,1,2,2,1,0]
    self.dimensions = (10, 6)
    self._terminal_state = None
    self._gamma = 1

  """
  returns a list all possible actions at state s
  """
  def actions(self, s: Any) -> list:
    return self._actions
  
  """
    takes action a at state s returning new state s'
  """
  def next_state(self, s:Any, a: Any) -> Tuple[float, Optional[Any]]:
    if s == self.terminal_state():
      return (0, self.terminal_state())
    
    after_move = (s[0] + a[0], s[1] + a[1] + self._cols_windyness[s[0]])
    in_play_area = in_dims(after_move, self.dimensions)
    if in_play_area == self.goal_state:
      return (-1, self.terminal_state())
    return (-1, in_play_area)
    
  """
  returns whichever value is to be used as the terminal state
  """
  def terminal_state(self) -> Any:
    return self._terminal_state

  """
  returns the start states
  """
  def start_states(self) -> [Any]:
    return [self.start_state]

  def gamma(self) -> float:
    return self._gamma

class StochasticWindyGridWorldKingsMove(WindyGridWorldKingsMove):
  """
    takes action a at state s returning new state s'
  """
  def next_state(self, s:Any, a: Any) -> Tuple[float, Optional[Any]]:
    if s == self.terminal_state():
      return (0, self.terminal_state())
    
    after_move = (s[0] + a[0], s[1] + a[1] + self._cols_windyness[s[0]] + random.randint(-1, 1))
    in_play_area = in_dims(after_move, self.dimensions)
    if in_play_area == self.goal_state:
      return (-1, self.terminal_state())
    return (-1, in_play_area)

