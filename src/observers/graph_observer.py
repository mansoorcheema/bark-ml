
from gym import spaces
import numpy as np
from bark.models.dynamic import StateDefinition
from modules.runtime.commons.parameters import ParameterServer
import math
import operator
from src.commons.spaces import BoundedContinuous, Discrete

from src.observers.observer import StateObserver


class SimpleObserver(StateObserver):
  def __init__(self,
               params=ParameterServer()):
    StateObserver.__init__(self, params)
    self._state_definition = [int(StateDefinition.X_POSITION),
                              int(StateDefinition.Y_POSITION),
                              int(StateDefinition.THETA_POSITION),
                              int(StateDefinition.VEL_POSITION)]
    self._observation_len = \
      self._max_num_vehicles*self._len_state

  def observe(self, world, agents_to_observe):
    """see base class
    """
    graph = -1.*np.ones(shape=(3, 6), dtype=np.float32)
    # connections
    graph[0, 0] = 0
    graph[0, 1] = 1
    graph[1, 0] = 1
    graph[1, 1] = 0
    agent_state_0 = self._select_state_by_index(world.agents[100].state)
    agent_state_1 = self._select_state_by_index(world.agents[101].state)
    # node values
    graph[0, 2] = np.cos(agent_state_0[2])*agent_state_0[3]
    graph[0, 3] = np.sin(agent_state_0[2])*agent_state_0[3]
    graph[1, 2] = np.cos(agent_state_1[2])*agent_state_1[3]
    graph[1, 3] = np.sin(agent_state_1[2])*agent_state_1[3]
    # right, left needs to match
    graph[0, 3] = agent_state_0[0] - 5114
    graph[1, 3] = agent_state_1[0] - 5110.1
    # edge values
    graph[0, 3] = agent_state_1[0] - agent_state_0[0]
    graph[0, 4] = agent_state_1[1] - agent_state_0[1]
    graph[1, 4] = agent_state_0[0] - agent_state_1[0]
    graph[1, 4] = agent_state_0[1] - agent_state_1[1]
    return graph

  def reset(self, world, agents_to_observe):
    super(SimpleObserver, self).reset(world, agents_to_observe)
    return world

  @property
  def observation_space(self):
    return spaces.Box(low=0., high=1., shape=(3, 6))

  @property
  def _len_state(self):
    return len(self._state_definition)


