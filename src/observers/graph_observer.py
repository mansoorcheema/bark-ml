
from gym import spaces
import numpy as np
from bark.models.dynamic import StateDefinition
from modules.runtime.commons.parameters import ParameterServer
import math
import operator
from src.commons.spaces import BoundedContinuous, Discrete

from src.observers.observer import StateObserver


class GraphObserver(StateObserver):
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
    # automatically create graph and verify it with the same stuff

    graph = -1.*np.ones(shape=(7, 7), dtype=np.float32)
    # connections
    graph[0, 0] = 0
    graph[0, 1] = 1

    graph[1, 0] = 0
    graph[1, 1] = 2

    graph[2, 0] = 1
    graph[2, 1] = 0

    graph[3, 0] = 1
    graph[3, 1] = 2

    graph[4, 0] = 2
    graph[4, 1] = 0

    graph[5, 0] = 2
    graph[5, 1] = 1

    agent_state_0 = self._select_state_by_index(world.agents[100].state)
    agent_state_1 = self._select_state_by_index(world.agents[101].state)
    agent_state_2 = self._select_state_by_index(world.agents[102].state)

    # node values
    graph[0, 2] = self._norm(np.cos(agent_state_0[2])*agent_state_0[3], [-1., 1.])
    graph[0, 3] = self._norm(np.sin(agent_state_0[2])*agent_state_0[3], [0., 20.])
    graph[1, 2] = self._norm(np.cos(agent_state_1[2])*agent_state_1[3], [-1., 1.])
    graph[1, 3] = self._norm(np.sin(agent_state_1[2])*agent_state_1[3], [0., 20.])
    graph[2, 2] = self._norm(np.cos(agent_state_2[2])*agent_state_2[3], [-1., 1.])
    graph[2, 3] = self._norm(np.sin(agent_state_2[2])*agent_state_2[3], [0., 20.])

    # distance to goal perserved in node values
    graph[0, 4] = self._norm(agent_state_0[0] - 5110.1, [-4., 4.])
    graph[1, 4] = self._norm(agent_state_1[0] - 5114.0, [-4., 4.])
    graph[2, 4] = self._norm(agent_state_2[0] - 5114.0, [-4., 4.])

    # edge values
    graph[0, 5] = self._norm(agent_state_1[0] - agent_state_0[0], [-4., 4.])
    graph[0, 6] = self._norm(agent_state_1[1] - agent_state_0[1], [-100., 100.])

    graph[1, 5] = self._norm(agent_state_2[0] - agent_state_0[0], [-4., 4.])
    graph[1, 6] = self._norm(agent_state_2[1] - agent_state_0[1], [-100., 100.])

    graph[2, 5] = self._norm(agent_state_0[0] - agent_state_1[0], [-4., 4.])
    graph[2, 6] = self._norm(agent_state_0[1] - agent_state_1[1], [-100., 100.])

    graph[3, 5] = self._norm(agent_state_2[0] - agent_state_1[0], [-4., 4.])
    graph[3, 6] = self._norm(agent_state_2[1] - agent_state_1[1], [-100., 100.])

    graph[4, 5] = self._norm(agent_state_0[0] - agent_state_2[0], [-4., 4.])
    graph[4, 6] = self._norm(agent_state_0[1] - agent_state_2[1], [-100., 100.])

    graph[5, 5] = self._norm(agent_state_1[0] - agent_state_2[0], [-4., 4.])
    graph[5, 6] = self._norm(agent_state_1[1] - agent_state_2[1], [-100., 100.])

    # TODO(@hart): plot graph in viewer
    return graph
  
  def _norm(self, state, range):
    return (state - range[0])/(range[1]-range[0])

  def reset(self, world, agents_to_observe):
    super(GraphObserver, self).reset(world, agents_to_observe)
    return world

  @property
  def observation_space(self):
    return spaces.Box(low=0., high=1., shape=(7, 7))

  @property
  def _len_state(self):
    return len(self._state_definition)


