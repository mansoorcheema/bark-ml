
from gym import spaces
import numpy as np
from bark.models.dynamic import StateDefinition
from src.commons.spaces import BoundedContinuous, Discrete
from modules.runtime.commons.parameters import ParameterServer
import math
import operator

from src.observers.observer import StateObserver


class ClosestAgentsObserver(StateObserver):
  def __init__(self, params=ParameterServer()):
    StateObserver.__init__(self, params)
    self._state_definition = [int(StateDefinition.X_POSITION),
                              int(StateDefinition.Y_POSITION),
                              int(StateDefinition.THETA_POSITION),
                              int(StateDefinition.VEL_POSITION)]
    self._velocity_range = \
      self._params["Runtime"]["RL"]["ClosestAgentsObserver"]["VelocityRange",
      "Boundaries for min and max velocity for normalization",
      [0, 100]]
    self._theta_range = \
      self._params["Runtime"]["RL"]["ClosestAgentsObserver"]["ThetaRange",
      "Boundaries for min and max theta for normalization",
      [0, 2*math.pi]]
    self._normalize = \
      self._params["Runtime"]["RL"]["ClosestAgentsObserver"]["Normalize",
      "Whether normalization should be performed",
      True]
    self._max_num_other_agents = \
      self._params["Runtime"]["RL"]["ClosestAgentsObserver"]["MaxOtherAgents",
      "The concatenation state size is the ego agent plus max num other agents",
      4]
    self._max_distance_other_agents = \
      self._params["Runtime"]["RL"]["ClosestAgentsObserver"]["MaxOtherDistance",
      "Agents further than this distance are not observed; if not max" + \
      "other agents are seen, remaining concatenation state is set to zero",
      30]

  def observe(self, world, agents_to_observe):
    """see base class
    """
    super(ClosestAgentsObserver, self).observe(
      world=world,
      agents_to_observe=agents_to_observe)
    observed_worlds =  world.observe(agents_to_observe)
    if (len(observed_worlds) == 0):
      concatenated_state = np.zeros(self._len_ego_state + \
        self._max_num_other_agents*self._len_relative_agent_state)
      return concatenated_state.fill(np.nan)
    ego_observed_world = observed_worlds[0]
    num_other_agents = len(ego_observed_world.other_agents)
    ego_state = ego_observed_world.ego_agent.state

    # calculate nearest agent distances
    nearest_distances = {}
    for agent_id, agent in ego_observed_world.other_agents.items():
      if agent_id == agents_to_observe[0]:
        continue
      dx = ego_state[int(StateDefinition.X_POSITION)] - \
        agent.state[int(StateDefinition.X_POSITION)]
      dy = ego_state[int(StateDefinition.Y_POSITION)] - \
        agent.state[int(StateDefinition.Y_POSITION)]
      dist =  dx**2 + dy**2
      nearest_distances[dist] = agent_id

    # preallocate np.array and add ego state
    concatenated_state = np.zeros(self._len_ego_state + \
      self._max_num_other_agents*self._len_relative_agent_state)
    concatenated_state[0:self._len_ego_state] = \
      self._select_state_by_index(self._norm(ego_state)) 
    
    # add max number of agents to state concatenation vector
    concat_pos = self._len_relative_agent_state
    nearest_distances = sorted(nearest_distances.items(),
                               key=operator.itemgetter(0))
    for agent_idx in range(0, self._max_num_other_agents):
      if agent_idx<len(nearest_distances) and \
        nearest_distances[agent_idx][0] <= self._max_distance_other_agents**2:
        agent_id = nearest_distances[agent_idx][1]
        agent = ego_observed_world.other_agents[agent_id]
        agent_rel_state = self._select_state_by_index(
          self._calculate_relative_agent_state(ego_state,
                                               self._norm(agent.state)))
        concatenated_state[concat_pos:concat_pos + \
          self._len_relative_agent_state] = agent_rel_state
      else:
        concatenated_state[concat_pos:concat_pos + \
          self._len_relative_agent_state] = \
            np.zeros(self._len_relative_agent_state)
      concat_pos += self._len_relative_agent_state
    return concatenated_state

  @property
  def observation_space(self):
    # TODO(@hart): use from spaces.py
    return spaces.Box(
      low=np.zeros(self._len_ego_state + \
        self._max_num_other_agents*self._len_relative_agent_state),
      high = np.ones(self._len_ego_state + \
        self._max_num_other_agents*self._len_relative_agent_state))

  def _norm(self, agent_state):
    if not self._normalize:
        return agent_state
    agent_state[int(StateDefinition.X_POSITION)] = \
      self._norm_to_range(agent_state[int(StateDefinition.X_POSITION)],
                          self._world_x_range)
    agent_state[int(StateDefinition.Y_POSITION)] = \
      self._norm_to_range(agent_state[int(StateDefinition.Y_POSITION)],
                          self._world_y_range)
    agent_state[int(StateDefinition.THETA_POSITION)] = \
      self._norm_to_range(agent_state[int(StateDefinition.THETA_POSITION)],
                          self._theta_range)
    agent_state[int(StateDefinition.VEL_POSITION)] = \
      self._norm_to_range(agent_state[int(StateDefinition.VEL_POSITION)],
                          self._velocity_range)
    return agent_state

  def _norm_to_range(self, value, range):
    return (value - range[0])/(range[1]-range[0])

  def _calculate_relative_agent_state(self, ego_agent_state, agent_state):
    return agent_state

  @property
  def _len_relative_agent_state(self):
    return len(self._state_definition)

  @property
  def _len_ego_state(self):
    return len(self._state_definition)