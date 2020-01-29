
from gym import spaces
import numpy as np
from bark.models.dynamic import StateDefinition
from bark.geometry import Point2d, SignedDistance
from modules.runtime.commons.parameters import ParameterServer
import math
import operator
from src.commons.spaces import BoundedContinuous, Discrete

from src.observers.observer import StateObserver


class GraphObserverV2(StateObserver):
  def __init__(self,
               max_num_vehicles=3,
               num_nearest_vehicles=2,
               params=ParameterServer()):
    StateObserver.__init__(self, params)
    self._state_definition = [int(StateDefinition.X_POSITION),
                              int(StateDefinition.Y_POSITION),
                              int(StateDefinition.THETA_POSITION),
                              int(StateDefinition.VEL_POSITION)]
    self._h0_len = 3
    self._e0_len = 2
    self._num_nearest_vehicles = num_nearest_vehicles
    self._max_num_vehicles = max_num_vehicles
    self._observation_len = self._max_num_vehicles*self._len_state
    self._initial_lane_corr = None


  def OrderedAgentIds(self, world, agents_to_observe):
    agent_id_list = []
    for oid in agents_to_observe:
      agent_id_list.append(oid)
    for agent_id, agent in world.agents.items():
      if agent_id not in agents_to_observe:
        agent_id_list.append(agent_id)
    return agent_id_list

  def FindNearestAgentIds(self, world, agent_id):
    agent_state = world.GetAgent(agent_id).state
    agent_point = Point2d(
      agent_state[int(StateDefinition.X_POSITION)],
      agent_state[int(StateDefinition.Y_POSITION)])
    nearest_agents = world.GetNearestAgents(
      agent_point, self._num_nearest_vehicles + 1)
    agent_id_list = []
    for nid in nearest_agents:
      if nid not in agent_id_list and nid != agent_id:
        agent_id_list.append(nid)
    return agent_id_list

  def CalculateNodeValue(self, world, agent_id):
    agent = world.GetAgent(agent_id)
    reduced_state = self._select_state_by_index(agent.state)
    vx = np.cos(reduced_state[2])*reduced_state[3]
    vy = np.sin(reduced_state[2])*reduced_state[3]
    d_goal = 0.
    if agent.road_corridor:
      goal_lane_corr = agent.road_corridor.lane_corridors[0]
      center_line = goal_lane_corr.center_line
      d_goal = SignedDistance(
        center_line,
        Point2d(reduced_state[0], reduced_state[1]),
        reduced_state[2])
    # print("distance to goal:", d_goal)
    # print("agent_id", vx, vy, d_goal)
    n_vx = self._norm(vx, [-8., 8.])
    n_vy = self._norm(vy, [0., 20.])
    n_d_goal = self._norm(d_goal, [-4., 4.])
    return np.array([n_vx, n_vy, n_d_goal], dtype=np.float32)

  def CalculateEdgeValue(self, world, from_id, to_id):
    from_agent = world.agents[from_id]
    to_agent = world.agents[to_id]
    reduced_from_state = self._select_state_by_index(from_agent.state)
    reduced_to_state = self._select_state_by_index(to_agent.state)
    dx = reduced_to_state[0] - reduced_from_state[0]
    dy = reduced_to_state[1] - reduced_from_state[1]
    # print("from_id: ", from_id, ", to_id:", to_id, ", dx: ", dx, ", dy: ", dy)
    n_dx = self._norm(dx, [-4., 4.])
    n_dy = self._norm(dy, [-100., 100.])
    return np.array([n_dx, n_dy], dtype=np.float32)

  def observe(self, world, agents_to_observe):
    """see base class
    """
    gen_graph = -1.*np.ones(
      shape=(self._max_num_vehicles*self._num_nearest_vehicles + 1, 7),
      dtype=np.float32)
    # 1. make sure ego agent is in front
    id_list = self.OrderedAgentIds(world, agents_to_observe)
    assert(id_list[0] == agents_to_observe[0])
    node_row_idx = edge_row_idx= 0
    # 2. loop through all agent
    for agent_id in id_list:
      # 3. add nodes
      gen_graph[node_row_idx, 2:2+self._h0_len] = \
        self.CalculateNodeValue(world, agent_id)
      nearest_ids = self.FindNearestAgentIds(world, agent_id)
      # print(node_value, nearest_ids)
      for from_id in nearest_ids:
        gen_graph[edge_row_idx, :2] = \
          np.array([id_list.index(from_id),
                    id_list.index(agent_id)], dtype=np.float32)
        gen_graph[edge_row_idx, self._h0_len+2:] = \
          self.CalculateEdgeValue(world, from_id, agent_id)
        edge_row_idx += 1
      node_row_idx += 1
    return gen_graph
    
    # return graph
  
  def _norm(self, state, range):
    return (state - range[0])/(range[1]-range[0])

  def reset(self, world, agents_to_observe):
    super(GraphObserverV2, self).reset(world, agents_to_observe)
    return world

  @property
  def observation_space(self):
    return spaces.Box(
      low=-1.,
      high=5.,
      shape=(7, 7))

  @property
  def _len_state(self):
    return len(self._state_definition)


