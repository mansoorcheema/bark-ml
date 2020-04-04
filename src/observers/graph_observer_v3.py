import math
from gym import spaces
import numpy as np
from bark.models.dynamic import StateDefinition
from bark.geometry import Line2d,Point2d, SignedDistance
from modules.runtime.commons.parameters import ParameterServer
import math
import operator
from src.commons.py_spaces import BoundedContinuous, Discrete

from src.observers.observer import StateObserver


class GraphObserverV3(StateObserver):
  def __init__(self,
               max_num_vehicles=5,
               num_nearest_vehicles=4,
               params=ParameterServer(),
               viewer=None):
    StateObserver.__init__(self, params)
    self._state_definition = [int(StateDefinition.X_POSITION),
                              int(StateDefinition.Y_POSITION),
                              int(StateDefinition.THETA_POSITION),
                              int(StateDefinition.VEL_POSITION)]
    self._h0_len = 4
    self._e0_len = 2
    self._num_nearest_vehicles = num_nearest_vehicles
    self._max_num_vehicles = max_num_vehicles
    self._num_graph_rows = 25 #math.pow(num_nearest_vehicles, max_num_vehicles) + 1
    self._observation_len = self._max_num_vehicles*self._len_state
    self._initial_lane_corr = None
    self._viewer = viewer

  def FindNearestAgentIds(self, world, agent_id, num_nearest=5):
    agent_state = world.GetAgent(agent_id).state
    agent_point = Point2d(
      agent_state[int(StateDefinition.X_POSITION)],
      agent_state[int(StateDefinition.Y_POSITION)])
    nearest_agents = world.GetNearestAgents(
      agent_point, num_nearest)
    agent_id_list = []
    for nid in nearest_agents:
      # TODO(@hart): evaluate self-connections
      if nid not in agent_id_list and nid != agent_id:
        agent_id_list.append(nid)
    return agent_id_list

  def CalculateNodeValue(self, observed_world, agent_id):
    agent = observed_world.GetAgent(agent_id)
    reduced_state = self._select_state_by_index(agent.state)
    vx = np.cos(reduced_state[2])*reduced_state[3]
    vy = np.sin(reduced_state[2])*reduced_state[3]
    x = self._norm(reduced_state[0], self._world_x_range)
    y = self._norm(reduced_state[1], self._world_y_range)
    # d_goal = 0.
    # if agent.road_corridor:
    #   goal_def = agent.goal_definition
    #   goal_center_line = goal_def.center_line
    #   ego_agent_state = agent.state
    #   d_goal = SignedDistance(
    #     goal_center_line,
    #     Point2d(ego_agent_state[1], ego_agent_state[2]),
    #     reduced_state[3])
    n_vx = self._norm(vx, [-8., 8.])
    n_vy = self._norm(vy, [0., 20.])
    # n_d_goal = self._norm(d_goal, [-8., 8.])
    # print(agent_id, [n_vx, n_vy, n_d_goal])
    # TODO(@hart): HACK; remove
    return np.array([x, y, n_vx, n_vy], dtype=np.float32)

  def CalculateEdgeValue(self, observed_world, from_id, to_id):
    from_agent = observed_world.agents[from_id]
    to_agent = observed_world.agents[to_id]
    reduced_from_state = self._select_state_by_index(from_agent.state)
    reduced_to_state = self._select_state_by_index(to_agent.state)
    dx = reduced_to_state[0] - reduced_from_state[0]
    dy = reduced_to_state[1] - reduced_from_state[1]
    # plot connections
    if self._viewer is not None:
      pt_from = Point2d(reduced_from_state[0], reduced_from_state[1])
      pt_to = Point2d(reduced_to_state[0], reduced_to_state[1])
      line = Line2d()
      line.AddPoint(pt_from)
      line.AddPoint(pt_to)
      color = "gray"
      alpha = 0.25
      if to_id == observed_world.ego_agent.id:
        color = "red"
        alpha = 1.0
      self._viewer.drawLine2d(line, color=color, alpha=alpha)
    # print("from_id: ", from_id, ", to_id:", to_id, ", dx: ", dx, ", dy: ", dy)
    n_dx = self._norm(dx, [-8., 8.])
    n_dy = self._norm(dy, [-250., 250.])
    return np.array([0, 0], dtype=np.float32)

  def observe(self, observed_world):
    """see base class
    """
    if self._viewer is not None:
      self._viewer.clear()
    gen_graph = -1.*np.ones(
      shape=(int(self._num_graph_rows), 8),
      dtype=np.float32)
    id_list = self.FindNearestAgentIds(
      observed_world, observed_world.ego_agent.id, self._max_num_vehicles)
    # we need to append the ego agent first!
    if observed_world.ego_agent.id in id_list:
      id_list.remove(observed_world.ego_agent.id)
    id_list.insert(0, observed_world.ego_agent.id)
    assert(id_list[0] == observed_world.ego_agent.id)

    node_row_idx = edge_row_idx= 0
    # 2. loop through all agent
    for agent_id in id_list:
      # 3. add nodes
      gen_graph[node_row_idx, 2:2+self._h0_len] = \
        self.CalculateNodeValue(observed_world, agent_id)
      # we only want to add edges for the ego and nearby agents
      #if agent_id in nearest_agent_ids:
      if agent_id == observed_world.ego_agent.id:
        nearest_ids = self.FindNearestAgentIds(
          observed_world, agent_id, self._num_nearest_vehicles)
        nearest_ids_ = []
        for lid in id_list:
          if lid in nearest_ids:
            nearest_ids_.append(lid)
        for from_id in nearest_ids_:
          # print(edge_row_idx, from_id, agent_id)
          gen_graph[edge_row_idx, :2] = \
            np.array([id_list.index(from_id),
                      id_list.index(agent_id)], dtype=np.float32)
          gen_graph[edge_row_idx, self._h0_len+2:] = \
            self.CalculateEdgeValue(observed_world, from_id, agent_id)
          edge_row_idx += 1
      node_row_idx += 1
    return gen_graph
  
  def _norm(self, state, range):
    return (state - range[0])/(range[1]-range[0])

  def reset(self, world):
    world = super().reset(world)
    return world

  @property
  def observation_space(self):
    return spaces.Box(
      low=0.,
      high=1.,
      shape=(int(self._num_graph_rows), 8))

  @property
  def _len_state(self):
    return len(self._state_definition)