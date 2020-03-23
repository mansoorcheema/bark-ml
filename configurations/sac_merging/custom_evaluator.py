import numpy as np
from bark.world.evaluation import \
  EvaluatorGoalReached, EvaluatorCollisionAgents, \
  EvaluatorStepCount, EvaluatorDrivableArea, EvaluatorCollisionEgoAgent
from modules.runtime.commons.parameters import ParameterServer
from bark.geometry import *
from bark.models.dynamic import StateDefinition

from src.evaluators.goal_reached import GoalReached

class CustomEvaluator(GoalReached):
  """Shows the capability of custom elements inside
     a configuration.
  """
  def __init__(self,
               params=ParameterServer(),
               eval_agent=None):
    GoalReached.__init__(self,
                         params,
                         eval_agent)

  def _add_evaluators(self):
    self._evaluators["goal_reached"] = EvaluatorGoalReached(
      self._controlled_agents[0])
    self._evaluators["drivable_area"] = EvaluatorDrivableArea()
    self._evaluators["collision"] = EvaluatorCollisionEgoAgent()
    self._evaluators["step_count"] = EvaluatorStepCount()

<<<<<<< HEAD
<<<<<<< HEAD:configurations/sac_merging/custom_evaluator.py
  def distance_to_goal(self, world):
    d = 0.
    for idx in self._controlled_agents:
      agent = world.agents[idx]
      state = agent.state
      goal_poly = agent.goal_definition.goal_shape
      # TODO(@hart): fix.. offset 0.75 so we drive to the middle of the polygon
      d += Distance(goal_poly, Point2d(state[1] + 0.75, state[2]))
    return d

  def deviation_velocity(self, world):
    desired_v = 10.
    delta_v = 0.
    for idx in self._controlled_agents:
      vel = world.agents[idx].state[int(StateDefinition.VEL_POSITION)]
      delta_v += (desired_v-vel)**2
    return delta_v
  
=======
  def distance_to_goal(self, observed_world):
=======
  def calculate_reward(self, observed_world, eval_results, action, observed_state):  # NOLINT
    success = eval_results["goal_reached"]
    collision = eval_results["collision"]
    drivable_area = eval_results["drivable_area"]

>>>>>>> 2f3d503ba54563298697c6ffcd8068183608ad74
    ego_agent = observed_world.ego_agent
    goal_def = ego_agent.goal_definition
    goal_center_line = goal_def.center_line
    ego_agent_state = ego_agent.state
    lateral_offset = Distance(goal_center_line,
                              Point2d(ego_agent_state[1], ego_agent_state[2]))
<<<<<<< HEAD
    return lateral_offset

>>>>>>> 2f3d503ba54563298697c6ffcd8068183608ad74:configurations/highway/custom_evaluator.py
  def calculate_reward(self, world, eval_results, action):
    success = eval_results["goal_reached"]
    collision = eval_results["collision"]
    drivable_area = eval_results["drivable_area"]

    distance_to_goals = self.distance_to_goal(world)
    actions = np.reshape(action, (-1, 2))
    accs = actions[:, 0]
    delta = actions[:, 1]

    # TODO(@hart): use parameter server
    inpt_reward = np.sqrt(np.sum((1/0.15*delta)**2 + (accs)**2))
<<<<<<< HEAD:configurations/sac_merging/custom_evaluator.py
    reward = 1. - collision * self._collision_penalty + \
      success * self._goal_reward - 0.01*inpt_reward - \
      0.1*distance_to_goals + drivable_area * self._collision_penalty - \
      0.1*self.deviation_velocity(world)
=======
    reward = 1. + collision * self._collision_penalty + \
      success * self._goal_reward - 0.1*inpt_reward - \
      0.01*distance_to_goals**2 + drivable_area * self._collision_penalty

>>>>>>> 2f3d503ba54563298697c6ffcd8068183608ad74:configurations/highway/custom_evaluator.py
=======

    actions = np.reshape(action, (-1, 2))
    accs = actions[:, 0]
    delta = actions[:, 1]
    # TODO(@hart): use parameter server
    inpt_reward = np.sum((4/0.15*delta)**2 + (accs)**2)
    reward = collision * self._collision_penalty + \
      success * self._goal_reward + \
      drivable_area * self._collision_penalty - \
      0.001*lateral_offset**2 + 0.001*inpt_reward
>>>>>>> 2f3d503ba54563298697c6ffcd8068183608ad74
    return reward

  def _evaluate(self, observed_world, eval_results, action, observed_state):
    """Returns information about the current world state
    """
    done = False
    success = eval_results["goal_reached"]
    collision = eval_results["collision"]
    drivable_area = eval_results["drivable_area"]
    step_count = eval_results["step_count"]
<<<<<<< HEAD

<<<<<<< HEAD:configurations/sac_merging/custom_evaluator.py
    print("Drivable area: {}, Collision: {}.".format(str(drivable_area), str(collision)))
    reward = self.calculate_reward(world, eval_results, action)    
=======
    reward = self.calculate_reward(observed_world, eval_results, action)    
>>>>>>> 2f3d503ba54563298697c6ffcd8068183608ad74:configurations/highway/custom_evaluator.py
    if step_count > self._max_steps or collision or drivable_area or success:
      done = True
    return reward, done, eval_results
    
=======
    reward = self.calculate_reward(observed_world, eval_results, action, observed_state)    
    if success or collision or step_count > self._max_steps or drivable_area:
      done = True
    return reward, done, eval_results
    
>>>>>>> 2f3d503ba54563298697c6ffcd8068183608ad74
