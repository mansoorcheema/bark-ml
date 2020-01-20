import numpy as np
from bark.world.evaluation import \
  EvaluatorGoalReached, EvaluatorCollisionAgents, \
  EvaluatorCollisionDrivingCorridor, EvaluatorStepCount, EvaluatorDrivableArea
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
    self._evaluators["collision"] = \
      EvaluatorCollisionAgents()
    self._evaluators["step_count"] = EvaluatorStepCount()

  def distance_to_goal(self, world):
    d = 0.
    for idx in self._controlled_agents:
      agent = world.agents[idx]
      state = agent.state
      goal_poly = agent.goal_definition.goal_shape
      # TODO(@hart): fix.. offset 0.75 so we drive to the middle of the polygon
      d += distance(goal_poly, Point2d(state[1] + 0.75, state[2]))
    return d

  def deviation_velocity(self, world):
    desired_v = 10.
    delta_v = 0.
    for idx in self._controlled_agents:
      vel = world.agents[idx].state[int(StateDefinition.VEL_POSITION)]
      delta_v += (desired_v-vel)**2
    return delta_v
  
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
    reward = 1. - collision * self._collision_penalty + \
      success * self._goal_reward - 0.01*inpt_reward - \
      0.1*distance_to_goals + drivable_area * self._collision_penalty - \
      0.1*self.deviation_velocity(world)
    return reward

  def _evaluate(self, world, eval_results, action):
    """Returns information about the current world state
    """
    done = False
    success = eval_results["goal_reached"]
    collision = eval_results["collision"]
    drivable_area = eval_results["drivable_area"]
    step_count = eval_results["step_count"]

    reward = self.calculate_reward(world, eval_results, action)    
    if step_count > self._max_steps or collision or drivable_area or success:
      done = True
    return reward, done, eval_results
    
