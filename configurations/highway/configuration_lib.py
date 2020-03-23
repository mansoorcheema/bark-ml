from absl import app
from absl import flags
import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.environments import parallel_py_environment

from modules.runtime.scenario.scenario_generation.uniform_vehicle_distribution \
  import UniformVehicleDistribution
from modules.runtime.scenario.scenario_generation.deterministic \
  import DeterministicScenarioGeneration
from modules.runtime.scenario.scenario_generation.configurable_scenario_generation import \
  ConfigurableScenarioGeneration

from modules.runtime.commons.parameters import ParameterServer
from modules.runtime.viewer.matplotlib_viewer import MPViewer
from modules.runtime.viewer.video_renderer import VideoRenderer
from modules.runtime.viewer.pygame_viewer import PygameViewer

from src.rl_runtime import RuntimeRL
from src.observers.nearest_state_observer import ClosestAgentsObserver
from src.observers.graph_observer import GraphObserver
from src.observers.graph_observer_v2 import GraphObserverV2
from src.wrappers.dynamic_model import DynamicModel
from src.wrappers.tfa_wrapper import TFAWrapper
from src.evaluators.goal_reached import GoalReached
# from src.agents.sac_agent import SACAgent
from src.agents.ppo_agent import PPOAgent
from src.agents.ppo_agent_gnn import PPOAgentGNN
from src.agents.sac_agent_gnn import SACAgentGNN
# from src.runners.sac_runner import SACRunner
from src.runners.ppo_runner import PPORunner
# from src.agents.sac_agent_graph import SACGraphAgent
from configurations.base_configuration import BaseConfiguration

# configuration specific evaluator
from configurations.highway.custom_evaluator import CustomEvaluator
# from bark_ml.observers import NearestObserver

class HighwayConfiguration(BaseConfiguration):
  """Hermetic and reproducible configuration class
  """
  def __init__(self,
               params):
    BaseConfiguration.__init__(
      self,
      params)

  def _build_configuration(self):
    """Builds a configuration using an agent
    """
    # self._scenario_generator = \
    #   DeterministicScenarioGeneration(num_scenarios=250,
    #                                   random_seed=0,
    #                                   params=self._params)
    self._scenario_generator = \
      ConfigurableScenarioGeneration(num_scenarios=100,
                                     params=self._params)
    self._behavior_model = DynamicModel(params=self._params)
    self._evaluator = CustomEvaluator(params=self._params)
    self._viewer  = MPViewer(params=self._params,
                             # use_world_bounds=True)
                             x_range=[-40, 40],
                             y_range=[-40, 40],
                             follow_agent_id=True)
    # self._viewer = VideoRenderer(renderer=self._viewer, world_step_time=0.2)
    if self._params["type"] == "graph":
      self._observer = GraphObserverV2(params=self._params,
                                       max_num_vehicles=5,
                                       viewer=self._viewer)
    else:
      self._observer = ClosestAgentsObserver(params=self._params)
    self._runtime = RuntimeRL(action_wrapper=self._behavior_model,
                              observer=self._observer,
                              evaluator=self._evaluator,
                              step_time=0.2,
                              viewer=self._viewer,
                              scenario_generator=self._scenario_generator)
    eval_tf_env = tf_py_environment.TFPyEnvironment(TFAWrapper(self._runtime))
    tfa_env = tf_py_environment.TFPyEnvironment(TFAWrapper(self._runtime))

    if self._params["type"] == "graph":
      self._agent = PPOAgentGNN(tfa_env, params=self._params)
      # self._agent = SACAgentGNN(tfa_env, params=self._params)
    else:
      self._agent = PPOAgent(tfa_env, params=self._params)

    self._runner = PPORunner(tfa_env,
                             eval_tf_env,
                             self._agent,
                             params=self._params,
                             unwrapped_runtime=self._runtime)
