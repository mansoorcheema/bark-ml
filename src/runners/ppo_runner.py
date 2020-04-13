import sys
import logging
import time
import numpy as np
import tensorflow as tf
tf.compat.v1.enable_v2_behavior()

from tf_agents.drivers import dynamic_step_driver
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.policies import actor_policy
from modules.runtime.commons.parameters import ParameterServer

from tf_agents.metrics import tf_metrics
from tf_agents.eval import metric_utils
from tf_agents.utils import common
from tf_agents.trajectories import time_step as ts

from src.runners.tfa_runner import TFARunner


logger = logging.getLogger()
# NOTE(@hart): this will print all statements
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

class PPORunner(TFARunner):
  """Runner that takes the runtime and agent
     and runs the training and evaluation as specified.
  """
  def __init__(self,
               runtime=None,
               eval_runtime=None,
               agent=None,
               params=ParameterServer(),
               unwrapped_runtime=None):
    TFARunner.__init__(self,
                       runtime=runtime,
                       agent=agent,
                       params=params,
                       unwrapped_runtime=unwrapped_runtime)
    self._eval_runtime = eval_runtime

  def _train(self):
    """Trains the agent as specified in the parameter file
    """
    # iterator = iter(self._agent._dataset)
    for i in range(0, self._params["ML"]["Runner"]["number_of_collections"]):
      global_iteration = self._agent._agent._train_step_counter.numpy()
      self._collection_driver.run()
      trajectories = self._agent._replay_buffer.gather_all()
      self._agent._agent.train(experience=trajectories)
      self._agent._replay_buffer.clear()
      if i % self._params["ML"]["Runner"]["evaluate_every_n_steps"] == 0:
        self.evaluate()
        self._agent.save()

  def evaluate(self, num=1):
    """Evaluates the agent
       Need to overwrite the class of the base function as the metric class somehow does
       not work.
    """
    global_iteration = self._agent._agent._train_step_counter.numpy()
    logger.info("Evaluating the agent's performance in {} episodes."
      .format(str(self._params["ML"]["Runner"]["evaluation_steps"])))
    # Ticket (https://github.com/tensorflow/agents/issues/59) recommends
    # to do the rendering in the original environment
    rewards = []
    steps = []
    if self._unwrapped_runtime is not None:
      for _ in range(0, self._params["ML"]["Runner"]["evaluation_steps"]):
        state = np.array([self._unwrapped_runtime.reset()], dtype=np.float32)
        is_terminal = False
        while not is_terminal:
          action_step = self._agent._eval_policy.action(
            ts.transition(np.array([state], dtype=np.float32), reward=0.0, discount=1.0))
          state, reward, is_terminal, _ = self._unwrapped_runtime.step(
            action_step.action.numpy())
          rewards.append(reward)
          steps.append(1)
    mean_reward = np.sum(np.array(rewards))/self._params["ML"]["Runner"]["evaluation_steps"]
    mean_steps = np.sum(np.array(steps))/self._params["ML"]["Runner"]["evaluation_steps"]
    tf.summary.scalar("mean_reward",
                      mean_reward,
                      step=global_iteration)
    tf.summary.scalar("mean_steps",
                      mean_steps,
                      step=global_iteration)
    logger.info(
      "The agent achieved average {} reward and {} steps in \
      {} episodes." \
      .format(str(mean_reward),
              str(mean_steps),
              str(self._params["ML"]["Runner"]["evaluation_steps"])))