import os
import matplotlib as mpl
import matplotlib.pyplot as plt
if os.environ.get('DISPLAY') == ':0':
  print('No display found. Using non-interactive Agg backend')
  mpl.use('Agg')

from absl import app
from absl import flags
import tensorflow as tf
from tf_agents.environments import tf_py_environment

from modules.runtime.commons.parameters import ParameterServer
from configurations.base_configuration import BaseConfiguration
from configurations.highway.configuration_lib import HighwayConfiguration
from tf_agents.trajectories import time_step as ts

FLAGS = flags.FLAGS
flags.DEFINE_enum('mode',
                  'train',
                  ['train', 'visualize', 'evaluate', 'ablation'],
                  'Mode the configuration should be executed in.')
flags.DEFINE_string('base_dir',
                    os.path.dirname(
                      os.path.dirname(os.path.dirname(__file__))),
                    'Base directory of bark-ml.')
flags.DEFINE_enum('type',
                  'graph',
                  ['graph', 'normal'],
                  'Mode the configuration should be executed in.')


def run_configuration(argv):
  params = ParameterServer(
    filename=FLAGS.base_dir + "/configurations/highway/config_three.json")
  scenario_generation = params["Scenario"]["Generation"]["DeterministicScenarioGeneration"]  # NOLINT
  map_filename = scenario_generation["MapFilename"]
  scenario_generation["MapFilename"] = FLAGS.base_dir + "/" + map_filename
  params["BaseDir"] = FLAGS.base_dir
  params["type"] = FLAGS.type
  configuration = HighwayConfiguration(params)
  
  if FLAGS.mode == 'train':
    configuration._runner.setup_writer()
    configuration.train()
  elif FLAGS.mode == 'visualize':
    params["ML"]["Agent"]["num_parallel_environments"] = 1
    configuration.visualize(10)
  elif FLAGS.mode == 'evaluate':
    configuration.evaluate()
  elif FLAGS.mode == 'ablation':
    # caution: use 5 vehicles
    eval_policy = configuration._agent._agent.policy
    scenario_generator = configuration._scenario_generator
    observer = configuration._observer
    viewer = configuration._viewer
    scenario, _ = \
      scenario_generator.get_next_scenario()
    world = scenario.get_world_state()

    for _ in range(0, 10):
      agent_state = world.agents[102].state
      agent_state[2] += 2.
      world.agents[102].SetState(agent_state)
      world.UpdateAgentRTree()
      observed_state = observer.observe(world, agents_to_observe=[100])
      time_step = ts.transition(observed_state, reward=0.0, discount=1.0)
      action = eval_policy.action(time_step)
      print(action)
      viewer.drawWorld(world)
      viewer.show(block=False)
      plt.pause(2.)
    viewer.show(block=True)
  



if __name__ == '__main__':
  app.run(run_configuration)