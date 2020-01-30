import os
import matplotlib as mpl
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

FLAGS = flags.FLAGS
flags.DEFINE_enum('mode',
                  'train',
                  ['train', 'visualize', 'evaluate'],
                  'Mode the configuration should be executed in.')
flags.DEFINE_string('base_dir',
                    os.path.dirname(
                      os.path.dirname(os.path.dirname(__file__))),
                    'Base directory of bark-ml.')

def run_configuration(argv):
  params = ParameterServer(
    filename=FLAGS.base_dir + "/configurations/highway/config.json")
  scenario_generation = params["Scenario"]["Generation"]["DeterministicScenarioGeneration"]  # NOLINT
  map_filename = scenario_generation["MapFilename"]
  scenario_generation["MapFilename"] = FLAGS.base_dir + "/" + map_filename
  params["BaseDir"] = FLAGS.base_dir
  configuration = HighwayConfiguration(params)
  
  if FLAGS.mode == 'train':
    configuration._runner.setup_writer()
    configuration.train()
  elif FLAGS.mode == 'visualize':
    params["ML"]["Agent"]["num_parallel_environments"] = 1
    configuration.visualize(10)
  elif FLAGS.mode == 'evaluate':
    configuration.evaluate()

if __name__ == '__main__':
  app.run(run_configuration)