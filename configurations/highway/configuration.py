import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
if os.environ.get('DISPLAY') == ':0':
  print('No display found. Using non-interactive Agg backend')
  mpl.use('Agg')
import logging
from absl import app
from absl import flags
import tensorflow as tf
from tf_agents.environments import tf_py_environment

from modules.runtime.commons.parameters import ParameterServer
from configurations.base_configuration import BaseConfiguration
from configurations.highway.configuration_lib import HighwayConfiguration
from tf_agents.trajectories import time_step as ts

# configuration specific evaluator
from configurations.highway.custom_evaluator import CustomEvaluator
from configurations.highway.configuration_lib import HighwayConfiguration

FLAGS = flags.FLAGS
flags.DEFINE_enum('mode',
                  'visualize',
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
  params = ParameterServer(filename=FLAGS.base_dir + "/configurations/highway/config.json")
  params["BaseDir"] = FLAGS.base_dir
  params["type"] = FLAGS.type
  configuration = HighwayConfiguration(params)
  
  if FLAGS.mode == 'train':
    configuration._runner.setup_writer()
    configuration.train()
  elif FLAGS.mode == 'visualize':
    params["ML"]["Agent"]["num_parallel_environments"] = 1
    configuration.visualize(10)
    # configuration._viewer.export_video("/home/hart/Dokumente/2020/bark-ml/configurations/highway/video/lane_change_3")
  elif FLAGS.mode == 'evaluate':
    configuration.evaluate(1000)
    print(configuration._runtime._collision_count/1000)
    print(configuration._runtime._success_count/1000)
  elif FLAGS.mode == 'ablation':
    # caution: use 5 vehicles
    eval_policy = configuration._agent._agent.policy
    scenario_generator = configuration._scenario_generator
    observer = configuration._observer
    viewer = configuration._viewer
    scenario, _ = \
      scenario_generator.get_next_scenario()
    world = scenario.get_world_state()
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('distance of lead vehicle (m)')
    ax1.set_ylabel('acceleration $a$', color=color)
    # ax1.plot(t, data1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('steering-rate $\delta$', color=color)  # we already handled the x-label with ax1
    # ax2.plot(t, data2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    deltas = []
    accs = []
    distances = []
    agent_state = world.agents[104].state
    start_distance = agent_state[2]
    viewer.drawWorld(world)
    for _ in range(0, 3):
      agent_state = world.agents[104].state
      agent_state[2] += 5.
      distances.append(agent_state[2]-start_distance)
      world.agents[104].SetState(agent_state)
      world.UpdateAgentRTree()
      observed_state = observer.observe(world, agents_to_observe=[100])
      time_step = ts.transition(observed_state, reward=0.0, discount=1.0)
      action = eval_policy.action(time_step)
      for agent_id, agent in world.agents.items():
        viewer.drawAgent(agent, color="gray")
      print(action)
      deltas.append(action.action[1])
      accs.append(action.action[0])
      # viewer.show(block=False)
      # plt.pause(1.)
    np_deltas = np.array(deltas)
    np_accs = np.array(accs)
    np_distances = np.array(distances)
    ax2.plot(np_distances, np_deltas, color="blue")
    ax1.plot(np_distances, np_accs, color="red")
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
    #viewer.show(block=True)
  

if __name__ == '__main__':
  app.run(run_configuration)