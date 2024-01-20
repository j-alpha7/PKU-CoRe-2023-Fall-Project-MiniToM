from gymnasium.spaces import Discrete, Box
import numpy as np
import os
from env import Environment
from animal import Rabbit, Sheep
from model import TorchRNNModel

import ray
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.tune.logger import pretty_print
from ray.tune.registry import get_trainable_cls
from ray.rllib.algorithms.ppo import PPOConfig, PPOTorchPolicy
from ray.rllib.algorithms.a3c import A3CConfig
from ray.rllib.algorithms.a3c.a3c_torch_policy import A3CTorchPolicy
from ray.rllib.models import ModelCatalog
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--size', type=int, default=20)
parser.add_argument('--obs_mode', type=str, choices=['full', 'partial'], default='full')
parser.add_argument('--obs_radius', type=int, default=2)
parser.add_argument('--rabbit_reward', type=int, default=1)
parser.add_argument('--sheep_reward', type=int, default=3)
parser.add_argument('--max_time', type=int, default=100)
parser.add_argument('--time_penalty', type=float, default=0.0)
parser.add_argument('--record', action='store_true')
args = parser.parse_args()

torch, nn = try_import_torch()
ModelCatalog.register_custom_model("my_model", TorchRNNModel)

ray.init()

env_config = {
    'height': args.size,
    'width': args.size,
    'observe_mode': args.obs_mode, # 'full' or 'partial'
    'obs_radius': args.obs_radius, # 如果'observe_mode' == 'full'， 该参数没用 
    'animals': [Rabbit, Sheep],
    'reward': {'Rabbit': args.rabbit_reward, 'Sheep': args.sheep_reward},
    'T_max': args.max_time,
    'time_penalty': args.time_penalty,
    'record': args.record,
}

num_layer = 1 + len(env_config['animals']) + 1
if env_config['observe_mode'] == 'full':
    height, width = env_config['height'], env_config['width']
else:
    height = width = 2 * env_config['obs_radius'] + 1
obs_space = Box(low=0, high=1, shape = (num_layer, height, width))
action_space = Discrete(5)

# Can also register the env creator function explicitly with:
# register_env("corridor", lambda config: SimpleCorridor(config))

train_config = (
    PPOConfig()
    # or "corridor" if registered above
    .environment(Environment, env_config=env_config)
    #.environment(SimpleCorridor, env_config={"corridor_length": 5})
    .framework("torch")
    .rl_module( _enable_rl_module_api=False)
    .rollouts(num_rollout_workers=4, num_envs_per_worker=5)#, batch_mode = "complete_episodes")
    # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
    .resources(num_gpus=1)
    .training(lr = 1e-4,
              model = {
                  "custom_model": "my_model",
                  "custom_model_config": {'obs_shape': (num_layer, height, width)}
                #   "fcnet_hiddens": [256],
                #   "fcnet_activation": "relu",
                #   "conv_filters":[
                #       [16, [3, 3], 1],
                #       [32, [3, 3], 1],
                #   ],
                #   "conv_activation": "relu",
                #   "use_lstm": True,
                #   "max_seq_len": 20,
                #   "lstm_cell_size": 256
              },
              sgd_minibatch_size = 1024,
              num_sgd_iter = 20,
              train_batch_size = 8000,
              _enable_learner_api=False,)
)

policies = {
    "train": (PPOTorchPolicy, obs_space, action_space, train_config)
}
def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return "train"

train_config = train_config.multi_agent(
    policies = policies,
    policy_mapping_fn = policy_mapping_fn,
    policies_to_train=["train"]
    )

algo = train_config.build()
algo.restore('/root/ray_results/PPO_Environment_2023-12-23_16-09-26in8e7zw6/checkpoint_001000')

for i in range(1001):
    result = algo.train()
    result['sampler_results']['hist_stats'] = None
    print(pretty_print(result))
    if (i+1)%100 == 0:
        path = algo.save()
        print(f"Checkpoint loaded in {path}")
algo.stop()