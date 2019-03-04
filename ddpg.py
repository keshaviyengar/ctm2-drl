import gym
from gym.wrappers import FlattenDictWrapper
import ctm2_envs
import numpy as np

from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.bench import Monitor
from stable_baselines.common.cmd_util import make_robotics_env
from stable_baselines.results_plotter import load_results, ts2xy

from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG

from numpy import inf
import yaml
import os

# Load in parameters
config = yaml.load(open('ddpg_parameters.yaml'))
ddpg_params = config['ddpg']
os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu_id']

# Create and wrap the environment
env = make_robotics_env(env_id='ctm2-continuous-v0', seed=1, allow_early_resets=True)
# the noise objects for DDPG
n_actions = env.action_space.shape[-1]

param_noise = AdaptiveParamNoiseSpec(ddpg_params['param_noise']['initial_stddev'],
                                     ddpg_params['param_noise']['desired_action_stddev'],
                                     ddpg_params['param_noise']['adoption_coefficient'])

action_noise = NormalActionNoise(mean=env.action_space.high / 2 + env.action_space.low / 2,
                                 sigma= env.action_space.high / 4 - env.action_space.low / 4)
model = DDPG(MlpPolicy, env, gamma=ddpg_params['gamma'], memory_policy=ddpg_params['memory_policy'],
             eval_env=None, nb_train_steps=ddpg_params['nb_train_steps'],
             nb_rollout_steps=ddpg_params['nb_rollout_steps'],
             nb_eval_steps=ddpg_params['nb_eval_steps'], param_noise=param_noise, action_noise=action_noise,
             normalize_observations=ddpg_params['normalize_observations'], tau=ddpg_params['tau'],
             batch_size=ddpg_params['batch_size'],
             param_noise_adaption_interval=ddpg_params['param_noise_adaption_interval'],
             normalize_returns=ddpg_params['normalize_returns'], enable_popart=ddpg_params['enable_popart'],
             observation_range=(-5.0, 5.0),
             critic_l2_reg=ddpg_params['critic_l2_reg'], return_range=(-inf, inf), actor_lr=ddpg_params['actor_lr'],
             critic_lr=ddpg_params['critic_lr'], clip_norm=ddpg_params['clip_norm'],
             reward_scale=ddpg_params['reward_scale'],
             render=ddpg_params['render'], render_eval=ddpg_params['render_eval'],
             memory_limit=ddpg_params['memory_limit'], verbose=ddpg_params['verbose'],
             tensorboard_log=ddpg_params['tensorboard_log'], _init_setup_model=ddpg_params['_init_setup_model'],
             policy_kwargs=ddpg_params['policy_kwargs'])

model.learn(total_timesteps=ddpg_params['total_timesteps'], log_interval=100)
