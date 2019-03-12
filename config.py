import gym
import ctm2_envs

from stable_baselines.common.cmd_util import make_robotics_env
from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec

import numpy as np


# sigma is how many standard deviations action_space.high is from the mean
# increasing sigma will result in less samples closer to the limit
def get_action_noise(env, mean, sigma):
    action_mean = np.mean([env.action_space.low, env.action_space.high], axis=0)
    action_sigma = (env.action_space.high - action_mean) / sigma
    return NormalActionNoise(mean=action_mean, sigma=action_sigma)


def get_param_noise(initial_stddev, desired_action_stddev, adoption_coefficient):
    return AdaptiveParamNoiseSpec(initial_stddev, desired_action_stddev, adoption_coefficient)


global_configuration = {
    "single_gpu": False,
    "gpu_id": '0',
    "seed": 1
}

experiment1_configuration = {
    "env": make_robotics_env(env_id="ctm2-continuous-dense-v0", seed=1, allow_early_resets=True),
    "experiment_id": 1,
    "save_model_name": "ddpg_1",
    "gamma": 0.99,
    "memory_policy": None,
    "eval_env": make_robotics_env(env_id="ctm2-continuous-dense-v0", seed=1, allow_early_resets=True),
    "nb_train_steps": 300,
    "nb_rollout_steps": 250,
    "nb_eval_steps": 250,
    "action_noise_mean": 0,
    "action_noise_sigma": 1,
    "tau": 0.001,
    "batch_size": 128,
    "param_noise_initial_stddev": 1.0,
    "param_noise_desired_action_stddev": 0.0,
    "param_noise_adaption_coefficient": 0.0,
    "param_noise_adaption_interval": 5000,
    "normalize_returns": False,
    "enable_popart": False,
    "normalize_observations": False,
    "observation_range": (-5.0, 5.0),
    "critic_l2_reg": 0.0,
    "return_range": (-100, 100),
    "actor_lr": 0.0001,
    "critic_lr": 0.001,
    "clip_norm": None,
    "reward_scale": 0.0,
    "render": False,
    "render_eval": False,
    "memory_limit": 100,
    "verbose": 1,
    "tensorboard_log": 'ctm2_tb/',
    "_init_setup_model": True,
    "policy_kwargs": None,
    "full_tensorboard_log": True,
    "total_timesteps": 1000000,
    "log_interval": 10
}

experiment2_configuration = {
    "env": make_robotics_env(env_id="ctm2-continuous-sparse-v0", seed=1, allow_early_resets=True),
    "experiment_id": 2,
    "save_model_name": "ddpg_1",
    "gamma": 0.99,
    "memory_policy": None,
    "eval_env": make_robotics_env(env_id="ctm2-continuous-sparse-v0", seed=1, allow_early_resets=True),
    "nb_train_steps": 300,
    "nb_rollout_steps": 250,
    "nb_eval_steps": 250,
    "action_noise_mean": 0,
    "action_noise_sigma": 1,
    "tau": 0.001,
    "batch_size": 128,
    "param_noise_initial_stddev": 1.0,
    "param_noise_desired_action_stddev": 0.0,
    "param_noise_adaption_coefficient": 0.0,
    "param_noise_adaption_interval": 5000,
    "normalize_returns": False,
    "enable_popart": False,
    "normalize_observations": False,
    "observation_range": (-5.0, 5.0),
    "critic_l2_reg": 0.0,
    "return_range": (-100, 100),
    "actor_lr": 0.0001,
    "critic_lr": 0.001,
    "clip_norm": None,
    "reward_scale": 0.0,
    "render": False,
    "render_eval": False,
    "memory_limit": 100,
    "verbose": 1,
    "tensorboard_log": 'ctm2_tb/',
    "_init_setup_model": True,
    "policy_kwargs": None,
    "full_tensorboard_log": True,
    "total_timesteps": 1000000,
    "log_interval": 10
}

experiment1 = {
    "algorithm": "DDPG",
    "policy": MlpPolicy,
    "env": experiment1_configuration['env'],
    "action_noise": get_action_noise(experiment1_configuration['env'],
                                     experiment1_configuration['action_noise_mean'],
                                     experiment1_configuration['action_noise_sigma']),
    "param_noise": get_param_noise(experiment1_configuration['param_noise_initial_stddev'],
                                   experiment1_configuration['param_noise_desired_action_stddev'],
                                   experiment1_configuration['param_noise_adaption_interval']),
    "eval_env": experiment1_configuration['eval_env'],
    "memory_policy": None,
    "configuration": experiment1_configuration
}

experiment2 = {
    "algorithm": "DDPG",
    "policy": MlpPolicy,
    "env": experiment1_configuration['env'],
    "action_noise": get_action_noise(experiment1_configuration['env'],
                                     experiment1_configuration['action_noise_mean'],
                                     experiment1_configuration['action_noise_sigma']),
    "param_noise": get_param_noise(experiment1_configuration['param_noise_initial_stddev'],
                                   experiment1_configuration['param_noise_desired_action_stddev'],
                                   experiment1_configuration['param_noise_adaption_interval']),
    "eval_env": experiment1_configuration['eval_env'],
    "memory_policy": None,
    "configuration": experiment2_configuration
}

experiments = [experiment1, experiment2]
