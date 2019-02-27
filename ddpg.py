import gym
import ctm2_rl
import numpy as np

from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy

from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG

from numpy import inf
import yaml
import os

best_mean_reward, n_steps = -np.inf, 0


def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_steps, best_mean_reward
    # Print stats every 1000 calls
    if (n_steps + 1) % 100 == 0:
        # Evaluate policy performance
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        if len(x) > 0:
            mean_reward = np.mean(y[-100:])
            print(x[-1], 'timesteps')
            print(
                "Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

            # New best model, you could save the agent here
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                # Example for saving best model
                print("Saving new best model")
                _locals['self'].save(log_dir + 'best_model.pkl')
    n_steps += 1
    return True


# Create log dir
log_dir = "/tmp/gym/"
os.makedirs(log_dir, exist_ok=True)

# Load in parameters
config = yaml.load(open('ddpg_parameters.yaml'))
ddpg_params = config['ddpg']
os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu_id']

# Create and wrap the environment
env = gym.make('ctm2-continuous-v0')
env = Monitor(env, log_dir, allow_early_resets=True)
env = DummyVecEnv([lambda: env])

# the noise objects for DDPG
n_actions = env.action_space.shape[-1]

param_noise = AdaptiveParamNoiseSpec(ddpg_params['param_noise']['initial_stddev'],
                                     ddpg_params['param_noise']['desired_action_stddev'],
                                     ddpg_params['param_noise']['adoption_coefficient'])

action_noise = NormalActionNoise(mean=ddpg_params['action_noise']['mean'] + np.zeros(n_actions),
                                 sigma=ddpg_params['action_noise']['sigma'] * np.ones(n_actions))
model = DDPG(MlpPolicy, env, gamma=ddpg_params['gamma'], memory_policy=ddpg_params['memory_policy'],
             eval_env=ddpg_params['eval_env'], nb_train_steps=ddpg_params['nb_train_steps'],
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

model.learn(total_timesteps=ddpg_params['total_timesteps'], callback=callback)
