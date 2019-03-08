from stable_baselines import DDPG
import os
from config import global_configuration, experiments
import threading


def launch_tensorboard(tensorboard_path):
    import os
    os.system('tensorboard --logdir=' + tensorboard_path)
    return


print("Starting tensorboard thread.")
t = threading.Thread(target=launch_tensorboard, args=(["ctm2_tb/"]))
t.start()

print("Starting experiments.")

if global_configuration["single_gpu"]:
    os.environ["CUDA_VISIBLE_DEVICES"] = global_configuration['gpu_id']

model = None
for exp in experiments:
    config = exp['configuration']
    algo = exp['algorithm']
    if algo == "DDPG":
        model = DDPG(policy=exp['policy'], env=exp['env'], action_noise=exp['action_noise'],
                     param_noise=exp['param_noise'], eval_env=exp['eval_env'], memory_policy=exp['memory_policy'],
                     gamma=config['gamma'], nb_train_steps=config['nb_train_steps'],
                     nb_rollout_steps=config['nb_rollout_steps'], nb_eval_steps=config['nb_eval_steps'],
                     normalize_observations=config['normalize_observations'], tau=config['tau'],
                     batch_size=config['batch_size'],
                     param_noise_adaption_interval=config['param_noise_adaption_interval'],
                     normalize_returns=config['normalize_returns'], enable_popart=config['enable_popart'],
                     observation_range=config['observation_range'], critic_l2_reg=config['critic_l2_reg'],
                     return_range=config['return_range'], actor_lr=config['actor_lr'],critic_lr=config['critic_lr'],
                     clip_norm=config['clip_norm'], reward_scale=config['reward_scale'], render=config['render'],
                     render_eval=config['render_eval'], memory_limit=config['memory_limit'], verbose=config['verbose'],
                     tensorboard_log=config['tensorboard_log'], _init_setup_model=config['_init_setup_model'],
                     policy_kwargs=config['policy_kwargs'], full_tensorboard_log=config['full_tensorboard_log']
                     )
    elif algo == "DQN":
        pass
    elif algo == "HER":
        pass
    else:
        NameError('algo not set correctly')
    model.learn(total_timesteps=config['total_timesteps'], callback=None, seed=global_configuration['seed'],
                log_interval=config['log_interval'], tb_log_name=exp['algorithm'], reset_num_timesteps=True)

    model.save(algo + str(config['experiment_id']))
