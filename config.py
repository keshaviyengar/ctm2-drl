import os

GPU_ID = 0

# DDPG_DENSE_PARAMS
DDPG_DENSE_LAUNCH_PARAMS = [
    '--env', 'ctm2-continuous-dense-v0',
    '--extra_import', 'ctm2_envs',
    '--alg', 'her',
    '--num_cpu', '3',
    '--num_timesteps', str(2.5e6),
    # env
    '--max_u', '1.',  # max absolute value of actions on different coordinates
    # ddpg
    '--layers', '1',  # number of layers in the critic/actor networks
    '--hidden', '64',  # number of neurons in each hidden layers
    '--network_class', 'baselines.her.actor_critic:ActorCritic',
    '--Q_lr', '0.001',  # critic learning rate
    '--pi_lr', '0.001',  # actor learning rate
    '--buffer_size', str(int(1E6)),  # for experience replay
    '--polyak', '0.95',  # polyak averaging coefficient
    '--action_l2', '1.0',  # quadratic penalty on actions (before rescaling by max_u)
    '--clip_obs', '200.',
    '--scope', 'ddpg',  # can be tweaked for testing
    '--relative_goals', 'False',
    # training
    '--n_cycles', '50',  # per epoch
    '--rollout_batch_size', '1',  # per mpi thread
    '--n_batches', '40',  # training batches per cycle
    '--batch_size', '256',  # per mpi thread, measured in transitions and reduced to even multiple of chunk_length.
    '--n_test_rollouts', '10',  # number of test rollouts per epoch, each consists of rollout_batch_size rollouts
    '--test_with_polyak', 'False',  # run test episodes with the target network
    # exploration
    '--random_eps', '0.3',  # percentage of time a random action is taken
    '--noise_eps', '0.2',  # std of gaussian noise added to not-completely-random actions as a percentage of max_u
    # HER
    '--replay_strategy', 'none',  # supported modes: future, none
    '--replay_k', '4',  # number of additional goals used for replay, only used if off_policy_data=future
    # normalization
    '--norm_eps', '1',  # epsilon used for observation normalization
    '--norm_clip', '5',  # normalized observations are cropped to this values
    '--save_path', 'policies/her'
]

# DDPG_SPARSE_PARAMS
DDPG_SPARSE_LAUNCH_PARAMS = [
    '--env', 'ctm2-continuous-sparse-v0',
    '--extra_import', 'ctm2_envs',
    '--alg', 'her',
    '--num_cpu', '3',
    '--num_timesteps', str(2.5e6),
    # env
    '--max_u', '1.',  # max absolute value of actions on different coordinates
    # ddpg
    '--layers', '1',  # number of layers in the critic/actor networks
    '--hidden', '64',  # number of neurons in each hidden layers
    '--network_class', 'baselines.her.actor_critic:ActorCritic',
    '--Q_lr', '0.001',  # critic learning rate
    '--pi_lr', '0.001',  # actor learning rate
    '--buffer_size', str(int(1E6)),  # for experience replay
    '--polyak', '0.95',  # polyak averaging coefficient
    '--action_l2', '1.0',  # quadratic penalty on actions (before rescaling by max_u)
    '--clip_obs', '200.',
    '--scope', 'ddpg',  # can be tweaked for testing
    '--relative_goals', 'False',
    # training
    '--n_cycles', '50',  # per epoch
    '--rollout_batch_size', '1',  # per mpi thread
    '--n_batches', '40',  # training batches per cycle
    '--batch_size', '256',  # per mpi thread, measured in transitions and reduced to even multiple of chunk_length.
    '--n_test_rollouts', '10',  # number of test rollouts per epoch, each consists of rollout_batch_size rollouts
    '--test_with_polyak', 'False',  # run test episodes with the target network
    # exploration
    '--random_eps', '0.3',  # percentage of time a random action is taken
    '--noise_eps', '0.2',  # std of gaussian noise added to not-completely-random actions as a percentage of max_u
    # HER
    '--replay_strategy', 'none',  # supported modes: future, none
    '--replay_k', '4',  # number of additional goals used for replay, only used if off_policy_data=future
    # normalization
    '--norm_eps', '1',  # epsilon used for observation normalization
    '--norm_clip', '5',  # normalized observations are cropped to this values
    '--save_path', 'policies/her'
]

# DDPG_SPARSE_PARAMS
DDPG_HER_DENSE_LAUNCH_PARAMS = [
    '--env', 'ctm2-continuous-dense-v0',
    '--extra_import', 'ctm2_envs',
    '--alg', 'her',
    '--num_cpu', '3',
    '--num_timesteps', str(2.5e6),
    # env
    '--max_u', '1.',  # max absolute value of actions on different coordinates
    # ddpg
    '--layers', '1',  # number of layers in the critic/actor networks
    '--hidden', '64',  # number of neurons in each hidden layers
    '--network_class', 'baselines.her.actor_critic:ActorCritic',
    '--Q_lr', '0.001',  # critic learning rate
    '--pi_lr', '0.001',  # actor learning rate
    '--buffer_size', str(int(1E6)),  # for experience replay
    '--polyak', '0.95',  # polyak averaging coefficient
    '--action_l2', '1.0',  # quadratic penalty on actions (before rescaling by max_u)
    '--clip_obs', '200.',
    '--scope', 'ddpg',  # can be tweaked for testing
    '--relative_goals', 'False',
    # training
    '--n_cycles', '50',  # per epoch
    '--rollout_batch_size', '1',  # per mpi thread
    '--n_batches', '40',  # training batches per cycle
    '--batch_size', '256',  # per mpi thread, measured in transitions and reduced to even multiple of chunk_length.
    '--n_test_rollouts', '10',  # number of test rollouts per epoch, each consists of rollout_batch_size rollouts
    '--test_with_polyak', 'False',  # run test episodes with the target network
    # exploration
    '--random_eps', '0.3',  # percentage of time a random action is taken
    '--noise_eps', '0.2',  # std of gaussian noise added to not-completely-random actions as a percentage of max_u
    # HER
    '--replay_strategy', 'future',  # supported modes: future, none
    '--replay_k', '4',  # number of additional goals used for replay, only used if off_policy_data=future
    # normalization
    '--norm_eps', '1',  # epsilon used for observation normalization
    '--norm_clip', '5',  # normalized observations are cropped to this values
    '--save_path', 'policies/her'
]

# DDPG_SPARSE_PARAMS
DDPG_HER_SPARSE_LAUNCH_PARAMS = [
    '--env', 'ctm2-continuous-sparse-v0',
    '--extra_import', 'ctm2_envs',
    '--alg', 'her',
    '--num_cpu', '3',
    '--num_timesteps', str(2.5e6),
    # env
    '--max_u', '1.',  # max absolute value of actions on different coordinates
    # ddpg
    '--layers', '1',  # number of layers in the critic/actor networks
    '--hidden', '64',  # number of neurons in each hidden layers
    '--network_class', 'baselines.her.actor_critic:ActorCritic',
    '--Q_lr', '0.001',  # critic learning rate
    '--pi_lr', '0.001',  # actor learning rate
    '--buffer_size', str(int(1E6)),  # for experience replay
    '--polyak', '0.95',  # polyak averaging coefficient
    '--action_l2', '1.0',  # quadratic penalty on actions (before rescaling by max_u)
    '--clip_obs', '200.',
    '--scope', 'ddpg',  # can be tweaked for testing
    '--relative_goals', 'False',
    # training
    '--n_cycles', '50',  # per epoch
    '--rollout_batch_size', '1',  # per mpi thread
    '--n_batches', '40',  # training batches per cycle
    '--batch_size', '256',  # per mpi thread, measured in transitions and reduced to even multiple of chunk_length.
    '--n_test_rollouts', '10',  # number of test rollouts per epoch, each consists of rollout_batch_size rollouts
    '--test_with_polyak', 'False',  # run test episodes with the target network
    # exploration
    '--random_eps', '0.3',  # percentage of time a random action is taken
    '--noise_eps', '0.2',  # std of gaussian noise added to not-completely-random actions as a percentage of max_u
    # HER
    '--replay_strategy', 'future',  # supported modes: future, none
    '--replay_k', '4',  # number of additional goals used for replay, only used if off_policy_data=future
    # normalization
    '--norm_eps', '1',  # epsilon used for observation normalization
    '--norm_clip', '5',  # normalized observations are cropped to this values
    '--save_path', 'policies/her'
]

EXPERIMENTS = [DDPG_DENSE_LAUNCH_PARAMS, DDPG_SPARSE_LAUNCH_PARAMS,
               DDPG_HER_DENSE_LAUNCH_PARAMS, DDPG_HER_SPARSE_LAUNCH_PARAMS]


