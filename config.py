DDPG_DENSE = {
    'parameters': [
        '--env', 'distal-1-continuous-dense-v0',
        '--extra_import', 'ctm2_envs',
        '--alg', 'her',
        '--num_cpu', '1',
        '--num_timesteps', str(12000),
        # env
        '--max_u', '1.',  # max absolute value of actions on different coordinates
        # ddpg
        '--layers', '2',  # number of layers in the critic/actor networks
        '--hidden', '16',  # number of neurons in each hidden layers
        '--network_class', 'baselines.her.actor_critic:ActorCritic',
        '--Q_lr', '0.001',  # critic learning rate
        '--pi_lr', '0.001',  # actor learning rate
        '--buffer_size', str(int(1E3)),  # for experience replay
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
        '--n_test_rollouts', '50',  # number of test rollouts per epoch, each consists of rollout_batch_size rollouts
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
        '--save_path', 'policies/ddpg_dense'],
    'gpu_id': '0',
    'name': 'ddpg_dense'
}

DDPG_SPARSE = {
    'parameters': [
        '--env', 'distal-1-continuous-sparse-v0',
        '--extra_import', 'ctm2_envs',
        '--alg', 'her',
        '--num_cpu', '1',
        '--num_timesteps', str(12000),
        # env
        '--max_u', '1.',  # max absolute value of actions on different coordinates
        # ddpg
        '--layers', '2',  # number of layers in the critic/actor networks
        '--hidden', '16',  # number of neurons in each hidden layers
        '--network_class', 'baselines.her.actor_critic:ActorCritic',
        '--Q_lr', '0.001',  # critic learning rate
        '--pi_lr', '0.001',  # actor learning rate
        '--buffer_size', str(int(1E3)),  # for experience replay
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
        '--n_test_rollouts', '50',  # number of test rollouts per epoch, each consists of rollout_batch_size rollouts
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
        '--save_path', 'policies/ddpg_sparse'],
    'gpu_id': '0',
    'name': 'ddpg_sparse'
}

HER_DENSE = {
    'parameters': [
        '--env', 'distal-1-continuous-dense-v0',
        '--extra_import', 'ctm2_envs',
        '--alg', 'her',
        '--num_cpu', '1',
        '--num_timesteps', str(12000),
        # env
        '--max_u', '1.',  # max absolute value of actions on different coordinates
        # ddpg
        '--layers', '2',  # number of layers in the critic/actor networks
        '--hidden', '16',  # number of neurons in each hidden layers
        '--network_class', 'baselines.her.actor_critic:ActorCritic',
        '--Q_lr', '0.01',  # critic learning rate
        '--pi_lr', '0.01',  # actor learning rate
        '--buffer_size', str(int(1E3)),  # for experience replay
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
        '--n_test_rollouts', '50',  # number of test rollouts per epoch, each consists of rollout_batch_size rollouts
        '--test_with_polyak', 'True',  # run test episodes with the target network
        # exploration
        '--random_eps', '0.3',  # percentage of time a random action is taken
        '--noise_eps', '0.2',  # std of gaussian noise added to not-completely-random actions as a percentage of max_u
        # HER
        '--replay_strategy', 'future',  # supported modes: future, none
        '--replay_k', '4',  # number of additional goals used for replay, only used if off_policy_data=future
        # normalization
        '--norm_eps', '1',  # epsilon used for observation normalization
        '--norm_clip', '5',  # normalized observations are cropped to this values
        '--save_path', 'policies/her_dense'],
    'gpu_id': '1',
    'name': 'her_dense'
}

HER_SPARSE = {
    'parameters': [
        '--env', 'distal-1-continuous-sparse-v0',
        '--extra_import', 'ctm2_envs',
        '--alg', 'her',
        '--num_cpu', '1',
        '--num_timesteps', str(2.5e6),
        # env
        '--max_u', '1.',  # max absolute value of actions on different coordinates
        # ddpg
        '--layers', '2',  # number of layers in the critic/actor networks
        '--hidden', '16',  # number of neurons in each hidden layers
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
        '--test_with_polyak', 'True',  # run test episodes with the target network
        # exploration
        '--random_eps', '0.3',  # percentage of time a random action is taken
        '--noise_eps', '0.2',  # std of gaussian noise added to not-completely-random actions as a percentage of max_u
        # HER
        '--replay_strategy', 'future',  # supported modes: future, none
        '--replay_k', '4',  # number of additional goals used for replay, only used if off_policy_data=future
        # normalization
        '--norm_eps', '1',  # epsilon used for observation normalization
        '--norm_clip', '5',  # normalized observations are cropped to this values
        '--save_path', 'policies/her_sparse'],
    'gpu_id': '0',
    'name': 'her_sparse'
}
