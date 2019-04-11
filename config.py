from datetime import datetime

now = datetime.now()
dt_string = now.strftime("%Y_%m_%d_%H:%M")

HER_SPARSE_1_V0 = {
    'parameters': [
        '--env', 'distal-1-continuous-sparse-mm-v0',
        '--extra_import', 'ctm2_envs',
        '--alg', 'her',
        '--num_cpu', '19',
        '--num_timesteps', str(8e6),
        '--render', 'False',
        # env
        '--max_u', '1.',  # max absolute value of actions on different coordinates
        # ddpg
        '--layers', '1',  # number of layers in the critic/actor networks
        '--hidden', '32',  # number of neurons in each hidden layers
        '--network_class', 'baselines.her.actor_critic:ActorCritic',
        '--Q_lr', '0.001',  # critic learning rate
        '--pi_lr', '0.001',  # actor learning rate
        '--buffer_size', str(int(1E6)),  # for experience replay
        '--polyak', '0.95',  # polyak averaging coefficient
        '--action_l2', '1.0',  # quadratic penalty on actions (before rescaling by max_u)
        '--clip_obs', '200.',
        '--scope', 'her_sparse_1',  # can be tweaked for testing
        '--relative_goals', 'False',
        # training
        '--n_cycles', '50',  # per epoch
        '--rollout_batch_size', '1',  # per mpi thread
        '--num_env', '1',
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
        '--norm_eps', '0.01',  # epsilon used for observation normalization
        '--norm_clip', '5',  # normalized observations are cropped to this values
        '--save_path', 'policies/' + dt_string + '/her_sparse_1_v0'],
    'gpu_id': '0',
    'name': 'her_sparse_1_v0'
}

HER_SPARSE_1_V1 = {
    'parameters': [
        '--env', 'distal-1-continuous-sparse-mm-v0',
        '--extra_import', 'ctm2_envs',
        '--alg', 'her',
        '--num_cpu', '19',
        '--num_timesteps', str(8e6),
        '--render', 'False',
        # env
        '--max_u', '1.',  # max absolute value of actions on different coordinates
        # ddpg
        '--layers', '2',  # number of layers in the critic/actor networks
        '--hidden', '64',  # number of neurons in each hidden layers
        '--network_class', 'baselines.her.actor_critic:ActorCritic',
        '--Q_lr', '0.001',  # critic learning rate
        '--pi_lr', '0.001',  # actor learning rate
        '--buffer_size', str(int(1E6)),  # for experience replay
        '--polyak', '0.95',  # polyak averaging coefficient
        '--action_l2', '1.0',  # quadratic penalty on actions (before rescaling by max_u)
        '--clip_obs', '200.',
        '--scope', 'her_sparse_1',  # can be tweaked for testing
        '--relative_goals', 'False',
        # training
        '--n_cycles', '50',  # per epoch
        '--rollout_batch_size', '1',  # per mpi thread
        '--num_env', '1',
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
        '--norm_eps', '0.01',  # epsilon used for observation normalization
        '--norm_clip', '5',  # normalized observations are cropped to this values
        '--save_path', 'policies/' + dt_string + '/her_sparse_1_v1'],
    'gpu_id': '0',
    'name': 'her_sparse_1_v1'
}

HER_SPARSE_1_V2 = {
    'parameters': [
        '--env', 'distal-1-continuous-sparse-mm-v0',
        '--extra_import', 'ctm2_envs',
        '--alg', 'her',
        '--num_cpu', '19',
        '--num_timesteps', str(2.5e6),
        '--render', 'False',
        # env
        '--max_u', '1.',  # max absolute value of actions on different coordinates
        # ddpg
        '--layers', '3',  # number of layers in the critic/actor networks
        '--hidden', '256',  # number of neurons in each hidden layers
        '--network_class', 'baselines.her.actor_critic:ActorCritic',
        '--Q_lr', '0.001',  # critic learning rate
        '--pi_lr', '0.001',  # actor learning rate
        '--buffer_size', str(int(1E6)),  # for experience replay
        '--polyak', '0.95',  # polyak averaging coefficient
        '--action_l2', '1.0',  # quadratic penalty on actions (before rescaling by max_u)
        '--clip_obs', '200.',
        '--scope', 'her_sparse_1',  # can be tweaked for testing
        '--relative_goals', 'False',
        # training
        '--n_cycles', '10',  # per epoch
        '--rollout_batch_size', '1',  # per mpi thread
        '--n_batches', '40',  # training batches per cycle
        '--batch_size', '256',  # per mpi thread, measured in transitions and reduced to even multiple of chunk_length.
        '--n_test_rollouts', '10',  # number of test rollouts per epoch, each consists of rollout_batch_size rollouts
        '--test_with_polyak', 'False',  # run test episodes with the target network
        # exploration
        '--random_eps', '0.3',  # percentage of time a random action is taken
        '--noise_eps', '0.5',  # std of gaussian noise added to not-completely-random actions as a percentage of max_u
        # HER
        '--replay_strategy', 'future',  # supported modes: future, none
        '--replay_k', '4',  # number of additional goals used for replay, only used if off_policy_data=future
        # normalization
        '--norm_eps', '0.01',  # epsilon used for observation normalization
        '--norm_clip', '5',  # normalized observations are cropped to this values
        '--save_path', 'policies/' + dt_string + '/her_sparse_1_v2'],
    'gpu_id': '1',
    'name': 'her_sparse_1_v2'
}

HER_SPARSE_1_V3 = {
    'parameters': [
        '--env', 'distal-1-continuous-sparse-mm-v0',
        '--extra_import', 'ctm2_envs',
        '--alg', 'her',
        '--num_cpu', '19',
        '--num_timesteps', str(2.5e6),
        '--render', 'False',
        # env
        '--max_u', '1.',  # max absolute value of actions on different coordinates
        # ddpg
        '--layers', '2',  # number of layers in the critic/actor networks
        '--hidden', '64',  # number of neurons in each hidden layers
        '--network_class', 'baselines.her.actor_critic:ActorCritic',
        '--Q_lr', '0.001',  # critic learning rate
        '--pi_lr', '0.001',  # actor learning rate
        '--buffer_size', str(int(1E6)),  # for experience replay
        '--polyak', '0.95',  # polyak averaging coefficient
        '--action_l2', '1.0',  # quadratic penalty on actions (before rescaling by max_u)
        '--clip_obs', '200.',
        '--scope', 'her_sparse_1',  # can be tweaked for testing
        '--relative_goals', 'True',
        # training
        '--n_cycles', '10',  # per epoch
        '--rollout_batch_size', '1',  # per mpi thread
        '--n_batches', '40',  # training batches per cycle
        '--batch_size', '256',  # per mpi thread, measured in transitions and reduced to even multiple of chunk_length.
        '--n_test_rollouts', '10',  # number of test rollouts per epoch, each consists of rollout_batch_size rollouts
        '--test_with_polyak', 'False',  # run test episodes with the target network
        # exploration
        '--random_eps', '0.3',  # percentage of time a random action is taken
        '--noise_eps', '0.5',  # std of gaussian noise added to not-completely-random actions as a percentage of max_u
        # HER
        '--replay_strategy', 'future',  # supported modes: future, none
        '--replay_k', '4',  # number of additional goals used for replay, only used if off_policy_data=future
        # normalization
        '--norm_eps', '0.01',  # epsilon used for observation normalization
        '--norm_clip', '5',  # normalized observations are cropped to this values
        '--save_path', 'policies/' + dt_string + '/her_sparse_1_v3'],
    'gpu_id': '1',
    'name': 'her_sparse_1_v3'
}

HER_SPARSE_2_V0 = {
    'parameters': [
        '--env', 'distal-2-continuous-sparse-mm-v0',
        '--extra_import', 'ctm2_envs',
        '--alg', 'her',
        '--num_cpu', '19',
        '--num_timesteps', str(2.5e6),
        '--render', 'False',
        # env
        '--max_u', '1.',  # max absolute value of actions on different coordinates
        # ddpg
        '--layers', '3',  # number of layers in the critic/actor networks
        '--hidden', '256',  # number of neurons in each hidden layers
        '--network_class', 'baselines.her.actor_critic:ActorCritic',
        '--Q_lr', '0.001',  # critic learning rate
        '--pi_lr', '0.001',  # actor learning rate
        '--buffer_size', str(int(1E6)),  # for experience replay
        '--polyak', '0.95',  # polyak averaging coefficient
        '--action_l2', '1.0',  # quadratic penalty on actions (before rescaling by max_u)
        '--clip_obs', '200.',
        '--scope', 'her_sparse_2',  # can be tweaked for testing
        '--relative_goals', 'False',
        # training
        '--n_cycles', '10',  # per epoch
        '--rollout_batch_size', '1',  # per mpi thread
        '--n_batches', '40',  # training batches per cycle
        '--batch_size', '256',  # per mpi thread, measured in transitions and reduced to even multiple of chunk_length.
        '--n_test_rollouts', '10',  # number of test rollouts per epoch, each consists of rollout_batch_size rollouts
        '--test_with_polyak', 'False',  # run test episodes with the target network
        # exploration
        '--random_eps', '0.3',  # percentage of time a random action is taken
        '--noise_eps', '0.5',  # std of gaussian noise added to not-completely-random actions as a percentage of max_u
        # HER
        '--replay_strategy', 'future',  # supported modes: future, none
        '--replay_k', '4',  # number of additional goals used for replay, only used if off_policy_data=future
        # normalization
        '--norm_eps', '0.01',  # epsilon used for observation normalization
        '--norm_clip', '5',  # normalized observations are cropped to this values
        '--save_path', 'policies/' + dt_string + '/her_sparse_2_v0'],
    'gpu_id': '1',
    'name': 'her_sparse_2_v0'
}
