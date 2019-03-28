import os

#from stable_baselines.her.experiment.train import main
from baselines.run import main

from config import DDPG_DENSE, DDPG_SPARSE, DDPG_HER_DENSE, DDPG_HER_SPARSE, GPU_ID
import threading

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

    print("Starting experiment.")
    os.environ['OPENAI_LOGDIR'] = 'logs/DDPG_DENSE/'
    main(DDPG_SPARSE)
