import os

#from stable_baselines.her.experiment.train import main
from baselines.run import main

from config import EXPERIMENTS, GPU_ID
import threading

os.environ['OPENAI_LOG_FORMAT'] = 'stdout,log,csv,tensorboard'

def launch_tensorboard(tensorboard_path):
    os.system('tensorboard --logdir=' + tensorboard_path)
    return


if __name__ == '__main__':
    print("Starting tensorboard thread.")
    t = threading.Thread(target=launch_tensorboard, args=(["logs/"]))
    t.start()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

    print("Starting experiments.")
    log_counter = 0
    for EXP in EXPERIMENTS:
        os.environ['OPENAI_LOGDIR'] = 'logs/' + str(log_counter)
        main(EXP)
        log_counter += 1
