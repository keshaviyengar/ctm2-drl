import os
from argparse import ArgumentParser
from datetime import datetime

from baselines.run import main
from config import DDPG_DENSE, DDPG_SPARSE, HER_DENSE, HER_SPARSE

os.environ['OPENAI_LOG_FORMAT'] = 'stdout,log,csv,tensorboard'
os.environ["CUDA_VISIBLE_DEVICES"] = DDPG_DENSE['gpu_id']
now = datetime.now()
dt_string = now.strftime("%Y_%m_%d_%H:%M")

parser = ArgumentParser()
parser.add_argument("-e", "--experiment", dest="experiment", type=str,
                    help="Specify experiment: ddpg / her + _ + sparse / dense", metavar="EXP")

args = parser.parse_args()
print("Starting experiment.")
print(args.experiment)
directory = 'logs/' + dt_string + '/'
if args.experiment == DDPG_DENSE['name']:
    print("Doing ddpg_dense")
    os.environ['OPENAI_LOGDIR'] = directory + DDPG_DENSE['name']
    main(DDPG_DENSE['parameters'])
elif args.experiment == DDPG_SPARSE['name']:
    print("Doing ddpg_sparse")
    os.environ['OPENAI_LOGDIR'] = directory + DDPG_SPARSE['name']
    main(DDPG_SPARSE['parameters'])
elif args.experiment == HER_DENSE['name']:
    print("Doing her_dense")
    os.environ['OPENAI_LOGDIR'] = directory + HER_DENSE['name']
    main(HER_DENSE['parameters'])
elif args.experiment == HER_SPARSE['name']:
    print("Doing her_sparse")
    os.environ['OPENAI_LOGDIR'] = directory + HER_SPARSE['name']
    main(HER_SPARSE['parameters'])
else:
    raise NameError

