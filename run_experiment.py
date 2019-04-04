import os
from argparse import ArgumentParser
from datetime import datetime

from baselines.run import main
from config import HER_SPARSE_1_V0, HER_SPARSE_2_V0, HER_SPARSE_2_V1

os.environ['OPENAI_LOG_FORMAT'] = 'stdout,log,csv,tensorboard'
now = datetime.now()
dt_string = now.strftime("%Y_%m_%d_%H:%M")

parser = ArgumentParser()
parser.add_argument("-e", "--experiment", dest="experiment", type=str,
                    help="Specify experiment: ddpg / her + _ + sparse / dense", metavar="EXP")

args = parser.parse_args()
print("Starting experiment.")
print(args.experiment)
directory = 'logs/' + dt_string + '/'
if args.experiment == HER_SPARSE_1_V0['name']:
    os.environ["CUDA_VISIBLE_DEVICES"] = HER_SPARSE_1_V0['gpu_id']
    os.environ['OPENAI_LOGDIR'] = directory + HER_SPARSE_1_V0['name']
    main(HER_SPARSE_1_V0['parameters'])

elif args.experiment == HER_SPARSE_2_V0['name']:
    os.environ["CUDA_VISIBLE_DEVICES"] = HER_SPARSE_2_V0['gpu_id']
    os.environ['OPENAI_LOGDIR'] = directory + HER_SPARSE_2_V0['name']
    main(HER_SPARSE_2_V0['parameters'])

elif args.experiment == HER_SPARSE_2_V1['name']:
    os.environ["CUDA_VISIBLE_DEVICES"] = HER_SPARSE_2_V1['gpu_id']
    os.environ['OPENAI_LOGDIR'] = directory + HER_SPARSE_2_V1['name']
    main(HER_SPARSE_2_V1['parameters'])

elif args.experiment == HER_SPARSE_2_V2['name']:
    os.environ["CUDA_VISIBLE_DEVICES"] = HER_SPARSE_2_V2['gpu_id']
    os.environ['OPENAI_LOGDIR'] = directory + HER_SPARSE_2_V2['name']
    main(HER_SPARSE_2_V2['parameters'])

else:
    raise NameError

