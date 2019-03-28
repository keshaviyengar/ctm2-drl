import os
from argparse import ArgumentParser

from baselines.run import main
from config import DDPG_DENSE, DDPG_SPARSE, HER_DENSE, HER_SPARSE

os.environ["CUDA_VISIBLE_DEVICES"] = DDPG_DENSE['gpu_id']

parser = ArgumentParser()
parser.add_argument("-e", "--experiment", dest="experiment", type=str,
                    help="Specify experiment: ddpg / her + _ + sparse / dense", metavar="EXP")

args = parser.parse_args()
print("Starting experiment.")
print(args.experiment)
if args.experiment == DDPG_DENSE['name']:
    os.environ['OPENAI_LOGDIR'] = 'logs/' + DDPG_DENSE['name']
    main(DDPG_DENSE['parameters'])
elif args.experiment == DDPG_SPARSE['name']:
    os.environ['OPENAI_LOGDIR'] = 'logs/' + DDPG_SPARSE['name']
    main(DDPG_SPARSE['parameters'])
elif args.experiment == HER_DENSE['name']:
    os.environ['OPENAI_LOGDIR'] = 'logs/' + HER_DENSE['name']
    main(HER_DENSE['parameters'])
elif args.experiment == HER_SPARSE['name']:
    os.environ['OPENAI_LOGDIR'] = 'logs/' + HER_SPARSE['name']
    main(HER_SPARSE['parameters'])
else:
    raise NameError
