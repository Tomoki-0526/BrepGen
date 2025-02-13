import os
import torch
from utils import get_args_vae

# Parse input augments
args = get_args_vae()

# Set PyTorch to use only the specified GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.gpu))

# Make project directory if not exist
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

from trainer import SurfVAETrainer
from dataset import SurfData
from trainer import EdgeVAETrainer
from dataset import EdgeData

M = 30  # max number of faces in a cad model
N = 20  # max number of edges of a face in a cad model

def run(args):
    if args.option == 'surface':
        train_dataset = SurfData(args.data, args.train_list, aug=args.data_aug)
        val_dataset = SurfData(args.data, args.val_list, validate=True, aug=False)
        test_dataset = SurfData(args.data, args.test_list, test=True, aug=False)
        vae = SurfVAETrainer(args, train_dataset, val_dataset, test_dataset)

        print('Start surface inference...')
        z = vae.inference_latent() 

    else:
        assert args.option == 'edge', 'please choose between surface or edge'
        train_dataset = EdgeData(args.data, args.train_list, aug=args.data_aug)
        val_dataset = EdgeData(args.data, args.val_list, validate=True, aug=False)
        test_dataset = EdgeData(args.data, args.test_list, test=True, aug=False)
        vae = EdgeVAETrainer(args, train_dataset, val_dataset, test_dataset)

        print('Start edge inference...')
        z = vae.inference_latent()

    return z

if __name__ == "__main__":
    run(args)