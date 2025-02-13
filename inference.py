import os
import numpy as np
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

z_save_dir = '/home/szj/DGCNNEncoderZ/data/z'

def run(args):
    if args.option == 'surface':
        train_dataset = SurfData(args.data, args.train_list, aug=args.data_aug)
        val_dataset = SurfData(args.data, args.val_list, validate=True, aug=False)
        test_dataset = SurfData(args.data, args.test_list, test=True, aug=False)
        vae = SurfVAETrainer(args, train_dataset, val_dataset, test_dataset)

        print('Start surface inference...')
        z = vae.inference_latent() 

        train_groups = train_dataset.groups
        train_groups_name = train_dataset.groups_name
        last_idx = 0
        for i, (chunk, uid) in enumerate(train_groups_name):
            num_faces = train_groups[i]
            output_dir = os.path.join(z_save_dir, chunk, uid, 'surf')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            for j in range(num_faces):
                z_face = z['train'][last_idx + j].cpu().detach().numpy()
                np.save(os.path.join(output_dir, 'surf_{}.npy'.format(j)), z_face)
            last_idx += num_faces
        val_groups = val_dataset.groups
        val_groups_name = val_dataset.groups_name
        last_idx = 0
        for i, (chunk, uid) in enumerate(val_groups_name):
            num_faces = val_groups[i]
            output_dir = os.path.join(z_save_dir, chunk, uid, 'surf')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            for j in range(num_faces):
                z_face = z['val'][last_idx + j].cpu().detach().numpy()
                np.save(os.path.join(output_dir, 'surf_{}.npy'.format(j)), z_face)
            last_idx += num_faces
        test_groups = test_dataset.groups
        test_groups_name = test_dataset.groups_name
        last_idx = 0
        for i, (chunk, uid) in enumerate(test_groups_name):
            num_faces = test_groups[i]
            output_dir = os.path.join(z_save_dir, chunk, uid, 'surf')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            for j in range(num_faces):
                z_face = z['test'][last_idx + j].cpu().detach().numpy()
                np.save(os.path.join(output_dir, 'surf_{}.npy'.format(j)), z_face)
            last_idx += num_faces

    else:
        assert args.option == 'edge', 'please choose between surface or edge'
        train_dataset = EdgeData(args.data, args.train_list, aug=args.data_aug)
        val_dataset = EdgeData(args.data, args.val_list, validate=True, aug=False)
        test_dataset = EdgeData(args.data, args.test_list, test=True, aug=False)
        vae = EdgeVAETrainer(args, train_dataset, val_dataset, test_dataset)

        print('Start edge inference...')
        z = vae.inference_latent()

if __name__ == "__main__":
    run(args)