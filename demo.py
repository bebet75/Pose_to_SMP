import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import imageio, os
from smplpytorch.pytorch.smpl_layer import SMPL_Layer
from display_utils import display_model
import pickle


with open('fit/output/HAA4D/baseball_swing_004_params.pkl', 'rb') as f:
    data = pickle.load(f)

if __name__ == '__main__':
    for i in range(len(data['pose_params'])):
        cuda = False
        batch_size = 1

        # Create the SMPL layer
        smpl_layer = SMPL_Layer(
            center_idx=0,
            gender='male',
            model_root='smplpytorch/native/models')

        # Generate random pose and shape parameters
        pose_params = torch.tensor([data['pose_params'][i]])
        shape_params = torch.tensor([data['shape_params'][i]])
        print(pose_params)

        # GPU mode
        if cuda:
            pose_params = pose_params.cuda()
            shape_params = shape_params.cuda()
            smpl_layer.cuda()

        # Forward from the SMPL layer
        verts, Jtr = smpl_layer(pose_params, th_betas=shape_params)


        if i < 10:
            stri = '0' + str(i)
        else: 
            stri = str(i)
        outputloc = 'Outputs/' + 'image' + stri + '.png'

        # Draw output vertices and joints
        display_model(
            {'verts': verts.cpu().detach(),
            'joints': Jtr.cpu().detach()},
            model_faces=smpl_layer.th_faces,
            with_joints=True,
            kintree_table=smpl_layer.kintree_table,
            savepath=outputloc,
            show=False)

    images = []
    filenames = sorted(fn for fn in os.listdir('Outputs') )
    for filename in filenames:
        images.append(imageio.imread('Outputs/'+filename))
    imageio.mimsave('clapping_example.gif', images, duration=0.2)
