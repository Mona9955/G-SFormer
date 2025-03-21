import torch
from common.h36m_dataset import Human36mDataset
from common.arguments import parse_args
from common.camera import world_to_camera, normalize_screen_coordinates
from common.generators_stride_rooling_1 import UnchunkedGenerator, ChunkedGenerator
import numpy as np
# from model_temporal_3 import Transformer
# from model_temporal_4 import Transformer
from model_temporal_8_res import Transformer

import os
import copy
import time
import torch.optim as optim
import torch.nn as nn
from common.loss import *
from progress.bar import Bar
import math
import random

seq2frame = True
amass_pretrain = True
args = parse_args()
pad = (args.number_of_frames - 1) // 2
subs_train = ['S1', 'S5', 'S6', 'S7', 'S8']
subs_test = ['S9', 'S11']
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'


seed = 0
random.seed(seed)
np.random.seed(seed)

print('Loading h36m dataset...')
dataset_path = "/data2/cuimengmeng/processed_h36m/data_3d_h36m.npz"
dataset = Human36mDataset(path=dataset_path)
dataset_path_2d = "/data2/cuimengmeng/processed_h36m/data_2d_h36m_cpn_ft_h36m_dbb.npz"
print('Loading 2D detections...')
keypoints_2d = np.load(dataset_path_2d, allow_pickle=True)


keypoints_metadata = keypoints_2d['metadata'].item()
keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
print('kps',kps_left, kps_right)
print('joints', joints_left, joints_right)
keypoints = keypoints_2d['positions_2d'].item()
for subject in subs_test:
    for action in keypoints[subject].keys():
        for cam_idx, kps in enumerate(keypoints[subject][action]):
            cam = dataset[subject][action]['cameras'][cam_idx]
            # std = 5
            # kps += np.random.normal(loc=0.0, scale=std, size=kps.shape)
            kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
            keypoints[subject][action][cam_idx] = kps

all_actions = {}
for subject in subs_test:
    for action in dataset[subject].keys():
        anim = dataset[subject][action]
        action_name = action.split(' ')[0]
        positions_3d = []
        for cam in anim['cameras']:
            pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
            pos_3d[:, 1:] -= pos_3d[:, :1]
            positions_3d.append(pos_3d)
        anim['positions_3d'] = positions_3d

        if action_name not in all_actions:
            all_actions[action_name] = []

        all_actions[action_name].append((subject, action))




def eval_data_prepare_seq2seq(inputs_2d, inputs_3d):

    assert inputs_2d.shape[:-1] == inputs_3d.shape[:-1], "2d and 3d inputs shape must be same! "+str(inputs_2d.shape)+str(inputs_3d.shape)
    receptive_field = 486
    seq_len = 243
    if inputs_2d.shape[0] / receptive_field > inputs_2d.shape[0] // receptive_field:
        out_num = 2 * (inputs_2d.shape[0] // receptive_field) + 2
    elif inputs_2d.shape[0] / receptive_field == inputs_2d.shape[0] // receptive_field:
        out_num = 2 * (inputs_2d.shape[0] // receptive_field)

    eval_input_2d = torch.empty(out_num, seq_len, inputs_2d.shape[1], inputs_2d.shape[2])
    eval_input_3d = torch.empty(out_num, seq_len, inputs_3d.shape[1], inputs_3d.shape[2])

    for i in range(out_num // 2):
        input_2d_i = inputs_2d[i * receptive_field:i * receptive_field + receptive_field, :, :]
        input_3d_i = inputs_3d[i * receptive_field:i * receptive_field + receptive_field, :, :]

        if input_2d_i.shape[0] < receptive_field:
            input_2d_i = inputs_2d[-receptive_field:, :, :]
            input_3d_i = inputs_3d[-receptive_field:, :, :]

        eval_input_2d[2 * i, :, :, :] = torch.from_numpy(input_2d_i[0:-1:2, :, :])
        eval_input_2d[2 * i + 1, :, :, :] = torch.from_numpy(input_2d_i[1::2, :, :])

        eval_input_3d[2 * i, :, :, :] = torch.from_numpy(input_3d_i[0:-1:2, :, :])
        eval_input_3d[2 * i + 1, :, :, :] = torch.from_numpy(input_3d_i[1::2, :, :])

    return eval_input_2d, eval_input_3d


def input_augmentation_seq2seq(input_2D, model_trans, joints_left, joints_right, joints_embed):

    input_2D_aug = copy.deepcopy(input_2D)
    input_2D_aug[:, :, :, 0] *= -1
    input_2D_aug[:, :, joints_left + joints_right] = input_2D_aug[:,:, joints_right + joints_left]
    output_3D_non_flip, enco_3D_non_flip = model_trans(input_2D+joints_embed)
    output_3D_flip, enco_3D_flip = model_trans(input_2D_aug+joints_embed)
    enco_3D_flip[:,:,:, 0] *= -1
    enco_3D_flip[:, :, joints_left + joints_right] = enco_3D_flip[:,:, joints_right + joints_left]
    output_3D = (enco_3D_non_flip + enco_3D_flip) / 2
    return output_3D



def input_augmentation(input_2D, model_trans, joints_left, joints_right, joints_embed):

    input_2D_aug = copy.deepcopy(input_2D)
    input_2D_aug[:, :, :, 0] *= -1
    input_2D_aug[:, :, joints_left + joints_right] = input_2D_aug[:,:, joints_right + joints_left]
    output_3D_non_flip, _ = model_trans(input_2D+joints_embed)
    output_3D_flip, _ = model_trans(input_2D_aug+joints_embed)
    output_3D_flip[:,:,:, 0] *= -1
    output_3D_flip[:, :, joints_left + joints_right] = output_3D_flip[:,:, joints_right + joints_left]
    output_3D = (output_3D_non_flip + output_3D_flip) / 2
    return output_3D




def fetch_actions(actions):
    out_poses_3d = []
    out_poses_2d = []

    for subject, action in actions:
        poses_2d = keypoints[subject][action]

        poses_3d = dataset[subject][action]['positions_3d']
        assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
        if seq2frame:
            for i in range(len(poses_3d)):  # Iterate across cameras
                out_poses_3d.append(poses_3d[i])
                out_poses_2d.append(poses_2d[i])


        else:
            for i in range(len(poses_3d)):  # Iterate across cameras
                ll_min = min(poses_3d[i].shape[0], poses_2d[i].shape[0])
                out_poses_3d.append(poses_3d[i][:ll_min])
                out_poses_2d.append(poses_2d[i][:ll_min])



    return out_poses_3d, out_poses_2d


if __name__ == '__main__':

    deco_n = 5  
    model = Transformer(deco_n=deco_n, d_model=256, dropout=0., drop_path_rate=0.1, length=args.number_of_frames)
    model = nn.DataParallel(model).cuda()

    if amass_pretrain:
        ckpt_dir = "/ckpts/gsformer_l_pretrain_40.5.pth.tar"
    else:
        ckpt_dir = "/ckpts/gsformer_l_41.58.pth.tar"

    ckpt = torch.load(ckpt_dir)
    model.load_state_dict(ckpt['state_dict'])

    # when evaluating with G-SFormer ckpt trained from scratch, you do not need to add "joints_embed" to 2D pose input
    J = 17
    joints_embed = torch.zeros(J, 2).cuda()
    position = torch.arange(0, J)
    joints_embed[:, 0] = torch.sin(position)
    joints_embed[:, 1] = torch.cos(position)
    joints_embed = joints_embed[None, None, :, :]

    with (torch.no_grad()):
        model.eval()
        action_loss_p1 = {}
        action_loss_p2 = {}
        total_loss_p1 = AccumLoss()
        total_loss_p2 = AccumLoss()

        bar_test = Bar('>>>', fill='>', max=len(all_actions.keys()))
        for action_key in all_actions.keys():
            test_loss_p1 = AccumLoss()
            test_loss_p2 = AccumLoss()
            infos = all_actions[action_key]
            out_poses_3d, out_poses_2d = fetch_actions(infos)

            if seq2frame:

                test_generator =  UnchunkedGenerator(batch_size=args.batch_size_test, cameras=None, poses_3d=out_poses_3d,
                                                    poses_2d=out_poses_2d,
                                                    pad=pad, augment=False, kps_left=kps_left,
                                                    kps_right=kps_right,
                                                    joints_left=joints_left, joints_right=joints_right, point=30)  #50

                for batch_test, (_, test_3d, test_2d) in enumerate(test_generator.next_epoch()):
                    test_3d = torch.from_numpy(test_3d.astype('float32')).cuda()  # N, 1, J, 3
                    test_2d = torch.from_numpy(test_2d.astype('float32')).cuda()  # N, T, J, 2
                    N_test = test_3d.shape[0]

                    output_3D_test = input_augmentation(test_2d, model, joints_left, joints_right, joints_embed)

                    test_3d[:, :, 0, :] = 0
                    output_3D_test[:, :, 0, :] = 0

                    loss_3d_test = mpjpe(output_3D_test, test_3d)
                    test_loss_p1.update(loss_3d_test.detach().cpu().numpy() * N_test, N_test)
                    total_loss_p1.update(loss_3d_test.detach().cpu().numpy() * N_test, N_test)

                    test_3d = test_3d.detach().cpu().numpy().reshape(-1, test_3d.shape[-2], test_3d.shape[-1])
                    output_3D_test = output_3D_test.detach().cpu().numpy().reshape(-1, test_3d.shape[-2], test_3d.shape[-1])
                    p2_loss_test = p_mpjpe(output_3D_test, test_3d)
                    test_loss_p2.update(p2_loss_test * N_test, N_test)
                    total_loss_p2.update(p2_loss_test * N_test, N_test)

            else:

                for (test_2d, test_3d) in zip(out_poses_2d, out_poses_3d):
                    test_2d, test_3d = eval_data_prepare_seq2seq(inputs_2d=test_2d, inputs_3d=test_3d)
                    test_3d = test_3d.cuda()
                    test_2d = test_2d.cuda()

                    N_test, T_t = test_3d.shape[:2]

                    output_3D_test = input_augmentation_seq2seq(test_2d, model, joints_left, joints_right, joints_embed)
                    test_3d[:, :, 0, :] = 0
                    output_3D_test[:, :, 0, :] = 0

                    loss_3d_test = mpjpe(output_3D_test, test_3d)
                    test_loss_p1.update(loss_3d_test.detach().cpu().numpy() * N_test * T_t, N_test * T_t)
                    total_loss_p1.update(loss_3d_test.detach().cpu().numpy() * N_test * T_t, N_test * T_t)

                    test_3d = test_3d.detach().cpu().numpy().reshape(-1, test_3d.shape[-2], test_3d.shape[-1])
                    output_3D_test = output_3D_test.detach().cpu().numpy().reshape(-1, test_3d.shape[-2],
                                                                                   test_3d.shape[-1])
                    p2_loss_test = p_mpjpe(output_3D_test, test_3d)
                    test_loss_p2.update(p2_loss_test * N_test * T_t, N_test * T_t)
                    total_loss_p2.update(p2_loss_test * N_test * T_t, N_test * T_t)

            p1 = round(test_loss_p1.avg, 4)
            p2 = round(test_loss_p2.avg, 4)
            action_loss_p1[action_key] = p1
            action_loss_p2[action_key] = p2
            print('\n')

            bar_test.suffix = 'action {}|test_loss_p1: {:.4f}, test_loss_p2: {:.4f}'.format(action_key,
                                                                                            test_loss_p1.avg,
                                                                                            test_loss_p2.avg)
            bar_test.next()

        bar_test.finish()

    action_avg_loss = 0.
    action_avg_loss_p2 = 0.
    for key in action_loss_p1.keys():
        action_avg_loss += round(action_loss_p1[key], 4)
        action_avg_loss_p2 += round(action_loss_p2[key], 4)
    action_avg_loss = round(action_avg_loss / len(action_loss_p1.keys()), 4)
    action_avg_loss_p2 = round(action_avg_loss_p2 / len(action_loss_p1.keys()), 4)

    msg = f'p1_loss {round(total_loss_p1.avg, 4)}, p2_loss {round(total_loss_p2.avg, 4)}, action_p1 {action_avg_loss}, action_p2 {action_avg_loss_p2}'
    print(msg)


