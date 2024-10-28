import torch
from common.load_data_h36m_stride_rooling import Fusion
from common.h36m_dataset import Human36mDataset
from common.arguments import parse_args
from common.generators_stride_rooling_1 import ChunkedGenerator, UnchunkedGenerator
import numpy as np
from model_temporal_8_res import Transformer
import os
import copy
import time
import torch.optim as optim
import torch.nn as nn
from common.loss import *
from progress.bar import Bar
import math

eval_batch_num = 5000
joints_left = [4, 5, 6, 11, 12, 13]
joints_right = [1, 2, 3, 14, 15, 16]
reload = False
point = 50
args = parse_args()
pad = (args.number_of_frames - 1) // 2
subs_train = ['S1','S5','S6','S7','S8']
subs_test = ['S9', 'S11']
os.environ['CUDA_VISIBLE_DEVICES']='2,3'
time_tuple = time.localtime(time.time())
mark = '{}-{}-{}-{}-{}'.format(time_tuple[0], time_tuple[1], time_tuple[2], time_tuple[3], time_tuple[4])
mark1 = f'model_{point}_t8_res'
ckpt_path = f'/data1/G-SFormer/ckpt_1/{mark1}-{mark}/'
if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path, exist_ok=True)

best_acc = 100.


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


def fetch_actions(actions, keypoints, dataset):
    out_poses_3d = []
    out_poses_2d = []

    for subject, action in actions:
        poses_2d = keypoints[subject][action]
        for i in range(len(poses_2d)):  # Iterate across cameras
            out_poses_2d.append(poses_2d[i])

        poses_3d = dataset[subject][action]['positions_3d']
        assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
        for i in range(len(poses_3d)):  # Iterate across cameras
            out_poses_3d.append(poses_3d[i])

    return out_poses_3d, out_poses_2d


def warm_up_lr(step, factor=0.01, milestone=200, init_lr= args.learning_rate):
    alpha = step / milestone
    warm_up_factor = factor * (1-alpha) + alpha
    out_lr = warm_up_factor * init_lr
    return out_lr


def lr_decay_warmup(optimizer, lr_now, step):
    lr = warm_up_lr(step=step, init_lr=lr_now)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr



if __name__ == '__main__':
    print('Loading h36m dataset...')
    dataset_path = "/data2/cuimengmeng/processed_h36m/data_3d_h36m.npz"
    dataset = Human36mDataset(path=dataset_path)

    dataset_path_2d = "/data2/cuimengmeng/processed_h36m/data_2d_h36m_cpn_ft_h36m_dbb.npz"
    print('Loading 2D detections...')
    keypoints_2d = np.load(dataset_path_2d, allow_pickle=True)



    deco_n = 5
    model = Transformer(deco_n=deco_n, d_model=256, dropout=0.1, drop_path_rate=0.1, length=args.number_of_frames)
    model = nn.DataParallel(model).cuda()

    if reload:
        pre_ckpt_dir = ""
        pre_ckpt = torch.load(pre_ckpt_dir)['state_dict']
        model.load_state_dict(pre_ckpt, strict=True)
        print(f'{len(pre_ckpt.items())} pre-parameters reloaded.')


    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
    print('INFO: Trainable parameter count:', model_params)

    lr_now = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr_now, amsgrad=True)
    #optimizer = optim.Adam(filter(lambda p:p.requires_grad, model.parameters()), lr=lr_now, amsgrad=True)

    train_data = Fusion(args, dataset, keypoints_2d, train=True, point=point)
    train_data_loader = train_data.generator

    test_data = Fusion(args, dataset, keypoints_2d, train=False, point=point)
    keypoints = test_data.keypoints
    kps_right, kps_left = test_data.kps_right, test_data.kps_left
    test_data_loader = test_data.generator
    all_actions = {}

    J = 17
    joints_embed = torch.zeros(J, 2).cuda()
    position = torch.arange(0, J)
    joints_embed[:, 0] = torch.sin(position)
    joints_embed[:, 1] = torch.cos(position)
    joints_embed = joints_embed[None, None, :, :]

    for subject in subs_test:
        for action in dataset[subject].keys():
            anim = dataset[subject][action]
            action_name = action.split(' ')[0]
            if action_name not in all_actions:
                all_actions[action_name] = []

            all_actions[action_name].append((subject, action))

    epoch= 1
    eval_step = 0

    while epoch < args.epochs:
        train_loss = AccumLoss()
        train_loss_enco = AccumLoss()

        bar_train = Bar('>>>', fill='>', max=train_data_loader.num_batches)
        #if epoch > 1:
        lr_now = lr_decay(optimizer, lr_now, args.lr_decay_rate1)
        if (epoch + 1) % args.epoch_decay == 0:
            lr_now = lr_decay(optimizer, lr_now, args.lr_decay_rate)

        print('==========================')
        print('>>> epoch: {} | lr: {:.5f}'.format(epoch, lr_now))
        for batch_num, (cam_train, batch_3d, batch_2d) in enumerate(train_data_loader.next_epoch()):
            model.train()
            #if epoch == 1 and batch_num<=200:
               #lr_now = lr_decay_warmup(optimizer, 5.1334e-4, batch_num)

            cam_train = torch.from_numpy(cam_train.astype('float32')).cuda()
            inputs_3d = torch.from_numpy(batch_3d.astype('float32')).cuda()  #N,T,J,3
            inputs_2d = torch.from_numpy(batch_2d.astype('float32')).cuda()

            input_traj = inputs_3d[:,:,0:1,:].clone()
            inputs_3d[:,:,0,:] = 0
            target_3d = inputs_3d[:, pad:pad + 1, :, :]

            optimizer.zero_grad()
            pred_3d, pred_3d_enco = model(inputs_2d+joints_embed)   #N,1,j,3    N,T,j,3

            loss_3d = mpjpe(pred_3d, target_3d)
            loss_3d_enco = mpjpe(pred_3d_enco, inputs_3d)

            loss = loss_3d + loss_3d_enco
            loss.backward()
            optimizer.step()
            N, T = inputs_2d.shape[0:2]

            train_loss.update(loss_3d.detach().cpu().numpy() * N, N)

            train_loss_enco.update(loss_3d_enco.detach().cpu().numpy() * N * T, N * T)

            bar_train.suffix = '{}/{}|loss_now: {:.4f}, enco_loss: {:.4f}, avg_loss: {:.4f}, enco_avg_loss: {:.4f}'.format(batch_num , train_data_loader.num_batches,
                                                                                 loss_3d.detach().cpu().numpy(), loss_3d_enco.detach().cpu().numpy(), train_loss.avg, train_loss_enco.avg)
            bar_train.next()

            if epoch >= 15:
                eval_batch_num = 3000

            if (batch_num+1) % eval_batch_num == 0:
                epoch_loss_3d_pos = 0
                epoch_loss_3d_pos_procrustes = 0

                bar_test = Bar('>>>', fill='>', max=len(all_actions.keys()))
                model.eval()
                with torch.no_grad():

                    action_loss_p1 = {}
                    action_loss_p2 = {}
                    total_loss_p1 = AccumLoss()
                    total_loss_p2 = AccumLoss()
                    for action_key in all_actions.keys():
                        test_loss_p1 = AccumLoss()
                        test_loss_p2 = AccumLoss()
                        infos = all_actions[action_key]
                        out_poses_3d, out_poses_2d = fetch_actions(infos, dataset=dataset, keypoints=keypoints)

                        test_generator = UnchunkedGenerator(batch_size=args.batch_size_test, cameras=None, poses_3d=out_poses_3d, poses_2d=out_poses_2d,
                                                            pad=pad, augment=False, kps_left=kps_left,
                                                            kps_right=kps_right,
                                                            joints_left=joints_left, joints_right=joints_right, point=point)

                        for batch_test, (_, test_3d, test_2d) in enumerate(test_generator.next_epoch()):
                            test_3d = torch.from_numpy(test_3d.astype('float32')).cuda()
                            test_2d = torch.from_numpy(test_2d.astype('float32')).cuda()


                            N_test = test_3d.shape[0]
                            if args.test_time_augmentation:
                                output_3D_test = input_augmentation(test_2d, model, joints_left,
                                                                                      joints_right, joints_embed)

                            else:
                                output_3D_test, output_3D_enco_test = model(test_2d+joints_embed)

                            test_3d[:, :, 0, :] = 0

                            loss_3d_test = mpjpe(output_3D_test, test_3d)
                            test_loss_p1.update(loss_3d_test.detach().cpu().numpy() * N_test, N_test)
                            total_loss_p1.update(loss_3d_test.detach().cpu().numpy() * N_test, N_test)

                            test_3d = test_3d.detach().cpu().numpy().reshape(-1, test_3d.shape[-2], test_3d.shape[-1])
                            output_3D_test = output_3D_test.detach().cpu().numpy().reshape(-1, test_3d.shape[-2],
                                                                                           test_3d.shape[-1])
                            p2_loss_test = p_mpjpe(output_3D_test, test_3d)
                            test_loss_p2.update(p2_loss_test * N_test, N_test)
                            total_loss_p2.update(p2_loss_test * N_test, N_test)

                        p1 = round(test_loss_p1.avg, 4)
                        p2 = round(test_loss_p2.avg, 4)
                        action_loss_p1[action_key] = p1
                        action_loss_p2[action_key] = p2
                        print('\n')

                        bar_test.suffix = 'action {}|test_loss_p1: {:.4f}, test_loss_p2: {:.4f}'.format(action_key, test_loss_p1.avg, test_loss_p2.avg)
                        bar_test.next()

                    bar_test.finish()


                action_avg_loss = 0.
                action_avg_loss_p2 = 0.
                for key in action_loss_p1.keys():
                    action_avg_loss += action_loss_p1[key]
                    action_avg_loss_p2 += action_loss_p2[key]
                action_avg_loss = round(action_avg_loss / len(action_loss_p1.keys()), 4)
                action_avg_loss_p2 = round(action_avg_loss_p2 / len(action_loss_p1.keys()), 4)

                msg = f'ep {epoch}-bs {batch_num}-lr {lr_now}: p1_loss {round(total_loss_p1.avg, 4)}, p2_loss {round(total_loss_p2.avg, 4)}, action_p1 {action_avg_loss}, action_p2 {action_avg_loss_p2}'

                print('\n' + msg)
                with open(os.path.join(ckpt_path, 'msg.txt'), 'a') as f:
                    f.write(msg + '\n')

                if total_loss_p1.avg < best_acc:
                    best_acc = total_loss_p1.avg

                if total_loss_p1.avg < best_acc+0.2:
                    file_name = 'ep_'+str(epoch) +'_'+ 'bs_'+str(batch_num) +'_'+str(round(total_loss_p1.avg, 4)) + '.pth.tar'
                    save_ckpt({'epoch': epoch,
                               'lr': lr_now,
                               'all_p1': round(total_loss_p1.avg, 4),
                               'action_p1': action_avg_loss,
                               'state_dict': model.state_dict()},
                              ckpt_path=ckpt_path,
                              file_name=file_name)
        epoch += 1
        bar_train.finish()
















