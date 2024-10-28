import torch.utils.data as data
import numpy as np

from common.utils import deterministic_random
from common.camera import world_to_camera, normalize_screen_coordinates
from common.generators_stride_rooling_1 import ChunkedGenerator, UnchunkedGenerator

class Fusion(data.Dataset):
    def __init__(self, opt, dataset, keypoints, train=True, point=70):
        self.data_type = opt.dataset
        self.train = train
        self.keypoints_name = opt.keypoints
        self.keypoints = keypoints

        self.train_list = opt.subjects_train.split(',')
        self.test_list = opt.subjects_test.split(',')
        self.action_filter = None if opt.actions == '*' else opt.actions.split(',')
        self.downsample = opt.downsample
        self.subset = opt.subset
        self.stride = opt.stride
        self.pad = (opt.number_of_frames-1)//2
        keypoints_metadata = self.keypoints['metadata'].item()
        keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
        self.kps_left, self.kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
        self.joints_left, self.joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
        self.opt = opt
        if self.train:
            self.keypoints = self.prepare_data(dataset, self.train_list)
            self.out_camera_params, self.out_poses_3d, self.out_poses_2d = self.fetch(dataset, self.train_list, subset=1)
            self.generator = ChunkedGenerator(batch_size=opt.batch_size//opt.stride, cameras=self.out_camera_params, poses_3d=self.out_poses_3d, poses_2d=self.out_poses_2d,
                                              chunk_length=1,pad=self.pad, augment=True, kps_left=self.kps_left, kps_right=self.kps_right,
                                              joints_left=self.joints_left,joints_right=self.joints_right, out_all=True, point=point)
            print('INFO: Training on {} samples'.format(self.generator.num_frames()))

        else:
            self.keypoints = self.prepare_data(dataset, self.test_list)
            self.out_camera_params,  self.out_poses_3d, self.out_poses_2d = self.fetch(dataset, self.test_list, subset=1)
            self.generator = UnchunkedGenerator(batch_size=opt.batch_size_test, cameras=self.out_camera_params, poses_3d=self.out_poses_3d, poses_2d=self.out_poses_2d,
                                              pad=self.pad, augment=False, kps_left=self.kps_left, kps_right=self.kps_right,
                                              joints_left=self.joints_left,joints_right=self.joints_right, point=point)

            print('INFO: Testing on {} samples'.format(self.generator.num_frames()))


            # batch_size, cameras, poses_3d, poses_2d,
            # chunk_length, pad = 0, causal_shift = 0,
            # shuffle = True, random_seed = 3407,
            # augment = False, kps_left = None, kps_right = None, joints_left = None, joints_right = None,
            # endless = False, out_all = False):

            # args.batch_size // args.stride, cameras_train, poses_train, poses_train_2d, args.stride,
            # pad = pad, causal_shift = causal_shift, shuffle = True, augment = args.data_augmentation,
            # kps_left = kps_left, kps_right = kps_right, joints_left = joints_left, joints_right = joints_right)

    def prepare_data(self, dataset, subs_list):

        for subject in subs_list:
            for action in dataset[subject].keys():
                anim = dataset[subject][action]
                positions_3d = []
                for cam in anim['cameras']:
                    pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                    pos_3d[:, 1:] -= pos_3d[:, :1]
                    positions_3d.append(pos_3d)
                anim['positions_3d'] = positions_3d

        keypoints = self.keypoints['positions_2d'].item()
        for subject in subs_list:
            for action in keypoints[subject].keys():
                for cam_idx, kps in enumerate(keypoints[subject][action]):
                    cam = dataset[subject][action]['cameras'][cam_idx]
                    kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
                    keypoints[subject][action][cam_idx] = kps

        return keypoints

    def fetch(self, dataset, subs_list, subset=1):
        out_poses_3d = []
        out_poses_2d = []
        out_camera_params = []
        for subject in subs_list:
            for action in self.keypoints[subject].keys():
                poses_2d = self.keypoints[subject][action]
                out_poses_2d.extend(poses_2d)

                if subject in dataset.cameras():
                    cams = dataset.cameras()[subject]
                    assert len(cams) == len(poses_2d), 'Camera count mismatch'
                    for cam in cams:
                        if 'intrinsic' in cam:
                            out_camera_params.append(cam['intrinsic'])

                # if 'positions_3d' in dataset[subject][action]:
                poses_3d = dataset[subject][action]['positions_3d']
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                out_poses_3d.extend(poses_3d)

        if len(out_camera_params) == 0:
            out_camera_params = None
        if len(out_poses_3d) == 0:
            out_poses_3d = None

        stride = self.downsample
        if subset < 1:
            for i in range(len(out_poses_2d)):
                n_frames = int(round(len(out_poses_2d[i]) // stride * subset) * stride)
                start = deterministic_random(0, len(out_poses_2d[i]) - n_frames + 1, str(len(out_poses_2d[i])))
                out_poses_2d[i] = out_poses_2d[i][start:start + n_frames:stride]
                if out_poses_3d is not None:
                    out_poses_3d[i] = out_poses_3d[i][start:start + n_frames:stride]
        elif stride > 1:
            # Downsample as requested
            for i in range(len(out_poses_2d)):
                out_poses_2d[i] = out_poses_2d[i][::stride]
                if out_poses_3d is not None:
                    out_poses_3d[i] = out_poses_3d[i][::stride]

        return out_camera_params, out_poses_3d, out_poses_2d










