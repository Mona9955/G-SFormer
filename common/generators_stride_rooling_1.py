# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from itertools import zip_longest
import numpy as np
import random



class ChunkedGenerator:
    """
    Batched data generator, used for training.
    The sequences are split into equal-length chunks and padded as necessary.

    Arguments:
    batch_size -- the batch size to use for training
    cameras -- list of cameras, one element for each video (optional, used for semi-supervised training)
    poses_3d -- list of ground-truth 3D poses, one element for each video (optional, used for supervised training)
    poses_2d -- list of input 2D keypoints, one element for each video
    chunk_length -- number of output frames to predict for each training example (usually 1)
    pad -- 2D input padding to compensate for valid convolutions, per side (depends on the receptive field)
    causal_shift -- asymmetric padding offset when causal convolutions are used (usually 0 or "pad")
    shuffle -- randomly shuffle the dataset before each epoch
    random_seed -- initial seed to use for the random generator
    augment -- augment the dataset by flipping poses horizontally
    kps_left and kps_right -- list of left/right 2D keypoints if flipping is enabled
    joints_left and joints_right -- list of left/right 3D joints if flipping is enabled
    """

    def __init__(self, batch_size, cameras, poses_3d, poses_2d,
                 chunk_length, pad=0, causal_shift=0,
                 shuffle=True, random_seed=3407,
                 augment=False, kps_left=None, kps_right=None, joints_left=None, joints_right=None,
                 endless=False, out_all=False, tds=2, point=70):
        assert poses_3d is None or len(poses_3d) == len(poses_2d), (len(poses_3d), len(poses_2d))
        assert cameras is None or len(cameras) == len(poses_2d)

        # Build lineage info
        pairs = []  # (seq_idx, start_frame, end_frame, flip) tuples
        for i in range(len(poses_2d)):
            assert poses_3d is None or poses_3d[i].shape[0] == poses_3d[i].shape[0]
            n_chunks = (poses_2d[i].shape[0] + chunk_length - 1) // chunk_length
            offset = (n_chunks * chunk_length - poses_2d[i].shape[0]) // 2
            bounds = np.arange(n_chunks + 1) * chunk_length - offset
            augment_vector = np.full(len(bounds - 1), False, dtype=bool)
            pairs += zip(np.repeat(i, len(bounds - 1)), bounds[:-1], bounds[1:], augment_vector)
            if augment:
                pairs += zip(np.repeat(i, len(bounds - 1)), bounds[:-1], bounds[1:], ~augment_vector)

        # Initialize buffers
        if cameras is not None:
            self.batch_cam = np.empty((batch_size, cameras[0].shape[-1]))
        if not out_all:
            self.batch_3d = np.empty((batch_size, chunk_length, poses_3d[0].shape[-2], poses_3d[0].shape[-1]))

        else:
            self.batch_3d = np.empty((batch_size, chunk_length + 2 * pad, poses_3d[0].shape[-2], poses_3d[0].shape[-1]))

        self.batch_2d = np.empty((batch_size, chunk_length + 2 * pad, poses_2d[0].shape[-2], poses_2d[0].shape[-1]))

        self.num_batches = (len(pairs) + batch_size - 1) // batch_size
        self.batch_size = batch_size
        self.random = np.random.RandomState(random_seed)
        self.pairs = pairs
        self.shuffle = shuffle
        self.pad = pad
        self.causal_shift = causal_shift
        self.endless = endless
        self.state = None

        self.cameras = cameras
        self.poses_3d = poses_3d
        self.poses_2d = poses_2d

        self.augment = augment
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right
        self.out_all = out_all
        self.tds = tds
        self.point = point

    def num_frames(self):
        return len(self.pairs)

    def random_state(self):
        return self.random

    def set_random_state(self, random):
        self.random = random

    def augment_enabled(self):
        return self.augment

    def next_pairs(self):
        if self.state is None:
            if self.shuffle:
                pairs = self.random.permutation(self.pairs)
            else:
                pairs = self.pairs
            return 0, pairs
        else:
            return self.state

    def next_epoch(self):
        enabled = True
        while enabled:
            start_idx, pairs = self.next_pairs()
            for b_i in range(start_idx, self.num_batches):
                chunks = pairs[b_i * self.batch_size: (b_i + 1) * self.batch_size]
                for i, (seq_i, start_3d, end_3d, flip) in enumerate(chunks):
                    start_2d = start_3d - self.pad*self.tds - self.causal_shift
                    end_2d = end_3d + self.pad*self.tds - self.causal_shift
                    # 2D poses

                    seq_2d = self.poses_2d[seq_i]
                    seq_len = seq_2d.shape[0]
                    low_2d = max(start_2d, 0)
                    high_2d = min(end_2d, seq_len)
                    pad_left_2d = low_2d - start_2d
                    pad_right_2d = end_2d - high_2d

                    new_data = seq_2d[low_2d:high_2d]
                    point = self.point
                    # point = random.choice([70, 80, 90, 100])
                    if pad_left_2d != 0:
                        clip = min(start_3d, pad_left_2d)
                        if pad_left_2d <= point or clip == 0:
                            padding_left_2d = np.tile(seq_2d[0:1], [pad_left_2d, 1, 1])
                        else:
                            padding_left_2d =seq_2d[:clip]
                            if padding_left_2d.shape[0] < pad_left_2d:
                                padding_left_2d = np.concatenate((np.tile(seq_2d[0:1], [pad_left_2d-padding_left_2d.shape[0], 1, 1]), padding_left_2d), axis=0)

                        new_data = np.concatenate((padding_left_2d, new_data), axis=0)

                    if pad_right_2d != 0:
                        clip = min(seq_len - 1 - start_3d, pad_right_2d)
                        if pad_right_2d <= point or clip == 0:
                            padding_right_2d = np.tile(seq_2d[-1:],[pad_right_2d, 1, 1])
                        else:
                            padding_right_2d = seq_2d[-clip:]
                            if padding_right_2d.shape[0] < pad_right_2d:
                                padding_right_2d = np.concatenate((padding_right_2d, np.tile(seq_2d[-1:], [pad_right_2d-padding_right_2d.shape[0], 1, 1])), axis=0)

                        new_data = np.concatenate((new_data, padding_right_2d), axis=0)

                    self.batch_2d[i] = new_data[::self.tds]

                    if flip:
                        # Flip 2D keypoints
                        self.batch_2d[i, :, :, 0] *= -1
                        self.batch_2d[i, :, self.kps_left + self.kps_right] = self.batch_2d[i, :, self.kps_right + self.kps_left]
                    # 3D poses
                    if self.poses_3d is not None:
                        seq_3d = self.poses_3d[seq_i]
                        seq_len_3d = seq_3d.shape[0]
                        if self.out_all:
                            start_3d_1 = start_3d - self.pad*self.tds - self.causal_shift
                            end_3d_1 = end_3d + self.pad*self.tds - self.causal_shift

                            low_3d = max(start_3d_1, 0)
                            high_3d = min(end_3d_1, seq_len_3d)
                            pad_left_3d = low_3d - start_3d_1
                            pad_right_3d = end_3d_1 - high_3d

                            # low_3d = low_2d
                            # high_3d = high_2d
                            # pad_left_3d = pad_left_2d
                            # pad_right_3d = pad_right_2d

                        else:
                            low_3d = max(start_3d, 0)
                            high_3d = min(end_3d, seq_3d.shape[0])
                            pad_left_3d = low_3d - start_3d
                            pad_right_3d = end_3d - high_3d

                        new_3d = seq_3d[low_3d:high_3d]

                        if pad_left_3d != 0:

                            clip = min(start_3d, pad_left_3d)
                            if pad_left_3d <= point or clip == 0:
                                padding_left_3d = np.tile(seq_3d[0:1], [pad_left_3d, 1, 1])
                            else:
                                padding_left_3d = seq_3d[:clip]
                                if padding_left_3d.shape[0] < pad_left_3d:
                                    padding_left_3d = np.concatenate((np.tile(seq_3d[0:1], [pad_left_3d - padding_left_3d.shape[0], 1, 1]), padding_left_3d), axis=0)

                            new_3d = np.concatenate((padding_left_3d, new_3d), axis=0)

                        if pad_right_3d != 0:
                            clip = min(seq_len_3d - 1 - start_3d, pad_right_3d)
                            if pad_right_3d <= point or clip <= 0:
                                padding_right_3d = np.tile(seq_3d[-1:], [pad_right_3d, 1, 1])
                            else:
                                padding_right_3d = seq_3d[-clip:]
                                if padding_right_3d.shape[0] < pad_right_3d:
                                    padding_right_3d = np.concatenate((padding_right_3d, np.tile(seq_3d[-1:], [pad_right_3d - padding_right_3d.shape[0], 1, 1])), axis=0)

                            new_3d = np.concatenate((new_3d, padding_right_3d), axis=0)

                        self.batch_3d[i] = new_3d[::self.tds]
                        if flip:
                            # Flip 3D joints
                            self.batch_3d[i, :, :, 0] *= -1
                            self.batch_3d[i, :, self.joints_left + self.joints_right] = \
                                self.batch_3d[i, :, self.joints_right + self.joints_left]

                    # Cameras
                    if self.cameras is not None:
                        self.batch_cam[i] = self.cameras[seq_i]
                        if flip:
                            # Flip horizontal distortion coefficients
                            self.batch_cam[i, 2] *= -1
                            self.batch_cam[i, 7] *= -1

                if self.endless:
                    self.state = (b_i + 1, pairs)
                if self.poses_3d is None and self.cameras is None:
                    yield None, None, self.batch_2d[:len(chunks)]
                elif self.poses_3d is not None and self.cameras is None:
                    yield None, self.batch_3d[:len(chunks)], self.batch_2d[:len(chunks)]
                elif self.poses_3d is None:
                    yield self.batch_cam[:len(chunks)], None, self.batch_2d[:len(chunks)]
                else:
                    yield self.batch_cam[:len(chunks)], self.batch_3d[:len(chunks)], self.batch_2d[:len(chunks)]

            if self.endless:
                self.state = None
            else:
                enabled = False







class UnchunkedGenerator:
    """
    Non-batched data generator, used for testing.
    Sequences are returned one at a time (i.e. batch size = 1), without chunking.

    If data augmentation is enabled, the batches contain two sequences (i.e. batch size = 2),
    the second of which is a mirrored version of the first.

    Arguments:
    cameras -- list of cameras, one element for each video (optional, used for semi-supervised training)
    poses_3d -- list of ground-truth 3D poses, one element for each video (optional, used for supervised training)
    poses_2d -- list of input 2D keypoints, one element for each video
    pad -- 2D input padding to compensate for valid convolutions, per side (depends on the receptive field)
    causal_shift -- asymmetric padding offset when causal convolutions are used (usually 0 or "pad")
    augment -- augment the dataset by flipping poses horizontally
    kps_left and kps_right -- list of left/right 2D keypoints if flipping is enabled
    joints_left and joints_right -- list of left/right 3D joints if flipping is enabled
    """

    def __init__(self, batch_size, cameras, poses_3d, poses_2d, pad=0, causal_shift=0,
                 augment=False, kps_left=None, kps_right=None, joints_left=None, joints_right=None, tds=2, point=70):
        assert poses_3d is None or len(poses_3d) == len(poses_2d)
        assert cameras is None or len(cameras) == len(poses_2d)

        self.augment = False
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right

        self.pad = pad
        self.causal_shift = causal_shift
        self.cameras = cameras
        self.poses_3d = [] if poses_3d is None else poses_3d
        self.poses_2d = poses_2d
        self.tds = tds
        self.batch_size = batch_size
        self.point = point
        pairs = []  # (seq_idx, start_frame, end_frame, flip) tuples
        for i in range(len(poses_3d)):
            n_chunks = poses_3d[i].shape[0]
            bounds = np.arange(n_chunks + 1)
            pairs += zip(np.repeat(i, len(bounds - 1)), bounds[:-1], bounds[1:])

        self.num_batches = (len(pairs) + batch_size - 1) // batch_size
        self.pairs = pairs
        if cameras is not None:
            self.batch_cam = np.empty((batch_size, cameras[0].shape[-1]))

        self.batch_3d = np.empty((batch_size, 1, poses_3d[0].shape[-2], poses_3d[0].shape[-1]))

        self.batch_2d = np.empty((batch_size, 1 + 2 * pad, poses_2d[0].shape[-2], poses_2d[0].shape[-1]))

    def num_frames(self):
        count = 0
        for p in self.poses_2d:
            count += p.shape[0]
        return count

    def num_videos(self):
        return len(self.poses_2d)

    def augment_enabled(self):
        return self.augment

    def set_augment(self, augment):
        self.augment = augment

    def next_epoch(self):
        for b_i in range(0, self.num_batches):
            chunks = self.pairs[b_i * self.batch_size: (b_i + 1) * self.batch_size]
            for i, (seq_i, start_3d, end_3d) in enumerate(chunks):
                start_2d = start_3d - self.pad * self.tds - self.causal_shift
                end_2d = end_3d + self.pad * self.tds - self.causal_shift
                # 2D poses

                seq_2d = self.poses_2d[seq_i]
                seq_len = seq_2d.shape[0]
                low_2d = max(start_2d, 0)
                high_2d = min(end_2d, seq_len)
                pad_left_2d = low_2d - start_2d
                pad_right_2d = end_2d - high_2d

                new_data = seq_2d[low_2d:high_2d]
                point = self.point
                if pad_left_2d != 0:
                    clip = min(start_3d, pad_left_2d)
                    if pad_left_2d <= point or clip == 0:
                        padding_left_2d = np.tile(seq_2d[0:1], [pad_left_2d, 1, 1])
                    else:
                        padding_left_2d = seq_2d[:clip]
                        if padding_left_2d.shape[0] < pad_left_2d:
                            padding_left_2d = np.concatenate(
                                (np.tile(seq_2d[0:1], [pad_left_2d - padding_left_2d.shape[0], 1, 1]), padding_left_2d),
                                axis=0)

                    new_data = np.concatenate((padding_left_2d, new_data), axis=0)

                if pad_right_2d != 0:
                    clip = min(seq_len - 1 - start_3d, pad_right_2d)
                    if pad_right_2d <= point or clip == 0:
                        padding_right_2d = np.tile(seq_2d[-1:], [pad_right_2d, 1, 1])
                    else:
                        padding_right_2d = seq_2d[-clip:]
                        if padding_right_2d.shape[0] < pad_right_2d:
                            padding_right_2d = np.concatenate((padding_right_2d, np.tile(seq_2d[-1:], [
                                pad_right_2d - padding_right_2d.shape[0], 1, 1])), axis=0)

                    new_data = np.concatenate((new_data, padding_right_2d), axis=0)

                self.batch_2d[i] = new_data[::self.tds]

                if self.poses_3d is not None:
                    seq_3d = self.poses_3d[seq_i]
                    low_3d = max(start_3d, 0)
                    high_3d = min(end_3d, seq_3d.shape[0])

                    new_3d = seq_3d[low_3d:high_3d]
                    self.batch_3d[i] = new_3d
                if self.cameras is not None:
                    self.batch_cam[i] = self.cameras[seq_i]

            if self.cameras is None:

                yield None, self.batch_3d[:len(chunks)], self.batch_2d[:len(chunks)]
            else:
                yield self.batch_cam[:len(chunks)], self.batch_3d[:len(chunks)], self.batch_2d[:len(chunks)]








