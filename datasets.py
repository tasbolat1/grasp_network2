from numpy import random
import torch
from torch.utils.data import Dataset
import numpy as np
import yaml
import pickle
from pathlib import Path
import utils
import models.quaternion as quat_ops
from numpy import genfromtxt

class GraspDataset(Dataset):
    def __init__(self, path_to_grasps = 'data/grasps4/preprocessed', path_to_pc='data/pcs4', split='train', allowed_categories = ['mug', 'box', 'bowl', 'bottle', 'cylinder'], augment=True, full_pc=False):

        self.noise_coeff=0.0035
        # load grasp_data
        # load grasp_data
        quaternions = []
        translations = []
        labels = []
        metadata = []
        categories = []
        for cat in allowed_categories:
            quaternions.append( np.load(f'{path_to_grasps}/{cat}/quaternions_{split}.npy') )
            translations.append( np.load(f'{path_to_grasps}/{cat}/translations_{split}.npy') )
            labels.append( np.load(f'{path_to_grasps}/{cat}/isaac_labels_{split}.npy') )
            metadata.append( np.load(f'{path_to_grasps}/{cat}/metadata_{split}.npy') )
            categories = categories + [cat]*len(metadata[-1])

        self.quaternions = torch.FloatTensor(np.concatenate(quaternions))
        self.translations = torch.FloatTensor(np.concatenate(translations))
        labels = np.concatenate(labels)
        self.labels = torch.FloatTensor(labels)
        self.metadata = np.concatenate(metadata)
        self.categories = categories

        self.pos_count = torch.sum(self.labels).item()
        self.pos_indcs = np.where(labels == 1)[0]
        self.neg_indcs = np.where(labels == 0)[0]
        self.total_count = len(self.labels)
        self.size = int(self.pos_count*2) # 1 to 1 POS to NEG ratio
        print(f'total {self.total_count}')

        # load pcs
        self.pcs = {}
        for cat in allowed_categories:
            self.pcs[cat] = [] 
            for k in range(21):
                try:
                    pc = pickle.load(open(f'{path_to_pc}/{cat}/{cat}{k:03}.pkl', 'rb'))
                    self.pcs[cat].append(pc)
                except:
                    pass

        # read uniform quaternions
        self.uniform_quaternions = genfromtxt('configs/data3_36864.qua', delimiter='\t')
        self.augment = augment
        self.full_pc = full_pc

    def __getitem__(self, index):
        # subsample negatives
        if index < self.pos_count:
            index = self.pos_indcs[index]
        else:
            index = np.random.randint(low=0, high=len(self.neg_indcs), size=1)[0]
            index = self.neg_indcs[index]

        # prepare pc
        cat = self.categories[index]
        obj_idx = int(self.metadata[index])
        pc_index = np.random.randint(low=0, high=999, size=1)[0]
        if self.full_pc:
            pc = self.pcs[cat][obj_idx][0:30]
            pc = np.concatenate(pc)
        else:
            pc = self.pcs[cat][obj_idx][pc_index]

        pc = utils.regularize_pc_point_count(pc, 1024, False)

        # jitter point cloud here
        sigma, clip = 0.0015, 0.007  # https://github.com/charlesq34/pointnet/blob/master/provider.py#L74
        pc = pc + np.clip(sigma * np.random.randn(*pc.shape), -1 * clip, clip).astype(np.float32)

        pc = torch.FloatTensor( pc )

        _quat = self.quaternions[index]
        _trans = self.translations[index]
        # augmentation
        if self.augment:
            pc, _quat, _trans = augment_grasp(pc, _quat, _trans, self.uniform_quaternions)

        return _quat,\
               _trans,\
               pc,\
               self.labels[index],\
               cat

    def __len__(self):
        return self.size


class GraspDatasetTolerance(Dataset):
    def __init__(self, path_to_grasps = 'data/grasps_tolerance/preprocessed_tight', path_to_pc='data/pcs4', split='train', allowed_categories = ['mug', 'box', 'bowl', 'bottle', 'cylinder'], augment=True, full_pc=False):

        self.noise_coeff=0.0035
        # load grasp_data
        # load grasp_data
        quaternions = []
        translations = []
        labels = []
        metadata = []
        categories = []
        for cat in allowed_categories:
            quaternions.append( np.load(f'{path_to_grasps}/{cat}/quaternions_{split}.npy') )
            translations.append( np.load(f'{path_to_grasps}/{cat}/translations_{split}.npy') )
            labels.append( np.load(f'{path_to_grasps}/{cat}/isaac_labels_{split}.npy') )
            metadata.append( np.load(f'{path_to_grasps}/{cat}/metadata_{split}.npy') )
            categories = categories + [cat]*len(metadata[-1])

        self.quaternions = torch.FloatTensor(np.concatenate(quaternions))
        self.translations = torch.FloatTensor(np.concatenate(translations))
        labels = np.concatenate(labels)
        self.labels = torch.FloatTensor(labels)
        self.metadata = np.concatenate(metadata)
        self.categories = categories

        self.size = len(self.labels)
        print(f'Total size: {self.size}')

        # load pcs
        self.pcs = {}
        for cat in allowed_categories:
            self.pcs[cat] = [] 
            for k in range(21):
                try:
                    pc = pickle.load(open(f'{path_to_pc}/{cat}/{cat}{k:03}.pkl', 'rb'))
                    self.pcs[cat].append(pc)
                except:
                    pass

        # read uniform quaternions
        self.uniform_quaternions = genfromtxt('configs/data3_36864.qua', delimiter='\t')
        self.augment = augment
        self.full_pc = full_pc

    def __getitem__(self, index):

        # prepare pc
        cat = self.categories[index]
        obj_idx = int(self.metadata[index])
        pc_index = np.random.randint(low=0, high=999, size=1)[0]
        if self.full_pc:
            pc = self.pcs[cat][obj_idx][0:30]
            pc = np.concatenate(pc)
        else:
            pc = self.pcs[cat][obj_idx][pc_index]

        pc = utils.regularize_pc_point_count(pc, 1024, False)

        # # add jitter
        sigma, clip = 0.0015, 0.007  # https://github.com/charlesq34/pointnet/blob/master/provider.py#L74
        pc = pc + np.clip(sigma * np.random.randn(*pc.shape), -1 * clip, clip).astype(np.float32)

        pc = torch.FloatTensor( pc )

        _quat = self.quaternions[index]
        _trans = self.translations[index]
        # augmentation
        if self.augment:
            pc, _quat, _trans = augment_grasp(pc, _quat, _trans, self.uniform_quaternions)

        return _quat,\
               _trans,\
               pc,\
               self.labels[index],\
               cat

    def __len__(self):
        return self.size


class GraspDatasetWithTolerance(Dataset):
    def __init__(self, path_to_grasps = 'data/preprocessed', path_to_pc='data/pcs4', split='train', allowed_categories = ['mug', 'box', 'bowl', 'bottle', 'cylinder'], augment=True, full_pc=False):

        self.ratio = 3
        path_to_grasps_original = 'data/grasps4/preprocessed'
        path_to_grasps_tolerance = 'data/grasps_tolerance/preprocessed_tight'
        # load grasp_data
        quaternions = []
        translations = []
        labels = []
        metadata = []
        categories = []

        for cat in allowed_categories:
            quaternions.append( np.load(f'{path_to_grasps_tolerance}/{cat}/quaternions_{split}.npy') )
            translations.append( np.load(f'{path_to_grasps_tolerance}/{cat}/translations_{split}.npy') )
            labels.append( np.load(f'{path_to_grasps_tolerance}/{cat}/isaac_labels_{split}.npy') )
            metadata.append( np.load(f'{path_to_grasps_tolerance}/{cat}/metadata_{split}.npy') )
            categories = categories + [cat]*len(metadata[-1])

        self.total_tolerance_size = len(categories)
        self.tolerance_pos = np.sum(np.concatenate(labels))

        print(f'Total tolerance size: {self.total_tolerance_size}.')
        print(f'Tolerance pos: {self.tolerance_pos}')
        print(f'Tolerance neg: {self.total_tolerance_size - self.tolerance_pos}')

        for cat in allowed_categories:
            quaternions.append( np.load(f'{path_to_grasps_original}/{cat}/quaternions_{split}.npy') )
            translations.append( np.load(f'{path_to_grasps_original}/{cat}/translations_{split}.npy') )
            labels.append( np.load(f'{path_to_grasps_original}/{cat}/isaac_labels_{split}.npy') )
            metadata.append( np.load(f'{path_to_grasps_original}/{cat}/metadata_{split}.npy') )
            categories = categories + [cat]*len(metadata[-1])


        self.quaternions = torch.FloatTensor(np.concatenate(quaternions))
        self.translations = torch.FloatTensor(np.concatenate(translations))
        labels = np.concatenate(labels)
        self.labels = torch.FloatTensor(labels)
        self.metadata = np.concatenate(metadata)
        self.categories = categories

        self.total_size = len(categories) # all data
        self.original_size = self.total_size - self.total_tolerance_size

        print(f'Total size: {self.total_size}')
        print(f'original size: {self.original_size}')

        print(f'Total positives: {np.sum(labels)}.')
 
        self.size = self.total_tolerance_size*self.ratio # 1 to 1 POS to NEG ratio

        # load pcs
        self.pcs = {}
        for cat in allowed_categories:
            self.pcs[cat] = [] 
            for k in range(21):
                try:
                    pc = pickle.load(open(f'{path_to_pc}/{cat}/{cat}{k:03}.pkl', 'rb'))
                    self.pcs[cat].append(pc)
                except:
                    pass

        # read uniform quaternions
        self.uniform_quaternions = genfromtxt('configs/data3_36864.qua', delimiter='\t')
        self.augment = augment
        self.full_pc = full_pc

    def __getitem__(self, index):

        if index > self.total_tolerance_size:
            index = np.random.randint(low=self.total_tolerance_size, high=self.total_size, size=1)[0]
            print(index)
        # prepare pc
        cat = self.categories[index]
        obj_idx = int(self.metadata[index])
        pc_index = np.random.randint(low=0, high=999, size=1)[0]
        if self.full_pc:
            pc = self.pcs[cat][obj_idx][0:30]
            pc = np.concatenate(pc)
        else:
            pc = self.pcs[cat][obj_idx][pc_index]

        # regularize point cloud -> N -> 1024
        pc = utils.regularize_pc_point_count(pc, 1024, False)

        # jitter point cloud here
        sigma, clip = 0.0015, 0.007  # https://github.com/charlesq34/pointnet/blob/master/provider.py#L74
        pc = pc + np.clip(sigma * np.random.randn(*pc.shape), -1 * clip, clip).astype(np.float32)


        pc = torch.FloatTensor( pc )

        _quat = self.quaternions[index]
        _trans = self.translations[index]
        # augmentation
        if self.augment:
            pc, _quat, _trans = augment_grasp(pc, _quat, _trans, self.uniform_quaternions)

        return _quat,\
               _trans,\
               pc,\
               self.labels[index],\
               cat

    def __len__(self):
        return self.size


def augment_grasp(pc, quaternion, translation, uniform_quaternions):
    '''
    pc: [n, 3]
    quaternion: [4]
    translation: [3]
    '''
    # sample random unit quaternion
    # rand_quat = torch.FloatTensor([[0.707,0.707,0,0.0]])
    rand_ind = np.random.randint(low=0, high=len(uniform_quaternions), size=1)[0]
    rand_quat = uniform_quaternions[rand_ind]
    rand_quat = torch.FloatTensor(rand_quat).unsqueeze(0)

    # rotate pc
    pc = pc.unsqueeze(0)
    rand_quat1 = rand_quat.unsqueeze(1).repeat([1,pc.shape[1], 1])
    #print(rand_quat1.shape)
    pc = quat_ops.rot_p_by_quaterion(pc, rand_quat1).squeeze()

    # rotate translation
    translation = translation.unsqueeze(0).unsqueeze(0)
    rand_quat2 = rand_quat.unsqueeze(1).repeat([1,1, 1])
    translation = quat_ops.rot_p_by_quaterion(translation, rand_quat2).squeeze()

    # rotate quaternion
    quaternion = quaternion.unsqueeze(0)
    quaternion = quat_ops.quaternion_mult(rand_quat, quaternion).squeeze()
    return pc, quaternion, translation