
import pickle
import torch
import numpy as np

# allowed_categories = ['mug', 'box', 'bowl', 'bottle', 'cylinder']
# split='test'
# path_to_grasps = 'data/grasps_tolerance/preprocessed_tight'
# quaternions = []
# translations = []
# labels = []
# metadata = []
# categories = []
# for cat in allowed_categories:
#     quaternions.append( np.load(f'{path_to_grasps}/{cat}/quaternions_{split}.npy') )
#     translations.append( np.load(f'{path_to_grasps}/{cat}/translations_{split}.npy') )
#     labels.append( np.load(f'{path_to_grasps}/{cat}/isaac_labels_{split}.npy') )
#     metadata.append( np.load(f'{path_to_grasps}/{cat}/metadata_{split}.npy') )
#     categories = categories + [cat]*len(metadata[-1])

# quaternions = torch.FloatTensor(np.concatenate(quaternions))
# translations = torch.FloatTensor(np.concatenate(translations))
# labels = np.concatenate(labels)
# labels = torch.FloatTensor(labels)
# metadata = np.concatenate(metadata)
# categories = categories

# size = len(labels)
# print(f'Total size: {size}')
# print(f'Positives: {torch.sum(labels)}')
# print(f'Negatives: {size-torch.sum(labels)}')



from scipy.spatial import transform
import torch
from datasets import GraspDatasetWithTolerance, GraspDataset, GraspDatasetTolerance
import utils
import trimesh
import numpy as np
from scipy.spatial.transform import Rotation as R


# test_dataset = GraspDatasetTolerance(path_to_pc='data/pcs4', split='test', augment=False)
# data = np.load('data/grasps_tolerance/mug/mug000_isaac/main_grasps.npz')
data = np.load('data/grasps_tolerance/cylinder/cylinder001_isaac/00000060.npz')

pc = pickle.load(open( 'data/pcs4/cylinder/cylinder001.pkl', 'rb') )[0]
print(pc.shape)

scene = trimesh.Scene()

pc_mesh = trimesh.points.PointCloud(vertices=pc)
scene.add_geometry(pc_mesh)
translations = data['translations']
quaternions = data['quaternions']
print(data.files)
labels = data['isaac_labels']
print(np.sum(labels))
for i in range(100):
    q,t,label = quaternions[i], translations[i], labels[i]
    print(f'label = {label}')


    transform = np.eye(4)
    transform[:3,3] = t
    transform[:3,:3] = R.from_quat(q).as_matrix()
    scene.add_geometry(utils.gripper_bd(label), transform=transform)
scene.show()

# 22908640