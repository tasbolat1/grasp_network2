from scipy.spatial import transform
import torch
from datasets import GraspDatasetWithTolerance, GraspDataset, GraspDatasetTolerance
import utils
import trimesh
import numpy as np
from scipy.spatial.transform import Rotation as R


# test_dataset = GraspDatasetTolerance(path_to_pc='data/pcs4', split='test', augment=False)
test_dataset = GraspDataset(path_to_grasps = 'data/grasps4/preprocessed', path_to_pc='data/pcs4', split='test', augment=False)

q,t,pc,label,cat=test_dataset[6000] 

print(len(test_dataset))
print(f'label = {label}')
scene = trimesh.Scene()
pc_mesh = trimesh.points.PointCloud(vertices=pc)

transform = np.eye(4)
transform[:3,3] = t.numpy()
transform[:3,:3] = R.from_quat(q.numpy()).as_matrix()
scene.add_geometry(utils.gripper_bd(), transform=transform)
scene.add_geometry(pc_mesh)
scene.show()

# 22908640