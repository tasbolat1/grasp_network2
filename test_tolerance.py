
import torch
import numpy as np

allowed_categories = ['mug', 'box', 'bowl', 'bottle', 'cylinder']
split='test'
path_to_grasps = 'data/grasps_tolerance/preprocessed_tight'
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

quaternions = torch.FloatTensor(np.concatenate(quaternions))
translations = torch.FloatTensor(np.concatenate(translations))
labels = np.concatenate(labels)
labels = torch.FloatTensor(labels)
metadata = np.concatenate(metadata)
categories = categories

size = len(labels)
print(f'Total size: {size}')
print(f'Positives: {torch.sum(labels)}')
print(f'Negatives: {size-torch.sum(labels)}')