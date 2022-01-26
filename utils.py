import numpy as np
from scipy.spatial.transform import Rotation as R
import math
import trimesh
import torch

def farthest_points(data,
                    nclusters,
                    dist_func,
                    return_center_indexes=False,
                    return_distances=False,
                    verbose=False):
    """
      Performs farthest point sampling on data points.
      Args:
        data: numpy array of the data points.
        nclusters: int, number of clusters.
        dist_dunc: distance function that is used to compare two data points.
        return_center_indexes: bool, If True, returns the indexes of the center of 
          clusters.
        return_distances: bool, If True, return distances of each point from centers.
      
      Returns clusters, [centers, distances]:
        clusters: numpy array containing the cluster index for each element in 
          data.
        centers: numpy array containing the integer index of each center.
        distances: numpy array of [npoints] that contains the closest distance of 
          each point to any of the cluster centers.
    """
    if nclusters >= data.shape[0]:
        if return_center_indexes:
            return np.arange(data.shape[0],
                             dtype=np.int32), np.arange(data.shape[0],
                                                        dtype=np.int32)

        return np.arange(data.shape[0], dtype=np.int32)

    clusters = np.ones((data.shape[0], ), dtype=np.int32) * -1
    distances = np.ones((data.shape[0], ), dtype=np.float32) * 1e7
    centers = []
    for iter in range(nclusters):
        index = np.argmax(distances)
        centers.append(index)
        shape = list(data.shape)
        for i in range(1, len(shape)):
            shape[i] = 1

        broadcasted_data = np.tile(np.expand_dims(data[index], 0), shape)
        new_distances = dist_func(broadcasted_data, data)
        distances = np.minimum(distances, new_distances)
        clusters[distances == new_distances] = iter
        if verbose:
            print('farthest points max distance : {}'.format(
                np.max(distances)))

    if return_center_indexes:
        if return_distances:
            return clusters, np.asarray(centers, dtype=np.int32), distances
        return clusters, np.asarray(centers, dtype=np.int32)

    return clusters


def distance_by_translation_grasp(p1, p2):
    """
      Gets two nx4x4 numpy arrays and computes the translation of all the
      grasps.
    """
    t1 = p1[:, :3, 3]
    t2 = p2[:, :3, 3]
    return np.sqrt(np.sum(np.square(t1 - t2), axis=-1))


def distance_by_translation_point(p1, p2):
    """
      Gets two nx3 points and computes the distance between point p1 and p2.
    """
    return np.sqrt(np.sum(np.square(p1 - p2), axis=-1))


def regularize_pc_point_count(pc, npoints, use_farthest_point=False):
    """
      If point cloud pc has less points than npoints, it oversamples.
      Otherwise, it downsample the input pc to have npoint points.
      use_farthest_point: indicates whether to use farthest point sampling
      to downsample the points. Farthest point sampling version runs slower.
    """
    if pc.shape[0] > npoints:
        if use_farthest_point:
            _, center_indexes = farthest_points(pc,
                                                npoints,
                                                distance_by_translation_point,
                                                return_center_indexes=True)
        else:
            center_indexes = np.random.choice(range(pc.shape[0]),
                                              size=npoints,
                                              replace=False)
        pc = pc[center_indexes, :]
    else:
        required = npoints - pc.shape[0]
        if required > 0:
            index = np.random.choice(range(pc.shape[0]), size=required)
            pc = np.concatenate((pc, pc[index, :]), axis=0)
    return pc

def partition_array_into_subarrays(array, sub_array_size):
    subarrays = []
    for i in range(0, math.ceil(array.shape[0] / sub_array_size)):
        subarrays.append(array[i * sub_array_size:(i + 1) * sub_array_size])
    return subarrays

def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    q = q.view(-1, 4)
    v = v.view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)

def get_mid_of_contact_points(grasp_cps):
  mid = (grasp_cps[:, 0, :] + grasp_cps[:, 1, :]) / 2.0
  return mid
def get_control_point_tensor(batch_size, use_torch=True, device="cpu"):
    """
      Outputs a tensor of shape (batch_size x 6 x 3).
      use_tf: switches between outputing a tensor and outputing a numpy array.
    """
    control_points = np.load('configs/panda.npy')[:, :3]
    control_points = [[0, 0, 0], [0, 0, 0], control_points[0, :],
                      control_points[1, :], control_points[-2, :],
                      control_points[-1, :]]
    control_points = np.asarray(control_points, dtype=np.float32)
    control_points = np.tile(np.expand_dims(control_points, 0),
                             [batch_size, 1, 1])

    if use_torch:
        return torch.tensor(control_points).to(device)

    return control_points


def transform_control_points(gt_grasps, batch_size, mode='qt', device="cpu"):
    """
      Transforms canonical points using gt_grasps.
      mode = 'qt' expects gt_grasps to have (batch_size x 7) where each 
        element is catenation of quaternion and translation for each
        grasps.
      mode = 'rt': expects to have shape (batch_size x 4 x 4) where
        each element is 4x4 transformation matrix of each grasp.
    """
    assert (mode == 'qt' or mode == 'rt'), mode
    grasp_shape = gt_grasps.shape
    if mode == 'qt':
        assert (len(grasp_shape) == 2), grasp_shape
        assert (grasp_shape[-1] == 7), grasp_shape
        control_points = get_control_point_tensor(batch_size, device=device)
        num_control_points = control_points.shape[1]
        input_gt_grasps = gt_grasps

        gt_grasps = torch.unsqueeze(input_gt_grasps,
                                    1).repeat(1, num_control_points, 1)

        gt_q = gt_grasps[:, :, :4]
        gt_t = gt_grasps[:, :, 4:]
        gt_control_points = qrot(gt_q, control_points)
        gt_control_points += gt_t

        return gt_control_points
    else:
        assert (len(grasp_shape) == 3), grasp_shape
        assert (grasp_shape[1] == 4 and grasp_shape[2] == 4), grasp_shape
        control_points = get_control_point_tensor(batch_size, device=device)
        shape = control_points.shape
        ones = torch.ones((shape[0], shape[1], 1), dtype=torch.float32)
        control_points = torch.cat((control_points, ones), -1)
        return torch.matmul(control_points, gt_grasps.permute(0, 2, 1))

def get_inlier_grasp_indices(grasp_list, query_point, threshold=1.0, device="cpu"):
    """This function returns all grasps whose distance between the mid of the finger tips and the query point is less than the threshold value. 
    
    Arguments:
        grasps are given as a list of [B,7] where B is the number of grasps and the other
        7 values represent teh quaternion and translation.
        query_point is a 1x3 point in 3D space.
        threshold represents the maximum distance between a grasp and the query_point
    """
    indices_to_keep = []
    for grasps in grasp_list:
        grasp_cps = transform_control_points(grasps,
                                             grasps.shape[0],
                                             device=device)
        mid_points = get_mid_of_contact_points(grasp_cps)
        dist = torch.norm(mid_points - query_point, 2, dim=-1)
        indices_to_keep.append(torch.where(dist <= threshold))
    return indices_to_keep

def gripper_bd(quality=None):
    gripper_line_points_main_part = np.array([
        [0.0401874312758446, -0.0000599553131906, 0.1055731475353241],
        [0.0401874312758446, -0.0000599553131906, 0.0672731392979622],
        [-0.0401874312758446, 0.0000599553131906, 0.0672731392979622],
        [-0.0401874312758446, 0.0000599553131906, 0.0726731419563293],
        [-0.0401874312758446, 0.0000599553131906, 0.1055731475353241],
    ])


    gripper_line_points_handle_part = np.array([
        [-0.0, 0.0000599553131906, 0.0672731392979622],
        [-0.0, 0.0000599553131906, -0.0032731392979622]
    ])
    

    if quality is not None:
        _B = quality*1.0
        _R = 1.0-_B
        _G = 0
        color = [_R, _G, _B, 1.0]
    else:
        color = None
    small_gripper_main_part = trimesh.path.Path3D(entities=[trimesh.path.entities.Line([0,1,2,3,4], color=color)],
                                                vertices = gripper_line_points_main_part)
    small_gripper_handle_part = trimesh.path.Path3D(entities=[trimesh.path.entities.Line([0, 1], color=color)],
                                                vertices = gripper_line_points_handle_part)
    small_gripper = trimesh.path.util.concatenate([small_gripper_main_part,
                                    small_gripper_handle_part])
    return small_gripper

# def get_transform_from_q_t(q,t):
#   all_transform = R.from_quat(q).as_matrix()
#   transforms = []
#   for i in range(q.shape[0]):
#     transform = np.eye(4)
#     transform[:3, :3] = all_transform[i]
#     transform[:3,3] = t[i]
#     transforms.append(transform)

#   transforms = np.array(transforms)
#   return transforms
    

