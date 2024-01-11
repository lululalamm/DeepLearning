from math import sqrt
import pickle
import os
import numpy as np
import cv2
from math import cos, sin, atan2, asin, sqrt, dist

from ..data.constant import U_BASE, W_EXP_BASE, W_SHP_BASE


def parse_roi_box_from_bbox(bbox):
    left, top, right, bottom = bbox[:4]
    old_size = (right - left + bottom - top) / 2
    center_x = right - (right - left) / 2.0
    center_y = bottom - (bottom - top) / 2.0 + old_size * 0.14
    size = int(old_size * 1.58)

    roi_box = [0] * 4
    roi_box[0] = center_x - size / 2
    roi_box[1] = center_y - size / 2
    roi_box[2] = roi_box[0] + size
    roi_box[3] = roi_box[1] + size

    return roi_box

def crop_img(img, roi_box):
    h, w = img.shape[:2]

    sx, sy, ex, ey = [int(round(_)) for _ in roi_box]
    dh, dw = ey - sy, ex - sx
    if len(img.shape) == 3:
        res = np.zeros((dh, dw, 3), dtype=np.uint8)
    else:
        res = np.zeros((dh, dw), dtype=np.uint8)
    if sx < 0:
        sx, dsx = 0, -sx
    else:
        dsx = 0

    if ex > w:
        ex, dex = w, dw - (ex - w)
    else:
        dex = dw

    if sy < 0:
        sy, dsy = 0, -sy
    else:
        dsy = 0

    if ey > h:
        ey, dey = h, dh - (ey - h)
    else:
        dey = dh

    res[dsy:dey, dsx:dex] = img[sy:ey, sx:ex]
    return res

def _parse_param(param):
    """matrix pose form
    param: shape=(trans_dim+shape_dim+exp_dim,), i.e., 62 = 12 + 40 + 10
    """

    # pre-defined templates for parameter
    n = param.shape[0]
    if n == 62:
        trans_dim, shape_dim, exp_dim = 12, 40, 10
    elif n == 72:
        trans_dim, shape_dim, exp_dim = 12, 40, 20
    elif n == 141:
        trans_dim, shape_dim, exp_dim = 12, 100, 29
    else:
        raise Exception(f'Undefined templated param parsing rule')

    R_ = param[:trans_dim].reshape(3, -1)
    R = R_[:, :3]
    offset = R_[:, -1].reshape(3, 1)
    alpha_shp = param[trans_dim:trans_dim + shape_dim].reshape(-1, 1)
    alpha_exp = param[trans_dim + shape_dim:].reshape(-1, 1)

    return R, offset, alpha_shp, alpha_exp

def similar_transform(pts3d, roi_box, size):
    pts3d[0, :] -= 1  # for Python compatibility
    pts3d[2, :] -= 1
    pts3d[1, :] = size - pts3d[1, :]

    sx, sy, ex, ey = roi_box
    scale_x = (ex - sx) / size
    scale_y = (ey - sy) / size
    pts3d[0, :] = pts3d[0, :] * scale_x + sx
    pts3d[1, :] = pts3d[1, :] * scale_y + sy
    s = (scale_x + scale_y) / 2
    pts3d[2, :] *= s
    pts3d[2, :] -= np.min(pts3d[2, :])
    return np.array(pts3d, dtype=np.float32)


# one object
def recon_ver(param, roi_box,dense=False,size=120):
    size = size

    R, offset, alpha_shp, alpha_exp = _parse_param(param)
        
    pts3d = R @ (U_BASE + W_SHP_BASE @ alpha_shp + W_EXP_BASE @ alpha_exp). \
        reshape(3, -1, order='F') + offset
    pts3d = similar_transform(pts3d, roi_box, size)

    return pts3d

def recon_vers(param_lst, roi_box_lst,dense=False,size=120):
    size = size
    
    ver_lst = []
    for param, roi_box in zip(param_lst, roi_box_lst):
        R, offset, alpha_shp, alpha_exp = _parse_param(param)
        
        pts3d = R @ (U_BASE + W_SHP_BASE @ alpha_shp + W_EXP_BASE @ alpha_exp). \
            reshape(3, -1, order='F') + offset
        pts3d = similar_transform(pts3d, roi_box, size)

        ver_lst.append(pts3d)

    return ver_lst



def draw_landmark68(img_ori,ver_lst,save_path, save_flag=False):
    height, width = img_ori.shape[:2]
    plt.figure(figsize=(12, height / width * 12))
    plt.imshow(img_ori)#[..., ::-1])
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.axis('off')

    alpha = 0.8
    markersize = 4
    lw = 1.5
    color = 'w'
    markeredgecolor = 'black'

    nums = [0, 17, 22, 27, 31, 36, 42, 48, 60, 68]
    # 0:16 턱 / 17:21 왼쪽 눈썹 / 22:26 오른쪽 눈썹 / 27:30 콧대 / 31:35 코 아래 / 36:41 왼쪽 눈 / 42:47 오른쪽 눈 / 48:59 입 바깥 / 60:67 입 안쪽

    for ver in ver_lst:
        # close eyes and mouths
        plot_close = lambda i1, i2: plt.plot([ver[0, i1], ver[0, i2]], [ver[1, i1], ver[1, i2]],
                                                        color=color, lw=lw, alpha=alpha - 0.1)
        #plot_close(41, 36)
        #plot_close(47, 42)
        #plot_close(59, 48)
        #plot_close(67, 60)

        for ind in range(len(nums) - 1):
            l, r = nums[ind], nums[ind + 1]
            plt.plot(ver[0, l:r], ver[1, l:r], color=color, lw=lw, alpha=alpha - 0.1)
            plt.plot(ver[0, l:r], ver[1, l:r], marker='o', linestyle='None', markersize=markersize,
                        color=color,
                        markeredgecolor=markeredgecolor, alpha=alpha)

    if save_flag:
        plt.savefig(save_path)

#def draw_landmark5(img_ori,ver_lst,save_path, save_flag=False):

# 2d106det
def transform(data, center, output_size, scale, rotation):
    scale_ratio = scale
    rot = float(rotation) * np.pi / 180.0
    #translation = (output_size/2-center[0]*scale_ratio, output_size/2-center[1]*scale_ratio)
    t1 = trans.SimilarityTransform(scale=scale_ratio)
    cx = center[0] * scale_ratio
    cy = center[1] * scale_ratio
    t2 = trans.SimilarityTransform(translation=(-1 * cx, -1 * cy))
    t3 = trans.SimilarityTransform(rotation=rot)
    t4 = trans.SimilarityTransform(translation=(output_size / 2,
                                                output_size / 2))
    t = t1 + t2 + t3 + t4
    M = t.params[0:2]
    cropped = cv2.warpAffine(data,
                             M, (output_size, output_size),
                             borderValue=0.0)
    return cropped, M

def trans_points2d(pts, M):
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32)
        new_pt = np.dot(M, new_pt)
        #print('new_pt', new_pt.shape, new_pt)
        new_pts[i] = new_pt[0:2]

    return new_pts

# 3ddfa vis pose
def P2sRt(P):
    """ decompositing camera matrix P.
    Args:
        P: (3, 4). Affine Camera Matrix.
    Returns:
        s: scale factor.
        R: (3, 3). rotation matrix.
        t2d: (2,). 2d translation.
    """
    t3d = P[:, 3]
    R1 = P[0:1, :3]
    R2 = P[1:2, :3]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2.0
    r1 = R1 / np.linalg.norm(R1)
    r2 = R2 / np.linalg.norm(R2)
    r3 = np.cross(r1, r2)

    R = np.concatenate((r1, r2, r3), 0)
    return s, R, t3d


def matrix2angle(R):
    """ compute three Euler angles from a Rotation Matrix. Ref: http://www.gregslabaugh.net/publications/euler.pdf
    refined by: https://stackoverflow.com/questions/43364900/rotation-matrix-to-euler-angles-with-opencv
    todo: check and debug
     Args:
         R: (3,3). rotation matrix
     Returns:
         x: yaw
         y: pitch
         z: roll
     """
    if R[2, 0] > 0.998:
        z = 0
        x = np.pi / 2
        y = z + atan2(-R[0, 1], -R[0, 2])
    elif R[2, 0] < -0.998:
        z = 0
        x = -np.pi / 2
        y = -z + atan2(R[0, 1], R[0, 2])
    else:
        x = asin(R[2, 0])
        y = atan2(R[2, 1] / cos(x), R[2, 2] / cos(x))
        z = atan2(R[1, 0] / cos(x), R[0, 0] / cos(x))

    return x, y, z


def calc_pose(param):
    P = param[:12].reshape(3, -1)  # camera matrix
    s, R, t3d = P2sRt(P)
    P = np.concatenate((R, t3d.reshape(3, -1)), axis=1)  # without scale
    pose = matrix2angle(R)
    pose = [p * 180 / np.pi for p in pose]

    return P, pose

def calc_hypotenuse(pts):
    bbox = [min(pts[0, :]), min(pts[1, :]), max(pts[0, :]), max(pts[1, :])]
    center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    radius = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
    bbox = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]
    llength = sqrt((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2)
    return llength / 3

def build_camera_box(rear_size=90):
    point_3d = []
    rear_depth = 0
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))

    front_size = int(4 / 3 * rear_size)
    front_depth = int(4 / 3 * rear_size)
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=np.float32).reshape(-1, 3)

    return point_3d


def plot_pose_box(img, P, ver, color=(40, 255, 0), line_width=2,box_rate=0.6):
    """ Draw a 3D box as annotation of pose.
    Ref:https://github.com/yinguobing/head-pose-estimation/blob/master/pose_estimator.py
    Args:
        img: the input image
        P: (3, 4). Affine Camera Matrix.
        kpt: (2, 68) or (3, 68)
    """
    llength = calc_hypotenuse(ver)
    point_3d = build_camera_box(llength*box_rate)
    # Map to 2d image points
    point_3d_homo = np.hstack((point_3d, np.ones([point_3d.shape[0], 1])))  # n x 4
    point_2d = point_3d_homo.dot(P.T)[:, :2]

    point_2d[:, 1] = - point_2d[:, 1]
    point_2d[:, :2] = point_2d[:, :2] - np.mean(point_2d[:4, :2], 0) + np.mean(ver[:2, :27], 1)
    point_2d = np.int32(point_2d.reshape(-1, 2))

    # Draw all the lines
    #cv2.polylines(img, [point_2d], True, color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[0]), tuple(
        point_2d[1]), (0,100,0), line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[1]), tuple(
        point_2d[2]), (0,100,0), line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[2]), tuple(
        point_2d[3]), (0,100,0), line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[3]), tuple(
        point_2d[4]), (0,100,0), line_width, cv2.LINE_AA)
    
    
    cv2.line(img, tuple(point_2d[5]), tuple(
        point_2d[6]),color , line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[6]), tuple(
        point_2d[7]), color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[7]), tuple(
        point_2d[8]), color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[8]), tuple(
        point_2d[9]), color, line_width, cv2.LINE_AA)
    
    
    cv2.line(img, tuple(point_2d[1]), tuple(
        point_2d[6]), color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[2]), tuple(
        point_2d[7]), color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[3]), tuple(
        point_2d[8]), color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[4]), tuple(
        point_2d[5]), color, line_width, cv2.LINE_AA)

    return img



# pipnet

def crop_pip(image,bbox,target_size=256,scale_mul=1.1):
    
    image_height, image_width, _ = image.shape
    xmin,ymin,xmax,ymax = bbox
    width = xmax-xmin
    height = ymax-ymin
    
    scale = scale_mul
    
    xmin -= int((scale-1)/2*width)
    ymin -= int((scale-1)/2*height)
    width *= scale
    height *= scale
    width = int(width)
    height = int(height)
    xmin = max(xmin, 0)
    ymin = max(ymin, 0)
    width = min(width, image_width-xmin-1)
    height = min(height, image_height-ymin-1)

    xmax = xmin + width
    ymax = ymin + height
    #

    image_crop = image[int(ymin):int(ymax), int(xmin):int(xmax), :]
    image_crop = cv2.resize(image_crop, (target_size, target_size))
    
    tmp_add = [xmin,ymin]
    tmp_mul = [width,height]
    
    return image_crop, tmp_add,tmp_mul

def forward_pip_np(output,input_size, net_stride, num_nb):

    outputs_cls, outputs_x, outputs_y, outputs_nb_x, outputs_nb_y = output
    tmp_batch, tmp_channel, tmp_height, tmp_width = outputs_cls.shape
    assert tmp_batch == 1

    outputs_cls = outputs_cls.reshape(tmp_batch*tmp_channel, -1)
    max_ids = np.argmax(outputs_cls, 1)
    max_cls = np.max(outputs_cls, 1)[0]
    max_ids = max_ids.reshape(-1, 1)
    max_ids_nb = max_ids.repeat(num_nb,1).reshape(-1, 1)

    outputs_x = outputs_x.reshape(tmp_batch*tmp_channel, -1)
    outputs_x_select = np.take_along_axis(outputs_x,max_ids,1)
    outputs_x_select = outputs_x_select.squeeze(1)
    outputs_y = outputs_y.reshape(tmp_batch*tmp_channel, -1)
    outputs_y_select =  np.take_along_axis(outputs_y,max_ids,1)
    outputs_y_select = outputs_y_select.squeeze(1)

    outputs_nb_x = outputs_nb_x.reshape(tmp_batch*num_nb*tmp_channel, -1)
    outputs_nb_x_select =  np.take_along_axis(outputs_nb_x,max_ids_nb,1)
    outputs_nb_x_select = outputs_nb_x_select.squeeze(1).reshape(-1, num_nb)
    outputs_nb_y = outputs_nb_y.reshape(tmp_batch*num_nb*tmp_channel, -1)
    outputs_nb_y_select =  np.take_along_axis(outputs_nb_y,max_ids_nb,1)
    outputs_nb_y_select = outputs_nb_y_select.squeeze(1).reshape(-1, num_nb)

    tmp_x = (max_ids%tmp_width).reshape(-1,1).astype(np.float32)+outputs_x_select.reshape(-1,1)
    tmp_y = (max_ids//tmp_width).reshape(-1,1).astype(np.float32)+outputs_y_select.reshape(-1,1)
    tmp_x /= 1.0 * input_size / net_stride
    tmp_y /= 1.0 * input_size / net_stride

    tmp_nb_x = (max_ids%tmp_width).reshape(-1,1).astype(np.float32)+outputs_nb_x_select
    tmp_nb_y = (max_ids//tmp_width).reshape(-1,1).astype(np.float32)+outputs_nb_y_select
    tmp_nb_x = tmp_nb_x.reshape(-1, num_nb)
    tmp_nb_y = tmp_nb_y.reshape(-1, num_nb)
    tmp_nb_x /= 1.0 * input_size / net_stride
    tmp_nb_y /= 1.0 * input_size / net_stride

    return tmp_x, tmp_y, tmp_nb_x, tmp_nb_y, outputs_cls, max_cls

def get_meanface(meanface_file, num_nb):
    with open(meanface_file) as f:
        meanface = f.readlines()[0]
        
    meanface = meanface.strip().split()
    meanface = [float(x) for x in meanface]
    meanface = np.array(meanface).reshape(-1, 2)
    # each landmark predicts num_nb neighbors
    meanface_indices = []
    for i in range(meanface.shape[0]):
        pt = meanface[i,:]
        dists = np.sum(np.power(pt-meanface, 2), axis=1)
        indices = np.argsort(dists)
        meanface_indices.append(indices[1:1+num_nb])
    
    # each landmark predicted by X neighbors, X varies
    meanface_indices_reversed = {}
    for i in range(meanface.shape[0]):
        meanface_indices_reversed[i] = [[],[]]
    for i in range(meanface.shape[0]):
        for j in range(num_nb):
            meanface_indices_reversed[meanface_indices[i][j]][0].append(i)
            meanface_indices_reversed[meanface_indices[i][j]][1].append(j)
    
    max_len = 0
    for i in range(meanface.shape[0]):
        tmp_len = len(meanface_indices_reversed[i][0])
        if tmp_len > max_len:
            max_len = tmp_len
    
    # tricks, make them have equal length for efficient computation
    for i in range(meanface.shape[0]):
        tmp_len = len(meanface_indices_reversed[i][0])
        meanface_indices_reversed[i][0] += meanface_indices_reversed[i][0]*10
        meanface_indices_reversed[i][1] += meanface_indices_reversed[i][1]*10
        meanface_indices_reversed[i][0] = meanface_indices_reversed[i][0][:max_len]
        meanface_indices_reversed[i][1] = meanface_indices_reversed[i][1][:max_len]

    # make the indices 1-dim
    reverse_index1 = []
    reverse_index2 = []
    for i in range(meanface.shape[0]):
        reverse_index1 += meanface_indices_reversed[i][0]
        reverse_index2 += meanface_indices_reversed[i][1]
    return meanface_indices, reverse_index1, reverse_index2, max_len

def pip_get_point(lands,points):
    pxs = points[:2]
    pys = points[2:]
    
    if pxs[1]==-1:
        result_x = lands[pxs[0]][0]
    else:
        result_x = lands[pxs[0]][0]+(lands[pxs[1]][0]-lands[pxs[0]][0])/2
        
    if pys[1]==-1:
        result_y = lands[pys[0]][1]
    else:
        result_y = lands[pys[0]][1]+(lands[pys[1]][1]-lands[pys[0]][1])/2
        
    return [result_x, result_y]

def pip_get_5point(lms_num,lands,dict_to_5):

    
    land5=[]
    
    for k in dict_to_5[lms_num]:
        land5.append(pip_get_point(lands,dict_to_5[lms_num][k]))
            
            
    return land5

def get_distance_point(p1,p2):
    eye_dist = dist(p1,p2)
    return eye_dist