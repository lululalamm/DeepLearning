import os
import numpy as np
import cv2
import math

# vitpose

# info points
# coco
    # point 0 : nose
    # point 1 : right eye
    # point 2 : left eye
    # point 3 : right ear
    # point 4 : left ear
    # point 5 : right shoulder
    # point 6 : left shoulder
    # point 7 : right elbow
    # point 8 : left elbow
    # point 9 : right wrist
    # point 10 : left wrist
    # point 11 : right waist
    # point 12 : left waist
    # point 13 : right knee
    # point 14 : left knee
    # point 15 : right ankle
    # point 16 : left ankle

FLIP_PAIRS ={
    'coco':[[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12],[13, 14], [15, 16]],
}

PALETTE = {'coco':np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                            [230, 230, 0], [255, 153, 255], [153, 204, 255],
                            [255, 102, 255], [255, 51, 255], [102, 178, 255],
                            [51, 153, 255], [255, 153, 153], [255, 102, 102],
                            [255, 51, 51], [153, 255, 153], [102, 255, 102],
                            [51, 255, 51], [0, 255, 0], [0, 0, 255],
                            [255, 0, 0], [255, 255, 255]]),
}

SKELETON={
    'coco':[[15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
            [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
            [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],[3, 5], [4, 6]],
}
LINK_COLOR={'coco':PALETTE['coco'][[0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16]],}
KPT_COLOR={'coco':PALETTE['coco'][[16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0]],}

NUM_JOINTS={'coco':17,}

def xyxy2xywh(bbox_xyxy):
    """Transform the bbox format from x1y1x2y2 to xywh.

    Args:
        bbox_xyxy (np.ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5). (left, top, right, bottom, [score])

    Returns:
        np.ndarray: Bounding boxes (with scores),
          shaped (n, 4) or (n, 5). (left, top, width, height, [score])
    """
    bbox_xywh = bbox_xyxy.copy()
    if len(bbox_xywh.shape)>1:
        bbox_xywh[:, 2] = bbox_xywh[:, 2] - bbox_xywh[:, 0] + 1
        bbox_xywh[:, 3] = bbox_xywh[:, 3] - bbox_xywh[:, 1] + 1
    else:
        bbox_xywh[2] = bbox_xywh[2] - bbox_xywh[0] + 1
        bbox_xywh[3] = bbox_xywh[3] - bbox_xywh[1] + 1

    return bbox_xywh


def box2cs(box,input_size): # input_size is (h,w)
    """This encodes bbox(x,y,w,h) into (center, scale)

    Args:
        x, y, w, h

    Returns:
        tuple: A tuple containing center and scale.

        - np.ndarray[float32](2,): Center of the bbox (x, y).
        - np.ndarray[float32](2,): Scale of the bbox w & h.
    """

    x, y, w, h = box[:4]
    aspect_ratio = input_size[1] / input_size[0]
    center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio

    # pixel std is 200.0
    scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)
    scale = scale * 1.25

    return center, scale

def normalization(rgb_img,mean_list=[0.485, 0.456, 0.406],std_list=[0.229, 0.224, 0.225]):
    MEAN = 255 * np.array(mean_list)
    STD = 255 * np.array(std_list)
    rgb_img = rgb_img.transpose(-1, 0, 1)
    norm_img = (rgb_img - MEAN[:, None, None]) / STD[:, None, None]
    
    return norm_img

def preprocessing(image,bboxes,scores,flip_pairs,image_size=[192, 256],num_joints=17,use_udp=True,box_format='xyxy'):
    if box_format=='xyxy':
        bboxes_xyxy = bboxes
        bboxes_xywh = xyxy2xywh(bboxes)
    else:
        bboxes_xywh = bboxes

    
    centers=[]
    scales=[]
    datas=[]
    
    for bi,box in enumerate(bboxes_xywh):
        center, scale = box2cs(box,(image_size[1],image_size[0]))
        centers.append(center)
        scales.append(scale)
        
        # prepare data
        data = {'center':center,'scale':scale,'bbox_score':scores[bi],
                 'bbox_id':bi,  # need to be assigned if batch_size > 1
                'bbox_xywh':box,
                'dataset':'test_data',
                'joints_3d':
                np.zeros((num_joints, 3), dtype=np.float32),
                'joints_3d_visible':
                np.zeros((num_joints, 3), dtype=np.float32),
                'rotation':
                0,
                'ann_info': {
                    'image_size': image_size,
                    'num_joints': num_joints,
                    'flip_pairs': flip_pairs
                    }
                }
        data['img']=image
        data = topdown_affine(data,image_size,use_udp)
        img = normalization(data['img'])
        img = np.expand_dims(img,axis=0)
        img = img.astype(np.float32)
        data['img']=img
        datas.append(data)
        
    return datas


def topdown_affine(data,image_size=[192,256],use_udp=True):
    image_size = np.array(image_size)
    joints_3d = data['joints_3d']
    joints_3d_visible = data['joints_3d_visible']
    c = data['center']
    s = data['scale']
    r = data['rotation']
    img = data['img']

    if use_udp:
        trans = get_warp_matrix(r, c * 2.0, image_size - 1.0, s * 200.0)
        if not isinstance(img, list):
            img = cv2.warpAffine(
                img,
                trans, (int(image_size[0]), int(image_size[1])),
                flags=cv2.INTER_LINEAR)
        else:
            img = [
                cv2.warpAffine(
                    i,
                    trans, (int(image_size[0]), int(image_size[1])),
                    flags=cv2.INTER_LINEAR) for i in img
            ]

        joints_3d[:, 0:2] = \
            warp_affine_joints(joints_3d[:, 0:2].copy(), trans)

    else:
        trans = get_affine_transform(c, s, r, image_size)
        if not isinstance(img, list):
            img = cv2.warpAffine(
                img,
                trans, (int(image_size[0]), int(image_size[1])),
                flags=cv2.INTER_LINEAR)
        else:
            img = [
                cv2.warpAffine(
                    i,
                    trans, (int(image_size[0]), int(image_size[1])),
                    flags=cv2.INTER_LINEAR) for i in img
            ]
        for i in range(data['num_joints']):
            if joints_3d_visible[i, 0] > 0.0:
                joints_3d[i,
                            0:2] = affine_transform(joints_3d[i, 0:2], trans)

    data['img'] = img
    data['joints_3d'] = joints_3d
    data['joints_3d_visible'] = joints_3d_visible
    
    return data

def transform_preds(coords, center, scale, output_size, use_udp=False):
    """Get final keypoint predictions from heatmaps and apply scaling and
    translation to map them back to the image.

    Note:
        num_keypoints: K

    Args:
        coords (np.ndarray[K, ndims]):

            * If ndims=2, corrds are predicted keypoint location.
            * If ndims=4, corrds are composed of (x, y, scores, tags)
            * If ndims=5, corrds are composed of (x, y, scores, tags,
              flipped_tags)

        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        output_size (np.ndarray[2, ] | list(2,)): Size of the
            destination heatmaps.
        use_udp (bool): Use unbiased data processing

    Returns:
        np.ndarray: Predicted coordinates in the images.
    """
    assert coords.shape[1] in (2, 4, 5)
    assert len(center) == 2
    assert len(scale) == 2
    assert len(output_size) == 2

    # Recover the scale which is normalized by a factor of 200.
    scale = scale * 200.0

    if use_udp:
        scale_x = scale[0] / (output_size[0] - 1.0)
        scale_y = scale[1] / (output_size[1] - 1.0)
    else:
        scale_x = scale[0] / output_size[0]
        scale_y = scale[1] / output_size[1]

    target_coords = np.ones_like(coords)
    target_coords[:, 0] = coords[:, 0] * scale_x + center[0] - scale[0] * 0.5
    target_coords[:, 1] = coords[:, 1] * scale_y + center[1] - scale[1] * 0.5

    return target_coords

def _get_max_preds(heatmaps):
    """Get keypoint predictions from score maps.

    Note:
        batch_size: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.

    Returns:
        tuple: A tuple containing aggregated results.

        - preds (np.ndarray[N, K, 2]): Predicted keypoint location.
        - maxvals (np.ndarray[N, K, 1]): Scores (confidence) of the keypoints.
    """
    assert isinstance(heatmaps,
                      np.ndarray), ('heatmaps should be numpy.ndarray')
    assert heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    N, K, _, W = heatmaps.shape
    heatmaps_reshaped = heatmaps.reshape((N, K, -1))
    idx = np.argmax(heatmaps_reshaped, 2).reshape((N, K, 1))
    maxvals = np.amax(heatmaps_reshaped, 2).reshape((N, K, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    preds[:, :, 0] = preds[:, :, 0] % W
    preds[:, :, 1] = preds[:, :, 1] // W

    preds = np.where(np.tile(maxvals, (1, 1, 2)) > 0.0, preds, -1)
    return preds, maxvals

def post_dark_udp(coords, batch_heatmaps, kernel=3):
    """DARK post-pocessing. Implemented by udp. Paper ref: Huang et al. The
    Devil is in the Details: Delving into Unbiased Data Processing for Human
    Pose Estimation (CVPR 2020). Zhang et al. Distribution-Aware Coordinate
    Representation for Human Pose Estimation (CVPR 2020).

    Note:
        - batch size: B
        - num keypoints: K
        - num persons: N
        - height of heatmaps: H
        - width of heatmaps: W

        B=1 for bottom_up paradigm where all persons share the same heatmap.
        B=N for top_down paradigm where each person has its own heatmaps.

    Args:
        coords (np.ndarray[N, K, 2]): Initial coordinates of human pose.
        batch_heatmaps (np.ndarray[B, K, H, W]): batch_heatmaps
        kernel (int): Gaussian kernel size (K) for modulation.

    Returns:
        np.ndarray([N, K, 2]): Refined coordinates.
    """
    if not isinstance(batch_heatmaps, np.ndarray):
        batch_heatmaps = batch_heatmaps.cpu().numpy()
    B, K, H, W = batch_heatmaps.shape
    N = coords.shape[0]
    assert (B == 1 or B == N)
    for heatmaps in batch_heatmaps:
        for heatmap in heatmaps:
            cv2.GaussianBlur(heatmap, (kernel, kernel), 0, heatmap)
    np.clip(batch_heatmaps, 0.001, 50, batch_heatmaps)
    np.log(batch_heatmaps, batch_heatmaps)

    batch_heatmaps_pad = np.pad(
        batch_heatmaps, ((0, 0), (0, 0), (1, 1), (1, 1)),
        mode='edge').flatten()

    index = coords[..., 0] + 1 + (coords[..., 1] + 1) * (W + 2)
    index += (W + 2) * (H + 2) * np.arange(0, B * K).reshape(-1, K)
    index = index.astype(int).reshape(-1, 1)
    i_ = batch_heatmaps_pad[index]
    ix1 = batch_heatmaps_pad[index + 1]
    iy1 = batch_heatmaps_pad[index + W + 2]
    ix1y1 = batch_heatmaps_pad[index + W + 3]
    ix1_y1_ = batch_heatmaps_pad[index - W - 3]
    ix1_ = batch_heatmaps_pad[index - 1]
    iy1_ = batch_heatmaps_pad[index - 2 - W]

    dx = 0.5 * (ix1 - ix1_)
    dy = 0.5 * (iy1 - iy1_)
    derivative = np.concatenate([dx, dy], axis=1)
    derivative = derivative.reshape(N, K, 2, 1)
    dxx = ix1 - 2 * i_ + ix1_
    dyy = iy1 - 2 * i_ + iy1_
    dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)
    hessian = np.concatenate([dxx, dxy, dxy, dyy], axis=1)
    hessian = hessian.reshape(N, K, 2, 2)
    hessian = np.linalg.inv(hessian + np.finfo(np.float32).eps * np.eye(2))
    coords -= np.einsum('ijmn,ijnk->ijmk', hessian, derivative).squeeze()
    return coords

def keypoints_from_heatmaps(heatmaps,
                            center,
                            scale,
                            unbiased=False,
                            post_process='default',
                            kernel=11,
                            valid_radius_factor=0.0546875,
                            use_udp=False,
                            target_type='GaussianHeatmap'):

    # Avoid being affected
    heatmaps = heatmaps.copy()

    # detect conflicts
    if unbiased:
        assert post_process not in [False, None, 'megvii']
    if post_process in ['megvii', 'unbiased']:
        assert kernel > 0
    if use_udp:
        assert not post_process == 'megvii'

    # normalize configs
    if post_process is False:
        warnings.warn(
            'post_process=False is deprecated, '
            'please use post_process=None instead', DeprecationWarning)
        post_process = None
    elif post_process is True:
        if unbiased is True:
            warnings.warn(
                'post_process=True, unbiased=True is deprecated,'
                " please use post_process='unbiased' instead",
                DeprecationWarning)
            post_process = 'unbiased'
        else:
            warnings.warn(
                'post_process=True, unbiased=False is deprecated, '
                "please use post_process='default' instead",
                DeprecationWarning)
            post_process = 'default'
    elif post_process == 'default':
        if unbiased is True:
            warnings.warn(
                'unbiased=True is deprecated, please use '
                "post_process='unbiased' instead", DeprecationWarning)
            post_process = 'unbiased'

    # start processing
    if post_process == 'megvii':
        heatmaps = _gaussian_blur(heatmaps, kernel=kernel)

    N, K, H, W = heatmaps.shape
    if use_udp:
        if target_type.lower() == 'GaussianHeatMap'.lower():
            preds, maxvals = _get_max_preds(heatmaps)
            preds = post_dark_udp(preds, heatmaps, kernel=kernel)
        elif target_type.lower() == 'CombinedTarget'.lower():
            for person_heatmaps in heatmaps:
                for i, heatmap in enumerate(person_heatmaps):
                    kt = 2 * kernel + 1 if i % 3 == 0 else kernel
                    cv2.GaussianBlur(heatmap, (kt, kt), 0, heatmap)
            # valid radius is in direct proportion to the height of heatmap.
            valid_radius = valid_radius_factor * H
            offset_x = heatmaps[:, 1::3, :].flatten() * valid_radius
            offset_y = heatmaps[:, 2::3, :].flatten() * valid_radius
            heatmaps = heatmaps[:, ::3, :]
            preds, maxvals = _get_max_preds(heatmaps)
            index = preds[..., 0] + preds[..., 1] * W
            index += W * H * np.arange(0, N * K / 3)
            index = index.astype(int).reshape(N, K // 3, 1)
            preds += np.concatenate((offset_x[index], offset_y[index]), axis=2)
        else:
            raise ValueError('target_type should be either '
                             "'GaussianHeatmap' or 'CombinedTarget'")
    else:
        preds, maxvals = _get_max_preds(heatmaps)
        if post_process == 'unbiased':  # alleviate biased coordinate
            # apply Gaussian distribution modulation.
            heatmaps = np.log(
                np.maximum(_gaussian_blur(heatmaps, kernel), 1e-10))
            for n in range(N):
                for k in range(K):
                    preds[n][k] = _taylor(heatmaps[n][k], preds[n][k])
        elif post_process is not None:
            # add +/-0.25 shift to the predicted locations for higher acc.
            for n in range(N):
                for k in range(K):
                    heatmap = heatmaps[n][k]
                    px = int(preds[n][k][0])
                    py = int(preds[n][k][1])
                    if 1 < px < W - 1 and 1 < py < H - 1:
                        diff = np.array([
                            heatmap[py][px + 1] - heatmap[py][px - 1],
                            heatmap[py + 1][px] - heatmap[py - 1][px]
                        ])
                        preds[n][k] += np.sign(diff) * .25
                        if post_process == 'megvii':
                            preds[n][k] += 0.5

    # Transform back to the image
    for i in range(N):
        preds[i] = transform_preds(
            preds[i], center[i], scale[i], [W, H], use_udp=use_udp)

    if post_process == 'megvii':
        maxvals = maxvals / 255.0 + 0.5

    return preds, maxvals

# pre
# def draw_pose(img,results,kpt_score_thr=0.3,radius=4,thickness=1,bbox_color='green',dataset='coco'):
#     dimg = img.copy()
#     img_h,img_w,_ = dimg.shape

#     for i in range(len(results)):
#         preds = results[i][0][0]
#         pscores = results[i][1][0]

#         kpts = np.array(preds, copy=False)
#         kpts_scores = np.array(pscores,copy=False)
        
#         # draw points
#         for kid, kpt in enumerate(kpts):
#             x_coord, y_coord = int(kpt[0]), int(kpt[1])
#             kpt_score = kpts_scores[kid]

#             if kpt_score > kpt_score_thr:
#                 color = tuple(int(c) for c in KPT_COLOR[dataset][kid])
#                 dimg = cv2.circle(dimg, (int(x_coord), int(y_coord)), radius,color, -1)

#         # draw links
#         for sk_id, sk in enumerate(SKELETON[dataset]):
#             pos1 = (int(kpts[sk[0], 0]), int(kpts[sk[0], 1]))
#             pos2 = (int(kpts[sk[1], 0]), int(kpts[sk[1], 1]))
#             if (pos1[0] > 0 and pos1[0] < img_w and pos1[1] > 0
#                     and pos1[1] < img_h and pos2[0] > 0 and pos2[0] < img_w
#                     and pos2[1] > 0 and pos2[1] < img_h
#                     and kpts_scores[sk[0]] > kpt_score_thr
#                     and kpts_scores[sk[1]] > kpt_score_thr):
#                 color = tuple(int(c) for c in LINK_COLOR[dataset][sk_id])

#                 dimg = cv2.line(dimg, pos1, pos2, color, thickness=thickness)

#     return dimg

def draw_pose(img,results,kpt_score_thr=0.3,radius=4,thickness=1,bbox_color='green',dataset='coco'): # results=[keypoints..., scores...]
    dimg = img.copy()
    img_h,img_w,_ = dimg.shape

    for i in range(len(results[0])):
        preds = results[0][i]
        pscores = results[1][i]

        kpts = np.array(preds, copy=False)
        kpts_scores = np.array(pscores,copy=False)
        
        # draw points
        for kid, kpt in enumerate(kpts):
            x_coord, y_coord = int(kpt[0]), int(kpt[1])
            kpt_score = kpts_scores[kid]

            if kpt_score > kpt_score_thr:
                color = tuple(int(c) for c in KPT_COLOR[dataset][kid])
                dimg = cv2.circle(dimg, (int(x_coord), int(y_coord)), radius,color, -1)

        # draw links
        for sk_id, sk in enumerate(SKELETON[dataset]):
            pos1 = (int(kpts[sk[0], 0]), int(kpts[sk[0], 1]))
            pos2 = (int(kpts[sk[1], 0]), int(kpts[sk[1], 1]))
            if (pos1[0] > 0 and pos1[0] < img_w and pos1[1] > 0
                    and pos1[1] < img_h and pos2[0] > 0 and pos2[0] < img_w
                    and pos2[1] > 0 and pos2[1] < img_h
                    and kpts_scores[sk[0]] > kpt_score_thr
                    and kpts_scores[sk[1]] > kpt_score_thr):
                color = tuple(int(c) for c in LINK_COLOR[dataset][sk_id])

                dimg = cv2.line(dimg, pos1, pos2, color, thickness=thickness)

    return dimg

def draw_pose_human(img,human,kpt_score_thr=0.3,radius=4,thickness=1,bbox_color='green',dataset='coco'):
    dimg = img.copy()
    img_h,img_w,_ = dimg.shape

    kps = human['pose']
    kpscore = human['pose_score']

    for i in range(len(kps)):
        kpts = np.array(kps[i], copy=False)
        kpts_scores = np.array(kpscore[i],copy=False)
        
        # draw points
        for kid, kpt in enumerate(kpts):
            x_coord, y_coord = int(kpt[0]), int(kpt[1])
            kpt_score = kpts_scores[kid]

            if kpt_score > kpt_score_thr:
                color = tuple(int(c) for c in KPT_COLOR[dataset][kid])
                dimg = cv2.circle(dimg, (int(x_coord), int(y_coord)), radius,color, -1)

        # draw links
        for sk_id, sk in enumerate(SKELETON[dataset]):
            pos1 = (int(kpts[sk[0], 0]), int(kpts[sk[0], 1]))
            pos2 = (int(kpts[sk[1], 0]), int(kpts[sk[1], 1]))
            if (pos1[0] > 0 and pos1[0] < img_w and pos1[1] > 0
                    and pos1[1] < img_h and pos2[0] > 0 and pos2[0] < img_w
                    and pos2[1] > 0 and pos2[1] < img_h
                    and kpts_scores[sk[0]] > kpt_score_thr
                    and kpts_scores[sk[1]] > kpt_score_thr):
                color = tuple(int(c) for c in LINK_COLOR[dataset][sk_id])

                dimg = cv2.line(dimg, pos1, pos2, color, thickness=thickness)

    return dimg

# topdown affine
def affine_transform(pt, trans_mat):
    """Apply an affine transformation to the points.

    Args:
        pt (np.ndarray): a 2 dimensional point to be transformed
        trans_mat (np.ndarray): 2x3 matrix of an affine transform

    Returns:
        np.ndarray: Transformed points.
    """
    assert len(pt) == 2
    new_pt = np.array(trans_mat) @ np.array([pt[0], pt[1], 1.])

    return new_pt

def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=(0., 0.),
                         inv=False):
    """Get the affine transform matrix, given the center/scale/rot/output_size.

    Args:
        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        rot (float): Rotation angle (degree).
        output_size (np.ndarray[2, ] | list(2,)): Size of the
            destination heatmaps.
        shift (0-100%): Shift translation ratio wrt the width/height.
            Default (0., 0.).
        inv (bool): Option to inverse the affine transform direction.
            (inv=False: src->dst or inv=True: dst->src)

    Returns:
        np.ndarray: The transform matrix.
    """
    assert len(center) == 2
    assert len(scale) == 2
    assert len(output_size) == 2
    assert len(shift) == 2

    # pixel_std is 200.
    scale_tmp = scale * 200.0

    shift = np.array(shift)
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = rotate_point([0., src_w * -0.5], rot_rad)
    dst_dir = np.array([0., dst_w * -0.5])

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def get_warp_matrix(theta, size_input, size_dst, size_target):
    """Calculate the transformation matrix under the constraint of unbiased.
    Paper ref: Huang et al. The Devil is in the Details: Delving into Unbiased
    Data Processing for Human Pose Estimation (CVPR 2020).

    Args:
        theta (float): Rotation angle in degrees.
        size_input (np.ndarray): Size of input image [w, h].
        size_dst (np.ndarray): Size of output image [w, h].
        size_target (np.ndarray): Size of ROI in input plane [w, h].

    Returns:
        np.ndarray: A matrix for transformation.
    """
    theta = np.deg2rad(theta)
    matrix = np.zeros((2, 3), dtype=np.float32)
    scale_x = size_dst[0] / size_target[0]
    scale_y = size_dst[1] / size_target[1]
    matrix[0, 0] = math.cos(theta) * scale_x
    matrix[0, 1] = -math.sin(theta) * scale_x
    matrix[0, 2] = scale_x * (-0.5 * size_input[0] * math.cos(theta) +
                              0.5 * size_input[1] * math.sin(theta) +
                              0.5 * size_target[0])
    matrix[1, 0] = math.sin(theta) * scale_y
    matrix[1, 1] = math.cos(theta) * scale_y
    matrix[1, 2] = scale_y * (-0.5 * size_input[0] * math.sin(theta) -
                              0.5 * size_input[1] * math.cos(theta) +
                              0.5 * size_target[1])
    return matrix

def warp_affine_joints(joints, mat):
    """Apply affine transformation defined by the transform matrix on the
    joints.

    Args:
        joints (np.ndarray[..., 2]): Origin coordinate of joints.
        mat (np.ndarray[3, 2]): The affine matrix.

    Returns:
        np.ndarray[..., 2]: Result coordinate of joints.
    """
    joints = np.array(joints)
    shape = joints.shape
    joints = joints.reshape(-1, 2)
    return np.dot(
        np.concatenate((joints, joints[:, 0:1] * 0 + 1), axis=1),
        mat.T).reshape(shape)