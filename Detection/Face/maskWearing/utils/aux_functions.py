# Author: aqeelanwar
# Created: 27 April,2020, 10:21 PM
# Email: aqeel.anwar@gatech.edu

from configparser import ConfigParser
import cv2, math, os
from PIL import Image, ImageDraw
from tqdm import tqdm
from utils.read_cfg import read_cfg
from utils.fit_ellipse import *
import random
from utils.create_mask import texture_the_mask, color_the_mask
from imutils import face_utils
import requests
from zipfile import ZipFile
from tqdm import tqdm
import bz2, shutil
import dlib


available_mask_color = [
    "#fc1c1a",
    "#177ABC",
    "#94B6D2",
    "#A5AB81",
    "#DD8047",
    "#6b425e",
    "#e26d5a",
    "#c92c48",
    "#6a506d",
    "#ffc900",
    "#ffffff",
    "#000000",
    "#49ff00",
]

available_mask_texture = [
    "masks/textures/check/check_1.png",
    "masks/textures/check/check_2.jpg",
    "masks/textures/check/check_3.png",
    "masks/textures/check/check_4.jpg",
    "masks/textures/check/check_5.jpg",
    "masks/textures/check/check_6.jpg",
    "masks/textures/check/check_7.jpg",
    "masks/textures/floral/floral_1.png",
    "masks/textures/floral/floral_2.jpg",
    "masks/textures/floral/floral_3.jpg",
    "masks/textures/floral/floral_4.jpg",
    "masks/textures/floral/floral_5.jpg",
    "masks/textures/floral/floral_6.jpg",
    "masks/textures/floral/floral_7.png",
    "masks/textures/floral/floral_8.png",
    "masks/textures/floral/floral_9.jpg",
    "masks/textures/floral/floral_10.png",
    "masks/textures/floral/floral_11.jpg",
    "masks/textures/floral/grey_petals.png",
    "masks/textures/fruits/bananas.png",
    "masks/textures/fruits/cherry.jpg",
    "masks/textures/fruits/cherry.png",
    "masks/textures/fruits/lemon.png",
    "masks/textures/fruits/pineapple.png",
    "masks/textures/fruits/strawberry.png",
    "masks/textures/others/heart_1.png",
    "masks/textures/others/polka.jpg",
]


def download_dlib_model():
    print_orderly("Get dlib model", 60)
    dlib_model_link = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    print("Downloading dlib model...")
    with requests.get(dlib_model_link, stream=True) as r:
        print("Zip file size: ", np.round(len(r.content) / 1024 / 1024, 2), "MB")
        destination = (
            "dlib_models" + os.path.sep + "shape_predictor_68_face_landmarks.dat.bz2"
        )
        if not os.path.exists(destination.rsplit(os.path.sep, 1)[0]):
            os.mkdir(destination.rsplit(os.path.sep, 1)[0])
        print("Saving dlib model...")
        with open(destination, "wb") as fd:
            for chunk in r.iter_content(chunk_size=32678):
                fd.write(chunk)
    print("Extracting dlib model...")
    with bz2.BZ2File(destination) as fr, open(
        "dlib_models/shape_predictor_68_face_landmarks.dat", "wb"
    ) as fw:
        shutil.copyfileobj(fr, fw)
    print("Saved: ", destination)
    print_orderly("done", 60)

    os.remove(destination)


def get_line(face_landmark, image, type="eye", debug=False):
    pil_image = Image.fromarray(image)
    d = ImageDraw.Draw(pil_image)
    left_eye = face_landmark["left_eye"]
    right_eye = face_landmark["right_eye"]
    left_eye_mid = np.mean(np.array(left_eye), axis=0)
    right_eye_mid = np.mean(np.array(right_eye), axis=0)
    eye_line_mid = (left_eye_mid + right_eye_mid) / 2

    if type == "eye":
        left_point = left_eye_mid
        right_point = right_eye_mid
        mid_point = eye_line_mid

    elif type == "nose_mid":
        nose_length = (
            face_landmark["nose_bridge"][-1][1] - face_landmark["nose_bridge"][0][1]
        )
        left_point = [left_eye_mid[0], left_eye_mid[1] + nose_length / 2]
        right_point = [right_eye_mid[0], right_eye_mid[1] + nose_length / 2]
        # mid_point = (
        #     face_landmark["nose_bridge"][-1][1] + face_landmark["nose_bridge"][0][1]
        # ) / 2

        mid_pointY = (
            face_landmark["nose_bridge"][-1][1] + face_landmark["nose_bridge"][0][1]
        ) / 2
        mid_pointX = (
            face_landmark["nose_bridge"][-1][0] + face_landmark["nose_bridge"][0][0]
        ) / 2
        mid_point = (mid_pointX, mid_pointY)

    elif type == "nose_tip":
        nose_length = (
            face_landmark["nose_bridge"][-1][1] - face_landmark["nose_bridge"][0][1]
        )
        left_point = [left_eye_mid[0], left_eye_mid[1] + nose_length]
        right_point = [right_eye_mid[0], right_eye_mid[1] + nose_length]
        mid_point = (
            face_landmark["nose_bridge"][-1][1] + face_landmark["nose_bridge"][0][1]
        ) / 2

    elif type == "bottom_lip":
        bottom_lip = face_landmark["bottom_lip"]
        bottom_lip_mid = np.max(np.array(bottom_lip), axis=0)
        shiftY = bottom_lip_mid[1] - eye_line_mid[1]
        left_point = [left_eye_mid[0], left_eye_mid[1] + shiftY]
        right_point = [right_eye_mid[0], right_eye_mid[1] + shiftY]
        mid_point = bottom_lip_mid

    elif type == "perp_line":
        bottom_lip = face_landmark["bottom_lip"]
        bottom_lip_mid = np.mean(np.array(bottom_lip), axis=0)

        left_point = eye_line_mid
        left_point = face_landmark["nose_bridge"][0]
        right_point = bottom_lip_mid

        mid_point = bottom_lip_mid

    elif type == "nose_long":
        nose_bridge = face_landmark["nose_bridge"]
        left_point = [nose_bridge[0][0], nose_bridge[0][1]]
        right_point = [nose_bridge[-1][0], nose_bridge[-1][1]]

        mid_point = left_point

    # d.line(eye_mid, width=5, fill='red')
    y = [left_point[1], right_point[1]]
    x = [left_point[0], right_point[0]]
    # cv2.imshow('h', image)
    # cv2.waitKey(0)
    eye_line = fit_line(x, y, image)
    d.line(eye_line, width=5, fill="blue")

    # Perpendicular Line
    # (midX, midY) and (midX - y2 + y1, midY + x2 - x1)
    y = [
        (left_point[1] + right_point[1]) / 2,
        (left_point[1] + right_point[1]) / 2 + right_point[0] - left_point[0],
    ]
    x = [
        (left_point[0] + right_point[0]) / 2,
        (left_point[0] + right_point[0]) / 2 - right_point[1] + left_point[1],
    ]
    perp_line = fit_line(x, y, image)
    if debug:
        d.line(perp_line, width=5, fill="red")
        pil_image.show()
    return eye_line, perp_line, left_point, right_point, mid_point


def get_points_on_chin(line, face_landmark, chin_type="chin"):
    chin = face_landmark[chin_type]
    points_on_chin = []
    for i in range(len(chin) - 1):
        chin_first_point = [chin[i][0], chin[i][1]]
        chin_second_point = [chin[i + 1][0], chin[i + 1][1]]

        flag, x, y = line_intersection(line, (chin_first_point, chin_second_point))
        if flag:
            points_on_chin.append((x, y))

    return points_on_chin


def plot_lines(face_line, image, debug=False):
    pil_image = Image.fromarray(image)
    if debug:
        d = ImageDraw.Draw(pil_image)
        d.line(face_line, width=4, fill="white")
        pil_image.show()


def line_intersection(line1, line2):
    # mid = int(len(line1) / 2)
    start = 0
    end = -1
    line1 = ([line1[start][0], line1[start][1]], [line1[end][0], line1[end][1]])

    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])
    x = []
    y = []
    flag = False

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return flag, x, y

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div

    segment_minX = min(line2[0][0], line2[1][0])
    segment_maxX = max(line2[0][0], line2[1][0])

    segment_minY = min(line2[0][1], line2[1][1])
    segment_maxY = max(line2[0][1], line2[1][1])

    if (
        segment_maxX + 1 >= x >= segment_minX - 1
        and segment_maxY + 1 >= y >= segment_minY - 1
    ):
        flag = True

    return flag, x, y


def fit_line(x, y, image):
    if x[0] == x[1]:
        x[0] += 0.1
    coefficients = np.polyfit(x, y, 1)
    polynomial = np.poly1d(coefficients)
    x_axis = np.linspace(0, image.shape[1], 50)
    y_axis = polynomial(x_axis)
    eye_line = []
    for i in range(len(x_axis)):
        eye_line.append((x_axis[i], y_axis[i]))

    return eye_line


def get_six_points(face_landmark, image):
    _, perp_line1, _, _, m = get_line(face_landmark, image, type="nose_mid")
    face_b = m

    perp_line, _, _, _, _ = get_line(face_landmark, image, type="perp_line")

    points1 = get_points_on_chin(perp_line1, face_landmark)
    points = get_points_on_chin(perp_line, face_landmark)

    if len(points1)<=0 and len(points)<=0:
        return None
    elif not points1:
        face_e = tuple(np.asarray(points[0]))
    elif not points:
        face_e = tuple(np.asarray(points1[0]))
    else:
        face_e = tuple((np.asarray(points[0]) + np.asarray(points1[0])) / 2)
    # face_e = points1[0]
    nose_mid_line, _, _, _, _ = get_line(face_landmark, image, type="nose_long")

    angle = get_angle(perp_line, nose_mid_line)
    # print("angle: ", angle)
    nose_mid_line, _, _, _, _ = get_line(face_landmark, image, type="nose_tip")
    points = get_points_on_chin(nose_mid_line, face_landmark)
    if len(points) < 2:
        face_landmark = get_face_ellipse(face_landmark)
        # print("extrapolating chin")
        points = get_points_on_chin(
            nose_mid_line, face_landmark, chin_type="chin_extrapolated"
        )
        if len(points) < 2:
            points = []
            points.append(face_landmark["chin"][0])
            points.append(face_landmark["chin"][-1])
    face_a = points[0]
    face_c = points[-1]
    # cv2.imshow('j', image)
    # cv2.waitKey(0)
    nose_mid_line, _, _, _, _ = get_line(face_landmark, image, type="bottom_lip")
    points = get_points_on_chin(nose_mid_line, face_landmark)
    if len(points)<=0:
        return None, None
    face_d = points[0]
    face_f = points[-1]

    six_points = np.float32([face_a, face_b, face_c, face_f, face_e, face_d])

    return six_points, angle


def get_angle(line1, line2):
    delta_y = line1[-1][1] - line1[0][1]
    delta_x = line1[-1][0] - line1[0][0]
    perp_angle = math.degrees(math.atan2(delta_y, delta_x))
    if delta_x < 0:
        perp_angle = perp_angle + 180
    if perp_angle < 0:
        perp_angle += 360
    if perp_angle > 180:
        perp_angle -= 180

    # print("perp", perp_angle)
    delta_y = line2[-1][1] - line2[0][1]
    delta_x = line2[-1][0] - line2[0][0]
    nose_angle = math.degrees(math.atan2(delta_y, delta_x))

    if delta_x < 0:
        nose_angle = nose_angle + 180
    if nose_angle < 0:
        nose_angle += 360
    if nose_angle > 180:
        nose_angle -= 180
    # print("nose", nose_angle)

    angle = nose_angle - perp_angle
    return angle


def mask_face(image, face_location, six_points, angle, args, type="surgical"):
    debug = False

    # Find the face angle
    threshold = 13
    if angle < -threshold:
        type += "_right"
    elif angle > threshold:
        type += "_left"

    face_height = face_location[2] - face_location[0]
    face_width = face_location[1] - face_location[3]
    # image = image_raw[
    #              face_location[0]-int(face_width/2): face_location[2]+int(face_width/2),
    #              face_location[3]-int(face_height/2): face_location[1]+int(face_height/2),
    #              :,
    #              ]
    # cv2.imshow('win', image)
    # cv2.waitKey(0)
    # Read appropriate mask image
    w = image.shape[0]
    h = image.shape[1]
    if not "empty" in type and not "inpaint" in type:
        cfg = read_cfg(config_filename="masks/masks.cfg", mask_type=type, verbose=False)
    else:
        if "left" in type:
            str = "surgical_blue_left"
        elif "right" in type:
            str = "surgical_blue_right"
        else:
            str = "surgical_blue"
        cfg = read_cfg(config_filename="masks/masks.cfg", mask_type=str, verbose=False)
    img = cv2.imread(cfg.template, cv2.IMREAD_UNCHANGED)

    # Process the mask if necessary
    if args.pattern:
        # Apply pattern to mask
        img = texture_the_mask(img, args.pattern, args.pattern_weight)

    if args.color:
        # Apply color to mask
        img = color_the_mask(img, args.color, args.color_weight)

    mask_line = np.float32(
        [cfg.mask_a, cfg.mask_b, cfg.mask_c, cfg.mask_f, cfg.mask_e, cfg.mask_d]
    )
    # Warp the mask
    M, mask = cv2.findHomography(mask_line, six_points)
    dst_mask = cv2.warpPerspective(img, M, (h, w))
    dst_mask_points = cv2.perspectiveTransform(mask_line.reshape(-1, 1, 2), M)
    mask = dst_mask[:, :, 3]
    face_height = face_location[2] - face_location[0]
    face_width = face_location[1] - face_location[3]
    image_face = image[
        face_location[0] + int(face_height / 2) : face_location[2],
        face_location[3] : face_location[1],
        :,
    ]

    image_face = image

    # Adjust Brightness
    mask_brightness = get_avg_brightness(img)
    img_brightness = get_avg_brightness(image_face)
    delta_b = 1 + (img_brightness - mask_brightness) / 255
    dst_mask = change_brightness(dst_mask, delta_b)

    # Adjust Saturation
    mask_saturation = get_avg_saturation(img)
    img_saturation = get_avg_saturation(image_face)
    delta_s = 1 - (img_saturation - mask_saturation) / 255
    dst_mask = change_saturation(dst_mask, delta_s)

    # Apply mask
    mask_inv = cv2.bitwise_not(mask)
    img_bg = cv2.bitwise_and(image, image, mask=mask_inv)
    img_fg = cv2.bitwise_and(dst_mask, dst_mask, mask=mask)
    out_img = cv2.add(img_bg, img_fg[:, :, 0:3])
    if "empty" in type or "inpaint" in type:
        out_img = img_bg
    # Plot key points

    if "inpaint" in type:
        out_img = cv2.inpaint(out_img, mask, 3, cv2.INPAINT_TELEA)
        # dst_NS = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)

    if debug:
        for i in six_points:
            cv2.circle(out_img, (i[0], i[1]), radius=4, color=(0, 0, 255), thickness=-1)

        for i in dst_mask_points:
            cv2.circle(
                out_img, (i[0][0], i[0][1]), radius=4, color=(0, 255, 0), thickness=-1
            )

    return out_img, mask


def draw_landmarks(face_landmarks, image):
    pil_image = Image.fromarray(image)
    d = ImageDraw.Draw(pil_image)
    for facial_feature in face_landmarks.keys():
        d.line(face_landmarks[facial_feature], width=5, fill="white")
    pil_image.show()


def get_face_ellipse(face_landmark):
    chin = face_landmark["chin"]
    x = []
    y = []
    for point in chin:
        x.append(point[0])
        y.append(point[1])

    x = np.asarray(x)
    y = np.asarray(y)

    a = fitEllipse(x, y)
    center = ellipse_center(a)
    phi = ellipse_angle_of_rotation(a)
    axes = ellipse_axis_length(a)
    a, b = axes

    arc = 2.2
    R = np.arange(0, arc * np.pi, 0.2)
    xx = center[0] + a * np.cos(R) * np.cos(phi) - b * np.sin(R) * np.sin(phi)
    yy = center[1] + a * np.cos(R) * np.sin(phi) + b * np.sin(R) * np.cos(phi)
    chin_extrapolated = []
    for i in range(len(R)):
        chin_extrapolated.append((xx[i], yy[i]))
    face_landmark["chin_extrapolated"] = chin_extrapolated
    return face_landmark


def get_avg_brightness(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)
    return np.mean(v)


def get_avg_saturation(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)
    return np.mean(v)


def change_brightness(img, value=1.0):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)
    v = value * v
    v[v > 255] = 255
    v = np.asarray(v, dtype=np.uint8)
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def change_saturation(img, value=1.0):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)
    s = value * s
    s[s > 255] = 255
    s = np.asarray(s, dtype=np.uint8)
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def check_path(path):
    is_directory = False
    is_file = False
    is_other = False
    if os.path.isdir(path):
        is_directory = True
    elif os.path.isfile(path):
        is_file = True
    else:
        is_other = True

    return is_directory, is_file, is_other


def shape_to_landmarks(shape):
    face_landmarks = {}
    face_landmarks["left_eyebrow"] = [
        tuple([shape[0][17], shape[1][17]]), # tuple(shape[17]),
        tuple([shape[0][18], shape[1][18]]), # tuple(shape[18]),
        tuple([shape[0][19], shape[1][19]]), # tuple(shape[19]),
        tuple([shape[0][20], shape[1][20]]), # tuple(shape[20]),
        tuple([shape[0][21], shape[1][21]]), # tuple(shape[21]),
    ]
    face_landmarks["right_eyebrow"] = [
        tuple([shape[0][22], shape[1][22]]), # tuple(shape[22]),
        tuple([shape[0][23], shape[1][23]]), # tuple(shape[23]),
        tuple([shape[0][24], shape[1][24]]), # tuple(shape[24]),
        tuple([shape[0][25], shape[1][25]]), # tuple(shape[25]),
        tuple([shape[0][26], shape[1][26]]), # tuple(shape[26]),
    ]
    face_landmarks["nose_bridge"] = [
        tuple([shape[0][27], shape[1][27]]), # tuple(shape[27]),
        tuple([shape[0][28], shape[1][28]]), # tuple(shape[28]),
        tuple([shape[0][29], shape[1][29]]), # tuple(shape[29]),
        tuple([shape[0][30], shape[1][30]]), # tuple(shape[30]),
    ]
    face_landmarks["nose_tip"] = [
        tuple([shape[0][31], shape[1][31]]), # tuple(shape[31]),
        tuple([shape[0][32], shape[1][32]]), # tuple(shape[32]),
        tuple([shape[0][33], shape[1][33]]), # tuple(shape[33]),
        tuple([shape[0][34], shape[1][34]]), # tuple(shape[34]),
        tuple([shape[0][35], shape[1][35]]), # tuple(shape[35]),
    ]
    face_landmarks["left_eye"] = [
        tuple([shape[0][36], shape[1][36]]), # tuple(shape[36]),
        tuple([shape[0][37], shape[1][37]]), # tuple(shape[37]),
        tuple([shape[0][38], shape[1][38]]), # tuple(shape[38]),
        tuple([shape[0][39], shape[1][39]]), # tuple(shape[39]),
        tuple([shape[0][40], shape[1][40]]), # tuple(shape[40]),
        tuple([shape[0][41], shape[1][41]]), # tuple(shape[41]),
    ]
    face_landmarks["right_eye"] = [
        tuple([shape[0][42], shape[1][42]]), # tuple(shape[42]),
        tuple([shape[0][43], shape[1][43]]), # tuple(shape[43]),
        tuple([shape[0][44], shape[1][44]]), # tuple(shape[44]),
        tuple([shape[0][45], shape[1][45]]), # tuple(shape[45]),
        tuple([shape[0][46], shape[1][46]]), # tuple(shape[46]),
        tuple([shape[0][47], shape[1][47]]), # tuple(shape[47]),
    ]
    face_landmarks["top_lip"] = [
        tuple([shape[0][48], shape[1][48]]), # tuple(shape[48]),
        tuple([shape[0][49], shape[1][49]]), # tuple(shape[49]),
        tuple([shape[0][50], shape[1][50]]), # tuple(shape[50]),
        tuple([shape[0][51], shape[1][51]]), # tuple(shape[51]),
        tuple([shape[0][52], shape[1][52]]), # tuple(shape[52]),
        tuple([shape[0][53], shape[1][53]]), # tuple(shape[53]),
        tuple([shape[0][54], shape[1][54]]), # tuple(shape[54]),
        tuple([shape[0][60], shape[1][60]]), # tuple(shape[60]),
        tuple([shape[0][61], shape[1][61]]), # tuple(shape[61]),
        tuple([shape[0][62], shape[1][62]]), # tuple(shape[62]),
        tuple([shape[0][63], shape[1][63]]), # tuple(shape[63]),
        tuple([shape[0][64], shape[1][64]]), # tuple(shape[64]),
    ]

    face_landmarks["bottom_lip"] = [
        tuple([shape[0][54], shape[1][54]]), # tuple(shape[54]),
        tuple([shape[0][55], shape[1][55]]), # tuple(shape[55]),
        tuple([shape[0][56], shape[1][56]]), # tuple(shape[56]),
        tuple([shape[0][57], shape[1][57]]), # tuple(shape[57]),
        tuple([shape[0][58], shape[1][58]]), # tuple(shape[58]),
        tuple([shape[0][59], shape[1][59]]), # tuple(shape[59]),
        tuple([shape[0][48], shape[1][48]]), # tuple(shape[48]),
        tuple([shape[0][64], shape[1][64]]), # tuple(shape[64]),
        tuple([shape[0][65], shape[1][65]]), # tuple(shape[65]),
        tuple([shape[0][66], shape[1][66]]), # tuple(shape[66]),
        tuple([shape[0][67], shape[1][67]]), # tuple(shape[67]),
        tuple([shape[0][60], shape[1][60]]), # tuple(shape[60]),
    ]

    face_landmarks["chin"] = [
        tuple([shape[0][0], shape[1][0]]), # tuple(shape[0]),
        tuple([shape[0][1], shape[1][1]]), # tuple(shape[1]),
        tuple([shape[0][2], shape[1][2]]), # tuple(shape[2]),
        tuple([shape[0][3], shape[1][3]]), # tuple(shape[3]),
        tuple([shape[0][4], shape[1][4]]), # tuple(shape[4]),
        tuple([shape[0][5], shape[1][5]]), # tuple(shape[5]),
        tuple([shape[0][6], shape[1][6]]), # tuple(shape[6]),
        tuple([shape[0][7], shape[1][7]]), # tuple(shape[7]),
        tuple([shape[0][8], shape[1][8]]), # tuple(shape[8]),
        tuple([shape[0][9], shape[1][9]]), # tuple(shape[9]),
        tuple([shape[0][10], shape[1][10]]), # tuple(shape[10]),
        tuple([shape[0][11], shape[1][11]]), # tuple(shape[11]),
        tuple([shape[0][12], shape[1][12]]), # tuple(shape[12]),
        tuple([shape[0][13], shape[1][13]]), # tuple(shape[13]),
        tuple([shape[0][14], shape[1][14]]), # tuple(shape[14]),
        tuple([shape[0][15], shape[1][15]]), # tuple(shape[15]),
        tuple([shape[0][16], shape[1][16]]), # tuple(shape[16]),
    ]
    return face_landmarks

def shape_to_landmarks_2(shape):
    face_landmarks = {}
    face_landmarks["left_eyebrow"] = [
        tuple([shape[17][0], shape[17][1]]), # tuple(shape[17]),
        tuple([shape[18][0], shape[18][1]]), # tuple(shape[18]),
        tuple([shape[19][0], shape[19][1]]), # tuple(shape[19]),
        tuple([shape[20][0], shape[20][1]]), # tuple(shape[20]),
        tuple([shape[21][0], shape[21][1]]), # tuple(shape[21]),
    ]
    face_landmarks["right_eyebrow"] = [
        tuple([shape[22][0], shape[22][1]]), # tuple(shape[22]),
        tuple([shape[23][0], shape[23][1]]), # tuple(shape[23]),
        tuple([shape[24][0], shape[24][1]]), # tuple(shape[24]),
        tuple([shape[25][0], shape[25][1]]), # tuple(shape[25]),
        tuple([shape[26][0], shape[26][1]]), # tuple(shape[26]),
    ]
    face_landmarks["nose_bridge"] = [
        tuple([shape[27][0], shape[27][1]]), # tuple(shape[27]),
        tuple([shape[28][0], shape[28][1]]), # tuple(shape[28]),
        tuple([shape[29][0], shape[29][1]]), # tuple(shape[29]),
        tuple([shape[30][0], shape[30][1]]), # tuple(shape[30]),
    ]
    face_landmarks["nose_tip"] = [
        tuple([shape[31][0], shape[31][1]]), # tuple(shape[31]),
        tuple([shape[32][0], shape[32][1]]), # tuple(shape[32]),
        tuple([shape[33][0], shape[33][1]]), # tuple(shape[33]),
        tuple([shape[34][0], shape[34][1]]), # tuple(shape[34]),
        tuple([shape[35][0], shape[35][1]]), # tuple(shape[35]),
    ]
    face_landmarks["left_eye"] = [
        tuple([shape[36][0], shape[36][1]]), # tuple(shape[36]),
        tuple([shape[37][0], shape[37][1]]), # tuple(shape[37]),
        tuple([shape[38][0], shape[38][1]]), # tuple(shape[38]),
        tuple([shape[39][0], shape[39][1]]), # tuple(shape[39]),
        tuple([shape[40][0], shape[40][1]]), # tuple(shape[40]),
        tuple([shape[41][0], shape[41][1]]), # tuple(shape[41]),
    ]
    face_landmarks["right_eye"] = [
        tuple([shape[42][0], shape[42][1]]), # tuple(shape[42]),
        tuple([shape[43][0], shape[43][1]]), # tuple(shape[43]),
        tuple([shape[44][0], shape[44][1]]), # tuple(shape[44]),
        tuple([shape[45][0], shape[45][1]]), # tuple(shape[45]),
        tuple([shape[46][0], shape[46][1]]), # tuple(shape[46]),
        tuple([shape[47][0], shape[47][1]]), # tuple(shape[47]),
    ]
    face_landmarks["top_lip"] = [
        tuple([shape[48][0], shape[48][1]]), # tuple(shape[48]),
        tuple([shape[49][0], shape[49][1]]), # tuple(shape[49]),
        tuple([shape[50][0], shape[50][1]]), # tuple(shape[50]),
        tuple([shape[51][0], shape[51][1]]), # tuple(shape[51]),
        tuple([shape[52][0], shape[52][1]]), # tuple(shape[52]),
        tuple([shape[53][0], shape[53][1]]), # tuple(shape[53]),
        tuple([shape[54][0], shape[54][1]]), # tuple(shape[54]),
        tuple([shape[60][0], shape[60][1]]), # tuple(shape[60]),
        tuple([shape[61][0], shape[61][1]]), # tuple(shape[61]),
        tuple([shape[62][0], shape[62][1]]), # tuple(shape[62]),
        tuple([shape[63][0], shape[63][1]]), # tuple(shape[63]),
        tuple([shape[64][0], shape[64][1]]), # tuple(shape[64]),
    ]

    face_landmarks["bottom_lip"] = [
        tuple([shape[54][0], shape[54][1]]), # tuple(shape[54]),
        tuple([shape[55][0], shape[55][1]]), # tuple(shape[55]),
        tuple([shape[56][0], shape[56][1]]), # tuple(shape[56]),
        tuple([shape[57][0], shape[57][1]]), # tuple(shape[57]),
        tuple([shape[58][0], shape[58][1]]), # tuple(shape[58]),
        tuple([shape[59][0], shape[59][1]]), # tuple(shape[59]),
        tuple([shape[48][0], shape[48][1]]), # tuple(shape[48]),
        tuple([shape[64][0], shape[64][1]]), # tuple(shape[64]),
        tuple([shape[65][0], shape[65][1]]), # tuple(shape[65]),
        tuple([shape[66][0], shape[66][1]]), # tuple(shape[66]),
        tuple([shape[67][0], shape[67][1]]), # tuple(shape[67]),
        tuple([shape[60][0], shape[60][1]]), # tuple(shape[60]),
    ]

    face_landmarks["chin"] = [
        tuple([shape[0][0], shape[0][1]]), # tuple(shape[0]),
        tuple([shape[1][0], shape[1][1]]), # tuple(shape[1]),
        tuple([shape[2][0], shape[2][1]]), # tuple(shape[2]),
        tuple([shape[3][0], shape[3][1]]), # tuple(shape[3]),
        tuple([shape[4][0], shape[4][1]]), # tuple(shape[4]),
        tuple([shape[5][0], shape[5][1]]), # tuple(shape[5]),
        tuple([shape[6][0], shape[6][1]]), # tuple(shape[6]),
        tuple([shape[7][0], shape[7][1]]), # tuple(shape[7]),
        tuple([shape[8][0], shape[8][1]]), # tuple(shape[8]),
        tuple([shape[9][0], shape[9][1]]), # tuple(shape[9]),
        tuple([shape[10][0], shape[10][1]]), # tuple(shape[10]),
        tuple([shape[11][0], shape[11][1]]), # tuple(shape[11]),
        tuple([shape[12][0], shape[12][1]]), # tuple(shape[12]),
        tuple([shape[13][0], shape[13][1]]), # tuple(shape[13]),
        tuple([shape[14][0], shape[14][1]]), # tuple(shape[14]),
        tuple([shape[15][0], shape[15][1]]), # tuple(shape[15]),
        tuple([shape[16][0], shape[16][1]]), # tuple(shape[16]),
    ]
    return face_landmarks


def rect_to_bb(rect):
    x1 = rect.left()
    x2 = rect.right()
    y1 = rect.top()
    y2 = rect.bottom()
    return (x1, x2, y2, x1)


def mask_image(image_path, args):
    # Read the image
    image = cv2.imread(image_path)
    original_image = image.copy()
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = image
    # face_locations = args.detector(gray, 1)
    # bboxes, _, _, offset, scale = args.detector.detect(gray)
    mask_type = args.mask_type
    verbose = args.verbose
    if args.code:
        ind = random.randint(0, len(args.code_count) - 1)
        mask_dict = args.mask_dict_of_dict[ind]
        mask_type = mask_dict["type"]
        args.color = mask_dict["color"]
        args.pattern = mask_dict["texture"]
        args.code_count[ind] += 1

    elif mask_type == "random":
        available_mask_types = get_available_mask_types()
        mask_type = random.choice(available_mask_types)
        if args.random_count%2 == 1:
            args.color = random.choice(available_mask_color)
            args.pattern = random.choice(available_mask_texture)
        else:
            args.color = None
            args.pattern = None

    # if verbose:
    #     tqdm.write("Faces found: {:2d}".format(len(bboxes)))
    # Process each face in the image
    masked_images = []
    mask_binary_array = []
    mask = []

    # if len(bboxes) <= 0:
    #     f = open('not_found.txt', 'a')
    #     f.write('{}\n'.format(image_path))
    #     f.close()

    bboxes = np.array([[0,0,119,119,1.0]])

    param_lst, roi_box_lst = args.predictor(gray, bboxes)
    ver_lst = args.predictor.recon_vers(param_lst, roi_box_lst, dense_flag=False)

    f = open('not_found.txt', 'a')
    for (i, bbox) in enumerate(bboxes):
        x1,y1,x2,y2,score = bbox.astype(np.int)
        shape = ver_lst[i]
        # shape = face_utils.shape_to_np(shape)
        face_landmarks = shape_to_landmarks(shape)
        # face_location = rect_to_bb(face_location)
        face_location = (x1, x2, y2, x1)
        # draw_landmarks(face_landmarks, image)
        six_points_on_face, angle = get_six_points(face_landmarks, image)
        if six_points_on_face is None and angle is None:
            f.write('{}\n'.format(image_path))
            args.notfound_count += 1
            print('not found ({}) - {}'.format(args.notfound_count, image_path))
            continue
        mask = []
        if mask_type != "all":
            if len(masked_images) > 0:
                image = masked_images.pop(0)
            image, mask_binary = mask_face(
                image, face_location, six_points_on_face, angle, args, type=mask_type
            )

            # compress to face tight
            face_height = face_location[2] - face_location[0]
            face_width = face_location[1] - face_location[3]
            masked_images.append(image)
            mask_binary_array.append(mask_binary)
            mask.append(mask_type)
        else:
            available_mask_types = get_available_mask_types()
            for m in range(len(available_mask_types)):
                if len(masked_images) == len(available_mask_types):
                    image = masked_images.pop(m)
                img, mask_binary = mask_face(
                    image,
                    face_location,
                    six_points_on_face,
                    angle,
                    args,
                    type=available_mask_types[m],
                )
                masked_images.insert(m, img)
                mask_binary_array.insert(m, mask_binary)
            mask = available_mask_types
            cc = 1
    f.close()

    return masked_images, mask, mask_binary_array, original_image


def mask_image_readed(image,args):
    # Read the image
    original_image = image.copy()
    gray = image

    mask_type = args.mask_type
    verbose = args.verbose
    if args.code:
        ind = random.randint(0, len(args.code_count) - 1)
        mask_dict = args.mask_dict_of_dict[ind]
        mask_type = mask_dict["type"]
        args.color = mask_dict["color"]
        args.pattern = mask_dict["texture"]
        args.code_count[ind] += 1

    elif mask_type == "random":
        available_mask_types = get_available_mask_types()
        mask_type = random.choice(available_mask_types)
        if args.random_count%2 == 1:
            args.color = random.choice(available_mask_color)
            args.pattern = random.choice(available_mask_texture)
        else:
            args.color = None
            args.pattern = None

    # if verbose:
    #     tqdm.write("Faces found: {:2d}".format(len(bboxes)))
    # Process each face in the image
    masked_images = []
    mask_binary_array = []
    mask = []

    # if len(bboxes) <= 0:
    #     f = open('not_found.txt', 'a')
    #     f.write('{}\n'.format(image_path))
    #     f.close()

    if args.use_detector=='scrfd':
        bboxes = np.array([[0,0,119,119,1.0]])
        param_lst, roi_box_lst = args.predictor(gray, bboxes)
        ver_lst = args.predictor.recon_vers(param_lst, roi_box_lst, dense_flag=False)
    elif args.use_detector=='dlib':
        rects = args.detector(gray,1)
        bboxes=[]
        ver_lst=[]
        for r in rects:
            bboxes.append([r.rect.left(),r.rect.top(),r.rect.right(),r.rect.bottom(),1.0])
            ver_lst.append(np.array([[p.x, p.y] for p in args.predictor(gray, r.rect).parts()]))
        bboxes = np.array(bboxes)
        ver_lst = np.array(ver_lst)
    elif args.use_detector=='fan':
        bboxes = np.array([[0,0,119,119,1.0]])
        ver_lst = args.predictor.get_landmarks(gray,detected_faces=bboxes)


    for (i, bbox) in enumerate(bboxes):
        x1,y1,x2,y2,score = bbox.astype(np.int)
        shape = ver_lst[i]
        # shape = face_utils.shape_to_np(shape)
        if args.use_detector=='scrfd':
            face_landmarks = shape_to_landmarks(shape)
        else:
            face_landmarks = shape_to_landmarks_2(shape)
        # face_location = rect_to_bb(face_location)
        face_location = (x1, x2, y2, x1)
        # draw_landmarks(face_landmarks, image)
        point_result = get_six_points(face_landmarks, image)
        if point_result is None:
            return None
        else:
            six_points_on_face, angle = point_result
        #six_points_on_face, angle = get_six_points(face_landmarks, image)
        mask = []
        if mask_type != "all":
            if len(masked_images) > 0:
                image = masked_images.pop(0)
            image, mask_binary = mask_face(
                image, face_location, six_points_on_face, angle, args, type=mask_type
            )

            # compress to face tight
            face_height = face_location[2] - face_location[0]
            face_width = face_location[1] - face_location[3]
            masked_images.append(image)
            mask_binary_array.append(mask_binary)
            mask.append(mask_type)
        else:
            available_mask_types = get_available_mask_types()
            for m in range(len(available_mask_types)):
                if len(masked_images) == len(available_mask_types):
                    image = masked_images.pop(m)
                img, mask_binary = mask_face(
                    image,
                    face_location,
                    six_points_on_face,
                    angle,
                    args,
                    type=available_mask_types[m],
                )
                masked_images.insert(m, img)
                mask_binary_array.insert(m, mask_binary)
            mask = available_mask_types
            cc = 1

    return masked_images, mask, mask_binary_array, original_image


def mask_image_readed_skip(image,ref_point,args):
    # Read the image
    original_image = image.copy()
    gray = image

    mask_type = args.mask_type
    verbose = args.verbose
    if args.code:
        ind = random.randint(0, len(args.code_count) - 1)
        mask_dict = args.mask_dict_of_dict[ind]
        mask_type = mask_dict["type"]
        args.color = mask_dict["color"]
        args.pattern = mask_dict["texture"]
        args.code_count[ind] += 1

    elif mask_type == "random":
        available_mask_types = get_available_mask_types()
        mask_type = random.choice(available_mask_types)
        if args.random_count%2 == 1:
            args.color = random.choice(available_mask_color)
            args.pattern = random.choice(available_mask_texture)
        else:
            args.color = None
            args.pattern = None

    # if verbose:
    #     tqdm.write("Faces found: {:2d}".format(len(bboxes)))
    # Process each face in the image
    masked_images = []
    mask_binary_array = []
    mask = []

    # if len(bboxes) <= 0:
    #     f = open('not_found.txt', 'a')
    #     f.write('{}\n'.format(image_path))
    #     f.close()

    if args.use_detector=='scrfd':
        bboxes = np.array([[0,0,119,119,1.0]])
        param_lst, roi_box_lst = args.predictor(gray, bboxes)
        ver_lst = args.predictor.recon_vers(param_lst, roi_box_lst, dense_flag=False)
    elif args.use_detector=='dlib':
        rects = args.detector(gray,1)
        bboxes=[]
        ver_lst=[]
        for r in rects:
            bboxes.append([r.rect.left(),r.rect.top(),r.rect.right(),r.rect.bottom(),1.0])
            ver_lst.append(np.array([[p.x, p.y] for p in args.predictor(gray, r.rect).parts()]))
        bboxes = np.array(bboxes)
        ver_lst = np.array(ver_lst)
    elif args.use_detector=='fan':
        bboxes = np.array([[0,0,119,119,1.0]])
        ver_lst = args.predictor.get_landmarks(gray,detected_faces=bboxes)


    for (i, bbox) in enumerate(bboxes):
        x1,y1,x2,y2,score = bbox.astype(np.int)

        if args.use_detector=='scrfd':
            check_ver = point_check(ref_point,ver_lst[i],args.skip_num)
        else:
            check_ver = point_check_2(ref_point,ver_lst[i],args.skip_num)
        if not check_ver:
            print("point check False")
            return "skip"

        shape = ver_lst[i]
        # shape = face_utils.shape_to_np(shape)
        if args.use_detector=='scrfd':
            face_landmarks = shape_to_landmarks(shape)
        else:
            face_landmarks = shape_to_landmarks_2(shape)
        # face_location = rect_to_bb(face_location)
        face_location = (x1, x2, y2, x1)
        # draw_landmarks(face_landmarks, image)
        point_result = get_six_points(face_landmarks, image)

        six_points_on_face, angle = point_result

        if six_points_on_face is None and angle is None:
            return None
        #six_points_on_face, angle = get_six_points(face_landmarks, image)
        mask = []
        if mask_type != "all":
            if len(masked_images) > 0:
                image = masked_images.pop(0)
            image, mask_binary = mask_face(
                image, face_location, six_points_on_face, angle, args, type=mask_type
            )

            # compress to face tight
            face_height = face_location[2] - face_location[0]
            face_width = face_location[1] - face_location[3]
            masked_images.append(image)
            mask_binary_array.append(mask_binary)
            mask.append(mask_type)
        else:
            available_mask_types = get_available_mask_types()
            for m in range(len(available_mask_types)):
                if len(masked_images) == len(available_mask_types):
                    image = masked_images.pop(m)
                img, mask_binary = mask_face(
                    image,
                    face_location,
                    six_points_on_face,
                    angle,
                    args,
                    type=available_mask_types[m],
                )
                masked_images.insert(m, img)
                mask_binary_array.insert(m, mask_binary)
            mask = available_mask_types
            cc = 1

    return masked_images, mask, mask_binary_array, original_image

def mask_image_use_txt(image,ref_point,six_points_on_face, angle,args):
    
    face_location = (0,119,119,0)

    # Read the image
    original_image = image.copy()
    gray = image

    mask_type = args.mask_type
    verbose = args.verbose
    if args.code:
        ind = random.randint(0, len(args.code_count) - 1)
        mask_dict = args.mask_dict_of_dict[ind]
        mask_type = mask_dict["type"]
        args.color = mask_dict["color"]
        args.pattern = mask_dict["texture"]
        args.code_count[ind] += 1

    elif mask_type == "random":
        available_mask_types = get_available_mask_types()
        mask_type = random.choice(available_mask_types)
        if args.random_count%2 == 1:
            args.color = random.choice(available_mask_color)
            args.pattern = random.choice(available_mask_texture)
        else:
            args.color = None
            args.pattern = None

    # if verbose:
    #     tqdm.write("Faces found: {:2d}".format(len(bboxes)))
    # Process each face in the image
    masked_images = []
    mask_binary_array = []
    mask = []

    # if len(bboxes) <= 0:
    #     f = open('not_found.txt', 'a')
    #     f.write('{}\n'.format(image_path))
    #     f.close()

    if six_points_on_face is None and angle is None:
        return None
    #six_points_on_face, angle = get_six_points(face_landmarks, image)
    mask = []
    if mask_type != "all":
        if len(masked_images) > 0:
            image = masked_images.pop(0)
        image, mask_binary = mask_face(
            image, face_location, six_points_on_face, angle, args, type=mask_type
        )

        # compress to face tight
        face_height = face_location[2] - face_location[0]
        face_width = face_location[1] - face_location[3]
        masked_images.append(image)
        mask_binary_array.append(mask_binary)
        mask.append(mask_type)
    else:
        available_mask_types = get_available_mask_types()
        for m in range(len(available_mask_types)):
            if len(masked_images) == len(available_mask_types):
                image = masked_images.pop(m)
            img, mask_binary = mask_face(
                image,
                face_location,
                six_points_on_face,
                angle,
                args,
                type=available_mask_types[m],
            )
            masked_images.insert(m, img)
            mask_binary_array.insert(m, mask_binary)
        mask = available_mask_types
        cc = 1

    return masked_images, mask, mask_binary_array, original_image

def inference_landmark(image,ref_point,args):
    # Read the image
    original_image = image.copy()
    gray = image



    if args.use_detector=='scrfd':
        bboxes = np.array([[0,0,119,119,1.0]])
        param_lst, roi_box_lst = args.predictor(gray, bboxes)
        ver_lst = args.predictor.recon_vers(param_lst, roi_box_lst, dense_flag=False)
    elif args.use_detector=='dlib':
        rects = args.detector(gray,1)
        bboxes=[]
        ver_lst=[]
        for r in rects:
            bboxes.append([r.rect.left(),r.rect.top(),r.rect.right(),r.rect.bottom(),1.0])
            ver_lst.append(np.array([[p.x, p.y] for p in args.predictor(gray, r.rect).parts()]))
        bboxes = np.array(bboxes)
        ver_lst = np.array(ver_lst)
    elif args.use_detector=='fan':
        bboxes = np.array([[0,0,119,119,1.0]])
        ver_lst = args.predictor.get_landmarks(gray,detected_faces=bboxes)


    for (i, bbox) in enumerate(bboxes):
        x1,y1,x2,y2,score = bbox.astype(np.int)

        if args.use_detector=='scrfd':
            check_ver = point_check(ref_point,ver_lst[i],args.skip_num)
        else:
            check_ver = point_check_2(ref_point,ver_lst[i],args.skip_num)
        if not check_ver:
            print("point check False")
            return None

        shape = ver_lst[i]
        # shape = face_utils.shape_to_np(shape)
        if args.use_detector=='scrfd':
            face_landmarks = shape_to_landmarks(shape)
        else:
            face_landmarks = shape_to_landmarks_2(shape)
        # face_location = rect_to_bb(face_location)
        face_location = (x1, x2, y2, x1)
        # draw_landmarks(face_landmarks, image)
        point_result = get_six_points(face_landmarks, image)

        six_points_on_face, angle = point_result

        if six_points_on_face is None and angle is None:
            return None

        
    return six_points_on_face, angle



def is_image(path):
    try:
        extensions = path[-4:]
        image_extensions = ["png", "PNG", "jpg", "JPG"]

        if extensions[1:] in image_extensions:
            return True 
        else:
            print("Please input image file. png / jpg")
            return False 
    except: 
        return False 


def get_available_mask_types(config_filename="masks/masks.cfg"):
    parser = ConfigParser()
    parser.optionxform = str
    parser.read(config_filename)
    available_mask_types = parser.sections()
    available_mask_types = [
        string for string in available_mask_types if "left" not in string
    ]
    available_mask_types = [
        string for string in available_mask_types if "right" not in string
    ]

    return available_mask_types


def print_orderly(str, n):
    # print("")
    hyphens = "-" * int((n - len(str)) / 2)
    str_p = hyphens + " " + str + " " + hyphens
    hyphens_bar = "-" * len(str_p)
    print(hyphens_bar)
    print(str_p)
    print(hyphens_bar)


def display_MaskTheFace():
    with open("utils/display.txt", "r") as file:
        for line in file:
            cc = 1
            print(line, end="")


def point_check(ref_point,ver,skip_num):
    land5=[]
    land5.append((ver[0][37]+(ver[0][38]-ver[0][37])/2 , ver[1][41]-(ver[1][41]-ver[1][37])/2))
    land5.append((ver[0][43]+(ver[0][44]-ver[0][43])/2, ver[1][47]-(ver[1][47]-ver[1][43])/2))
    land5.append((ver[0][30], ver[1][30]))
    land5.append((ver[0][48]+(ver[0][60]-ver[0][48])/2 , ver[1][60]-(ver[1][48]-ver[1][60])/2))
    land5.append((ver[0][64]+(ver[0][54]-ver[0][64])/2 , ver[1][64]-(ver[1][54]-ver[1][64])/2))
    
    for pi,p in enumerate(land5):
        sub_x = abs(ref_point[pi][0]-p[0])
        sub_y = abs(ref_point[pi][1]-p[1])
    
        if sub_x>=skip_num or sub_y>=skip_num:
            return False
    
    return True

def point_check_2(ref_point,ver,skip_num):
    land5=[]
    land5.append((ver[37][0]+(ver[38][0]-ver[37][0])/2 , ver[41][1]-(ver[41][1]-ver[37][1])/2))
    land5.append((ver[43][0]+(ver[44][0]-ver[43][0])/2, ver[47][1]-(ver[47][1]-ver[43][1])/2))
    land5.append((ver[30][0], ver[30][1]))
    land5.append((ver[48][0]+(ver[60][0]-ver[48][0])/2 , ver[60][1]-(ver[48][1]-ver[60][1])/2))
    land5.append((ver[64][0]+(ver[54][0]-ver[64][0])/2 , ver[64][1]-(ver[54][1]-ver[64][1])/2))
    
    for pi,p in enumerate(land5):
        sub_x = abs(ref_point[pi][0]-p[0])
        sub_y = abs(ref_point[pi][1]-p[1])
        if sub_x>=skip_num or sub_y>=skip_num:
            return False
    
    return True
