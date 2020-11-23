# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import warnings
import time
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
from moire import read_cfg, fake_detection  # prepare_environment
import pandas as pd
from datetime import datetime
import json
import imutils
import sys

sys.path.insert(0, 'Pytorch_Retinaface/')

# sys.path.insert(0, 'FaceBoxes.PyTorch/')
# import mytest

from Pytorch_Retinaface.detect import load_net, do_detect

warnings.filterwarnings('ignore')

SAMPLE_IMAGE_PATH = ""

gpu_idx = -1
net, device, cfg = load_net(gpu_idx)


def check_image(image, model_dir):
    height, width, channel = image.shape
    if width / height != 3 / 4:
        print("Image is not appropriate!!!\nHeight/Width should be 4/3.")
        return False
    else:
        return True


def rotate_box(bbox, angle, h, w):
    x1, y1, x2, y2, conf = bbox
    if angle == 0:
        return bbox
    elif angle == 90:
        return w - y2, x1, w - y1, x2, conf
    elif angle == 180:
        return w - x2, h - y2, w - x1, h - y1, conf
    else:
        return y1, h - x2, y2, h - x1, conf


def check_box_angle(landmarks):
    y1 = landmarks[1]
    y2 = landmarks[3]
    y = landmarks[5]
    if y1 < y and y2 < y:
        return 0
    elif y1 < y < y2:
        return 270
    elif y1 > y and y2 > y:
        return 180
    elif y2 < y < y1:
        return 90
    print("UNKNOWN")
    return 0


def rotate_image(image, angle):
    image_rs = None
    if angle == 0:
        image_rs = image
    if angle == 90:
        image_rs = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        image_rs = cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        image_rs = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image_rs

def faceboxes_detect(image, img_heights, exact_thresh):
    box = None
    old_conf = 0.5
    image_rs = None
    angle = None
    resize_w = None
    resize_h = None
    landmark = None
    for img_height in img_heights:
        img = imutils.resize(image, height=img_height)
        for i in range(4):
            if i != 0:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            bboxs, landmarks = do_detect(img, net, device, cfg)
            if bboxs is None or len(bboxs) == 0:
                continue
            idx = np.argmax(bboxs[:, 4])
            bbox = bboxs[idx]
            if bbox[-1] > old_conf:
                old_conf = bbox[-1]
                box = bbox
                angle = 90 * i
                resize_h, resize_w = img.shape[:2]
                landmark = landmarks[idx]

            if old_conf > exact_thresh:
                break

    if box is not None:
        image_rs = rotate_image(image, angle)
        ori_h, ori_w = image_rs.shape[:2]
        x, y, a, b, conf = box
        box = [int(x * ori_w / resize_w), int(y * ori_h / resize_h), int(a * ori_w / resize_w),
               int(b * ori_h / resize_h), conf]
        # x, y, a, b, conf = box
        # cv2.rectangle(image_rs, (x, y), (a, b), (0, 0, 255), 2)
        # cv2.imshow("image_rs", image_rs)
        # cv2.waitKey(0)
        angle = check_box_angle(landmark)
        print("angle", angle)
        image_rs = rotate_image(image_rs, angle)
        ori_h, ori_w = image_rs.shape[:2]
        box = list(rotate_box(box, angle, ori_h, ori_w))
        x, y, a, b, _ = box

        box[2] = a - x
        box[3] = b - y

        # cv2.rectangle(image_rs, (x, y), (a, b), (0, 0, 255), 2)
        # cv2.imshow("image_rs", image_rs)
        # cv2.waitKey(0)
    return image_rs, box


def dl_face_spoof_detect(image, model_dir, model_test, image_cropper, img_heights, exact_thresh):
    temp = image
    image, image_bbox = faceboxes_detect(temp, img_heights, exact_thresh)
    # image, image_bbox = model_test.get_bbox(temp)
    if image is None:
        # image, image_bbox = model_test.get_bbox(temp)
        # if image is None:
        return False, -1, image, image_bbox
    prediction = np.zeros((1, 3))
    test_speed = 0
    # sum the prediction from single model's result
    for model_name in os.listdir(model_dir):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": image,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        start = time.time()
        prediction += model_test.predict(img, os.path.join(model_dir, model_name))
        test_speed += time.time() - start
    print("Prediction cost {:.2f} s".format(test_speed))
    label = np.argmax(prediction)
    value = prediction[0][label] / 2
    print("confidence=", value)
    if label == 1:
        return False, value, image, image_bbox
    return True, value, image, image_bbox


def draw_prediction(image, image_bbox, prediction):
    # draw result of prediction
    label = np.argmax(prediction)
    value = prediction[0][label] / 2
    if label == 1:
        result_text = "RealFace Score: {:.2f}".format(value)
        color = (255, 0, 0)
    else:
        result_text = "FakeFace Score: {:.2f}".format(value)
        color = (0, 0, 255)
    cv2.rectangle(
        image,
        (image_bbox[0], image_bbox[1]),
        (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
        color, 2)
    cv2.putText(
        image,
        result_text,
        (image_bbox[0], image_bbox[1] - 5),
        cv2.FONT_HERSHEY_COMPLEX, 0.5 * image.shape[0] / 1024, color)
    return image


if __name__ == "__main__":
    # prepare environments
    # ctx, queue, mf, prg = prepare_environment()
    # prepare parameters
    folder_int, folder_out, sigma_, sigmaMax, k, thresh, delta, device_id, model_dir, save_dir, img_heights, exact_thresh, device = read_cfg()
    img_heights = list(json.loads(img_heights))
    # load model
    # face_model = mytest.face_boxes_model()
    # face_model.load_face_model()

    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()

    file_images = os.listdir(folder_int)
    results = []
    # read image
    for f in file_images:
        link_image = os.path.join(folder_int, f)
        img = cv2.imread(link_image)
        if img is None:
            print("can't read image")
            results.append([link_image, -1])
            continue

        # bbox can be None if detected fail

        # if bbox is not None:
        #     x, y, a, b, _ = bbox
        #     ## face only from original image
        #     img = image[y:(y+b), x:(x+a)]
            # cv2.imshow("ok", img)
            # cv2.waitKey(0)

        conf = None
        check_result = fake_detection(img.copy(), img_, sigma_, sigmaMax, k, thresh, ctx, queue, mf, prg, delta, device)

        if check_result:
            print(link_image, "is fake with score=", 0)
            results.append([link_image, "fake_detected_by_opencv", 0])
        else:
            check_result, conf, image, bbox = dl_face_spoof_detect(img.copy(), model_dir, model_test, image_cropper, img_heights, exact_thresh)
            if check_result:
                print(link_image, "is fake")
                results.append([link_image, "fake_detected_by_dl", conf])
        if not check_result:
            print(link_image, "is truth")
            results.append([link_image, 0, conf])

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    pd.DataFrame(results, columns=["path_to_image", "is_fake", "conf"]).to_csv(
        os.path.join(save_dir, "{}_result.csv".format(datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))))
