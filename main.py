# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import argparse
import warnings
import time

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
warnings.filterwarnings('ignore')


SAMPLE_IMAGE_PATH = ""

def check_image(image, model_dir):
    height, width, channel = image.shape
    if width/height != 3/4:
        print("Image is not appropriate!!!\nHeight/Width should be 4/3.")
        return False
    else:
        return True


def dl_face_spoof_detect(image, model_dir, model_test, image_cropper):
    image_bbox = model_test.get_bbox(image)
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
        test_speed += time.time()-start
    print("Prediction cost {:.2f} s".format(test_speed))
    label = np.argmax(prediction)
    print(prediction)
    value = prediction[0][label] / 2
    if label == 1:
        return False, value
    return True, value

def draw_prediction(image, image_bbox, prediction):
    # draw result of prediction
    label = np.argmax(prediction)
    value = prediction[0][label]/2
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
        cv2.FONT_HERSHEY_COMPLEX, 0.5*image.shape[0]/1024, color)
    return image

from  moire import read_cfg, prepare_environment, fake_detection
import pandas as pd
from datetime import datetime

if __name__ == "__main__":
    # prepare environments
    ctx, queue, mf, prg = prepare_environment()
    # prepare parameters
    folder_int, folder_out, sigma_, sigmaMax, k, thresh, delta, device_id, model_dir, save_dir = read_cfg()

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
        check_result, conf = dl_face_spoof_detect(img, model_dir, model_test, image_cropper)
        if check_result:
            print(link_image,"is fake with score=", conf)
            results.append([link_image, 1])
        elif fake_detection(img, sigma_, sigmaMax, k, thresh, ctx, queue, mf, prg, delta):
            print(link_image, "is fake")
            results.append([link_image, 1])
        else:
            print(link_image, "is truth")
            results.append([link_image, 0])
    pd.DataFrame(results, columns=["path_to_image", "is_fake"]).to_csv(os.path.join(save_dir, "{}_result.csv".format(datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))))