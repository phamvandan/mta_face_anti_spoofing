import cv2
import imutils
import numpy as np

def rotate_box(bbox, w, h, angle, ori_w, ori_h):
    x1, y1, x2, y2, _ = bbox
    if angle == 0:
        boxes = np.asarray([x1*ori_w/w, y1*ori_h/h, x2*ori_w/w, y2*ori_h/h])
        return boxes
    elif angle == 90:
        boxes = np.asarray([y1*ori_w/h, (w - x2)*ori_h/w, y2*ori_w/h, (w - x1)*ori_h/w])
        return boxes
    elif angle == 180:
        boxes = np.asarray([(w - x2)*ori_w/w, (h - y2)*ori_h/h, (w - x1)*ori_w/w, (h - y1)*ori_h/h])
        return boxes
    elif angle == 270:
        boxes = np.asarray([(h - y2)*ori_w/h, x1*ori_h/w, (h - y1)*ori_w/h, x2*ori_h/w])
        return boxes

def faceboxes_detect(image, face_model, img_heights, exact_thresh):
    ori_h, ori_w = image.shape[:2]
    origin_box = None
    box = None
    old_conf = 0.5
    image_rs = None
    for img_height in img_heights:
        img = imutils.resize(image, height=img_height)
        for i in range(4):
            if i != 0:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            h, w = img.shape[:2]
            temp = np.float32(img)
            bboxs = face_model.detect_faces(temp)
            if bboxs is None or len(bboxs)==0:
                continue
            bbox = bboxs[np.argmax(bboxs[:, 4])]
            if bbox[-1] > old_conf:
                old_conf = bbox[-1]
                box = bbox
                image_rs = img
                origin_box = rotate_box(box, w, h, 90 * i, ori_w, ori_h)
        if old_conf > exact_thresh:
            break
    return image_rs, box, origin_box

if __name__ == '__main__':
    img_heights = [800, 1500]
    ## load model
    exact_thresh = 0.8
    face_model = mytest.face_boxes_model()
    face_model.load_face_model()
    image = cv2.imread("/media/dan/Storage/Lab/face_anti_spoofing/code/mta_face_anti_spoofing/images/temp2.jpg")
    image_rs, box  = faceboxes_detect(image, face_model, img_heights, exact_thresh)
    x,y,a,b,_ = box
    print(image_rs.shape)
    cv2.rectangle(image_rs, (x,y), (a,b), (255,0,0))
    cv2.imshow("ok", image_rs)
    cv2.waitKey(0)