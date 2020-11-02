import cv2
import imutils
import numpy as np
def faceboxes_detect(image, face_model, img_heights, exact_thresh):
    box = None
    old_conf = 0.5
    image_rs = None
    for img_height in img_heights:
        img = imutils.resize(image, height=img_height)
        for i in range(4):
            if i != 0:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                temp = np.float32(img)
                bboxs = face_model.detect_faces(temp)
                if bboxs is None or len(bboxs)==0:
                    continue
                bbox = bboxs[np.argmax(bboxs[:, 4])]
                if bbox[-1] > old_conf:
                    old_conf = bbox[-1]
                    box = bbox
                    image_rs = img
        if old_conf > exact_thresh:
            break
    return image_rs, box

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