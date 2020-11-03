from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg
from layers.functions.prior_box import PriorBox
from utils.nms_wrapper import nms
#from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.faceboxes import FaceBoxes
from utils.box_utils import decode
from utils.timer import Timer
import time

def load_para():
    parser = argparse.ArgumentParser(description='FaceBoxes')
    parser.add_argument('--folder', '-f', help='image folder')
    parser.add_argument('--savefolder', '-sf', help='image folder')

    parser.add_argument('-m', '--trained_model', default='FaceBoxes.PyTorch/weights/FaceBoxes.pth',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
    parser.add_argument('--dataset', default='PASCAL', type=str, choices=['AFW', 'PASCAL', 'FDDB'], help='dataset')
    parser.add_argument('--confidence_threshold', default=0.05, type=float, help='confidence_threshold')
    parser.add_argument('--top_k', default=5000, type=int, help='top_k')
    parser.add_argument('--nms_threshold', default=0.3, type=float, help='nms_threshold')
    parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
    parser.add_argument('-sh', '--show_image', action="store_true", default=True, help='show detection results')
    parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
    args = parser.parse_args()
    return args


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

class face_boxes_model:
    def __init__(self):
        self.net = None 
        self.device = None
        self.args = load_para()
    
    def detect_faces(self, img, resize=1.0):
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        scale = scale.to(self.device)

        loc, conf = self.net(img)  # forward pass
        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        # ignore low scores
        inds = np.where(scores > self.args.confidence_threshold)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:self.args.top_k]
        boxes = boxes[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        #keep = py_cpu_nms(dets, self.args.nms_threshold)
        keep = nms(dets, self.args.nms_threshold, force_cpu=self.args.cpu)
        dets = dets[keep, :]

        # keep top-K faster NMS
        dets = dets[:self.args.keep_top_k, :]

        return dets

    def load_face_model(self):
        torch.set_grad_enabled(False)
        # net and model
        net = FaceBoxes(phase='test', size=None, num_classes=2)    # initialize detector
        net = load_model(net, self.args.trained_model, self.args.cpu)
        net.eval()
        cudnn.benchmark = True
        self.device = torch.device("cpu" if self.args.cpu else "cuda")
        self.net = net.to(self.device)

if __name__ == '__main__':
    
    print('Finished loading model!')
    face_boxes_md = face_boxes_model()
    face_boxes_md.load_face_model()
    # testing scale
    resize = 1.0
    _t = {'forward_pass': Timer(), 'misc': Timer()}
    import glob
    filenames = glob.glob("/media/dan/Storage/Code/face_matching/test_img/*")
    import time, imutils
    for image_path in filenames:
    # testing begin
    # for i, img_name in enumerate(test_dataset):
    #     image_path = testset_folder + img_name + '.jpg'
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img_raw = imutils.resize(img_raw, height=100)
        img = np.float32(img_raw)
        if resize != 1:
            img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        
        dets = face_boxes_md.detect_faces(img)
        for b in dets:
            if b[4] < face_boxes_md.args.vis_thres:
                continue
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(img_raw, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
        cv2.imshow('res', img_raw)
        cv2.waitKey(0)