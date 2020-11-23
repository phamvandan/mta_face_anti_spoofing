import cv2
import numpy as np
from skimage.filters import difference_of_gaussians, window
from scipy.fftpack import fftn, fftshift
import random, os



def moire_image(I, debug=1):
    rows, cols = I.shape
    m = cv2.getOptimalDFTSize(rows)
    n = cv2.getOptimalDFTSize(cols)
    padded = cv2.copyMakeBorder(I, 0, m - rows, 0, n - cols,
                                cv2.BORDER_CONSTANT, value=[0, 0, 0])

    planes = [np.float32(padded), np.zeros(padded.shape, np.float32)]
    complexI = cv2.merge(planes)  # Add to the expanded another plane with zeros

    cv2.dft(complexI,
            complexI)  # this way the result may fit in the source matrix

    cv2.split(complexI, planes)  # planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    cv2.magnitude(planes[0], planes[1], planes[0])  # planes[0] = magnitude
    magI = planes[0]

    matOfOnes = np.ones(magI.shape, dtype=magI.dtype)
    cv2.add(matOfOnes, magI, magI)  # switch to logarithmic scale
    cv2.log(magI, magI)

    magI_rows, magI_cols = magI.shape
    # crop the spectrum, if it has an odd number of rows or columns
    magI = magI[0:(magI_rows & -2), 0:(magI_cols & -2)]
    cx = int(magI_rows / 2)
    cy = int(magI_cols / 2)
    q0 = magI[0:cx, 0:cy]  # Top-Left - Create a ROI per quadrant
    q1 = magI[cx:cx + cx, 0:cy]  # Top-Right
    q2 = magI[0:cx, cy:cy + cy]  # Bottom-Left
    q3 = magI[cx:cx + cx, cy:cy + cy]  # Bottom-Right
    tmp = np.copy(q0)  # swap quadrants (Top-Left with Bottom-Right)
    magI[0:cx, 0:cy] = q3
    magI[cx:cx + cx, cy:cy + cy] = tmp
    tmp = np.copy(q1)  # swap quadrant (Top-Right with Bottom-Left)
    magI[cx:cx + cx, 0:cy] = q2
    magI[0:cx, cy:cy + cy] = tmp
    # print(magI)
    magII = cv2.normalize(magI, None, 0, 1,
                          cv2.NORM_MINMAX)  # Transform the matrix with float values into a

    if debug == 1:
        cv2.imshow("fourier", magII)

    return magI


def get_filted(img, k, sigma):
    filted = difference_of_gaussians(img, sigma, k * sigma)
    filter_img = filted * window('hann', img.shape)
    result = fftshift(np.abs(fftn(filter_img)))
    result = cv2.normalize(result, result, 0, 1, cv2.NORM_MINMAX)
    result = np.uint8(result * 255.)
    return result


def check(p, l):
    t_0 = 0.
    t_1 = 0.
    p_0 = 0.
    p_1 = 0.
    sl = []
    # print(p, l)
    p = np.array(p, dtype=np.float32)
    for i in range(0, 255, 1):
        sl.append(p[i] * 1.)
        if p[i] == 0:
            continue
        p[i] = p[i] * 1.
        p[i] = p[i] / l
        p_0 = p_0 + p[i]
        p_1 = p_1 + i * p[i]
    avg = -100000
    # print("avg", p)
    remember = 0
    for i in range(0, 256, 1):
        p_0 -= p[i]
        p_1 -= p[i] * i
        t_0 += p[i]
        t_1 += p[i] * i
        if p_0 == 0:
            continue
        m1 = p_1 / p_0
        if t_0 == 0:
            continue
        m0 = t_1 / t_0
        eA = t_1 + p_1
        eB = m0 * t_0 + m1 * p_0
        eAB = m0 * t_1 + m1 * p_1
        eBB = m0 * m0 * t_0 + m1 * m1 * p_0
        p_AB1 = eAB - eA * eB
        p_AB2 = eBB - eB * eB
        if p_AB2 == 0:
            remember = i
            break
        p_AB = p_AB1 * p_AB1 / p_AB2
        if p_AB > avg:
            remember = i
            avg = p_AB
    res = 0.0
    # print(remember, sl)
    for i in range(remember, 255, 1):
        res += sl[i]
    # print(res)
    return res / l


def is_moire(img):
    thres = np.zeros(256, dtype=np.int32)
    for i in img:
        thres[i]+=1
    r, c = img.shape
    shape_ = r*c
    thres = check(thres, shape_)

    return thres

def fake_detection(img_, sigma_, sigmaMax, k, thresh, delta):
    try:
        img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
    except:
        img_ = img_
    sigma = sigma_
    min_thres = 1
    dd_img = False
    rows, cols = img_.shape
    slide_r = rows // 9
    slide_c = cols // 9
    img = get_filted(img_, k, sigma)
    while sigma < sigmaMax:
        for size_l in range(4, 6, 1):
            if dd_img:
                break
            for size_r in range(4, 6, 1):
                if dd_img:
                    break
                for i in range(0, 9 - size_l, 1):
                    if dd_img:
                        break
                    for j in range(0, 9 - size_r, 1):
                        r = slide_r * i
                        rr = slide_r * (i + size_l)
                        c = slide_c * j
                        cc = slide_c * (j + size_r)
                        thres = is_moire(img[r:rr, c:cc])
                        if min_thres > thres:
                            min_thres = thres
                        if (thres < thresh):
                            return True

        sigma += delta
    return False

import configparser


def read_cfg(file_name="config.cfg"):
    config = configparser.ConfigParser()
    config.read(file_name)
    folder_int = config.get("moire", "in")
    folder_out = config.get("moire", "out")
    sigma_ = float(config.get("moire", "sigma_"))
    sigmaMax = float(config.get("moire", "sigma_max"))
    k = float(config.get("moire", "k"))
    thresh = float(config.get("moire", "thresh"))
    delta = float(config.get("moire", "delta"))
    device_id = int(config.get("dl_model", "device_id"))
    model_dir = config.get("dl_model", "model_dir")
    save_dir = config.get("dl_model", "save_dir")
    img_heights = config.get("facebox", "img_heights")
    exact_thresh = float(config.get("facebox", "exact_thresh"))
    return folder_int, folder_out, sigma_, sigmaMax, k, thresh, delta, device_id, model_dir, save_dir, img_heights, exact_thresh


if __name__ == "__main__":
    ## prepare environment

    ## read config parameters
    folder_int, folder_out, sigma_, sigmaMax, k, thresh, delta, device_id, model_dir, save_dir, img_heights, exact_thresh = read_cfg()

    file_images = os.listdir(folder_int)
    for f in file_images:
        link_image = os.path.join(folder_int, f)
        img = cv2.imread(link_image, 0)
        if img is None:
            print("can't read image")
        else:
            ## fake_detection
            if fake_detection(img, sigma_, sigmaMax, k, thresh,  delta):
                print(f, "is fake")
            else:
                print(f, "is not fake")
