import numpy as np
import cv2
import time
from os.path import *
from os import *
from copy import deepcopy
import math


# data_folder = dirname(realpath(__file__)) + '/../data/face_train/'
face_data = r'D:/face_train_opcv'
from_path = r'D:/img_align_celeba'
SHAPE = (80, 80)

class FaceDetect:
    cascPath = dirname(realpath(__file__)) + '/../haarcascade_frontalface_default.xml'

    def __init__(self):
        self.faceCascade = cv2.CascadeClassifier(self.cascPath)

    def extract_faces(self, img):
        rs = []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        for (x, y, w, h) in faces:
            rs.append(img[y:y + h, x:x + w])
        return rs
    def extract_faces_pos(self, img):
        rs = []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return faces


FACEDETECT = FaceDetect()

def walk_path(path=from_path):
    if not exists(face_data):
        makedirs(face_data)
    r = True
    for (root, dirs, files) in walk(path, topdown=True):
        if not r:
            break
        for file in files:
            if not exists(face_data + '/' + file):
                frame = cv2.imread(root + sep + file)
                faces = FACEDETECT.extract_faces(frame)
                for face in faces:
                    cv2.imwrite(face_data + '/' + file, resize(face))


def feature_extract(img):
    rs = deepcopy(img)
    for step in [resize, gamma_correction, DOG_transfer, contrast_equalization]:
        rs = step(rs)
    return rs


def isGray(img):
    return len(img.shape) < 3 or img.shape[2] == 1


def merge_pixels(img, n_pixel=4):
    n_pixel = int(math.sqrt(n_pixel))
    rs = [[0 for j in range(int(img.shape[1] / n_pixel))] for i in range(int(img.shape[0] / n_pixel))]
    for i in range(int(img.shape[0] / n_pixel)):
        for j in range(int(img.shape[1] / n_pixel)):
            # x = np.average(img[i * n_pixel:(i + 1) * n_pixel, j * n_pixel:(j + 1) * n_pixel].reshape(-1))
            # img[i * n_pixel:(i + 1) * n_pixel, j * n_pixel:(j + 1) * n_pixel] = x
            rs[i][j] = np.average(img[i * n_pixel:(i + 1) * n_pixel, j * n_pixel:(j + 1) * n_pixel].reshape(-1))
    return np.array(rs)
    # return img


def convert2gray(img):
    img = img.astype('uint8')
    if isGray(img):
        return img
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except:
        gray = img
    return gray


def gamma_correction(img):
    mid = 0.5
    mean = np.mean(convert2gray(img))
    gamma = math.log(mid * 255) / math.log(mean)
    return np.power(img, gamma).clip(0, 255).astype(np.uint8)


def DOG_transfer(img):
    gray = convert2gray(img)
    sigma = 2
    K = 1.05
    g1 = cv2.GaussianBlur(gray, (19, 19), sigmaX=sigma, sigmaY=sigma)
    g2 = cv2.GaussianBlur(gray, (19, 19), sigmaX=sigma * K, sigmaY=sigma * K)

    DOG = g2 - g1
    DOG[DOG > 0] = 255
    DOG[DOG <= 0] = 0
    return DOG.astype(np.uint8)


def contrast_equalization(img):
    # clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8))
    # return clahe.apply(img)
    return cv2.equalizeHist(img)


def fps_calculating(prev_frame_time=0, new_frame_time=0, img=None):
    new_frame_time = time.time()
    try:
        fps = 1 / (new_frame_time - prev_frame_time)
    except:
        fps = 999999999999
    prev_frame_time = new_frame_time

    fps = int(fps)
    fps = str(fps)
    if img is not None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
    return prev_frame_time, new_frame_time, fps, img


def resize(img, dim=SHAPE):
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


if __name__ == '__main__':
    walk_path()
    # cap = cv2.VideoCapture(0)
    # prev_frame_time = 0
    # new_frame_time = 0
    # while (cap.isOpened()):
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     frame = cv2.flip(frame, 1)
    #     cv2.imshow('origin', frame)
    #     frame = gamma_correction(frame)
    #     # cv2.imshow('gamma_correction', frame)
    #     frame = DOG_transfer(frame)
    #     # cv2.imshow('dog_transfer', frame)
    #     frame = contrast_equalization(frame)
    #     cv2.imshow('contrast_equalization', frame)
    #
    #     # press 'Q' if you want to exit
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    #
    # # When everything done, release the capture
    # cap.release()
    # # Destroy the all windows now
    # cv2.destroyAllWindows()
