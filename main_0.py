from image_preprocessing.processing import feature_extract, face_data, SHAPE, FACEDETECT
from image_preprocessing.logger import get_logger
from bayes_model.model import GaussNB, makedirs, exists, basename
import os
import cv2
import numpy as np

logger = get_logger(__file__)


class ImageNB(GaussNB):
    def train_by_folders(self, paths=None, root=None, simulate=None, limit=None):
        folders = []
        if root is not None:
            folders = [f.path for f in os.scandir(root) if f.is_dir()]
        else:
            if paths is None or len(paths) == 0:
                return
            folders = paths
        Xs = [np.array([255/2 for i in range(80*80)], dtype='uint8')]
        Ys = ['non_face']
        fl = -1
        for folder in folders:
            fl += 1
            fc = -1
            for f in os.scandir(folder):
                if f.is_file() and ('.jpg' in f.path or '.png' in f.path) and (
                        limit is None or limit[fl] is None or fc < limit[fl]):
                    fc += 1
                    Xs.append(feature_extract(cv2.imread(f.path)))
                    Ys.append(basename(folder))
        super().fit(Xs=Xs, Ys=Ys)
        if simulate is not None and len(simulate) == len(folders):
            self.over_all_probability = simulate

    def display_rs(self):
        rs = {}
        for idx in range(len(self.label)):
            a = np.array([kn[0] for kn in self.knowledge[idx]], dtype='uint8')
            shape = (int(np.sqrt(a.shape[0])), int(np.sqrt(a.shape[0])))
            rs[self.label[idx]] = a.reshape(shape)
        return rs

    def predict(self, img):
        img = feature_extract(img)
        return super().predict(X=img)


def gen_non_face_data(path='D:/non_face', cnt=20000):
    r = cnt
    if not exists(path):
        makedirs(path)
    video_capture = cv2.VideoCapture(0)
    while cnt >= 0:
        ret, frame = video_capture.read()
        frame = cv2.flip(frame, 1)
        faces = FACEDETECT.extract_faces_pos(frame)
        for y in range(frame.shape[0] - 80):
            if cnt < 0:
                break
            for x in range(frame.shape[1] - 80):
                if (x, y, 80, 80) not in faces:
                    if cnt < 0:
                        break
                    cnt -= 1
                    logger.info('Finish {} %'.format(int((1 - cnt / r) * 10000) / 100))
                    cv2.imwrite(img=frame[y:y + 80, x:x + 80], filename=path + '/' + str(cnt) + '.jpg')
                    cv2.imshow('non_face', frame[y:y + 80, x:x + 80])
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break


def face_detect_from_webcam(classifier):
    sample = classifier.display_rs()
    video_capture = cv2.VideoCapture(0)
    size = 200
    while True:
        ret, frame = video_capture.read()
        frame = cv2.flip(frame, 1)
        y, x = int(frame.shape[0] / 2), int(frame.shape[1] / 2)
        gray = frame[y - int(size / 2):y + int(size / 2), x - int(size / 2):x + int(size / 2)]
        cv2.rectangle(frame, (x - int(size / 2), y - int(size / 2)), (x + int(size / 2), y + int(size / 2)),
                      (100, 100, 100), 2)
        lb = classifier.predict(gray)
        logger.info(lb)
        if lb != 'non_face':
            cv2.rectangle(frame, (x - int(size / 2), y - int(size / 2)), (x + int(size / 2), y + int(size / 2)),
                          (0, 255, 0), 2)
            cv2.imshow('face', gray)

        cv2.imshow('Video', frame)
        for label, i in sample.items():
            cv2.imshow(label, i)
        cv2.imshow('area', feature_extract(gray))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # gen_non_face_data()
    classifier = ImageNB().load('D:/trained_model/My_Classifier')
    # classifier.train_by_folders(paths=[face_data, 'D:/non_face'], limit=[2000, None], simulate=[1, 1e20])
    # classifier.save()
    face_detect_from_webcam(classifier)
