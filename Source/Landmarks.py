import cv2
import dlib
import numpy as np


predictor_path = "../Data/swa_face.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
    print("Processing file: {}".format(f))
    img = io.imread(f)

dets = detector(img)
for k, d in enumerate(dets):
    shape = predictor(img, d)

vec = np.empty([68, 2], dtype = int)
numpy.savetxt('..Data/landmarks.txt',vec, delimiter=',', fmt = '%.04f')
