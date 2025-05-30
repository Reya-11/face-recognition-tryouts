import dlib
import cv2

predictor_path = 'assets/shape_predictor_68_face_landmarks.dat'
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(predictor_path)

def align_face(image):
    faces = face_detector(image, 1)
    if len(faces) == 0:
        return None
    face = faces[0]
    landmarks = landmark_predictor(image, face)
    points = [(p.x, p.y) for p in landmarks.parts()]
    left_eye = points[36]
    right_eye = points[45]
    return cv2.resize(image, (160, 160))
