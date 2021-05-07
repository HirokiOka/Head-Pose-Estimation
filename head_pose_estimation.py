import os,sys
import cv2
import dlib
import numpy as np
from imutils import face_utils

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('./data/haarcascade_frontalface_alt2.xml')
face_parts_detector = dlib.shape_predictor('./data/shape_predictor_68_face_landmarks.dat')
cvFont = cv2.FONT_HERSHEY_PLAIN
model_points = np.array([
        (0.0,0.0,0.0), # 30
        (-30.0,-125.0,-30.0), # 21
        (30.0,-125.0,-30.0), # 22
        (-60.0,-70.0,-60.0), # 39
        (60.0,-70.0,-60.0), # 42
        (-40.0,40.0,-50.0), # 31
        (40.0,40.0,-50.0), # 35
        (-70.0,130.0,-100.0), # 48
        (70.0,130.0,-100.0), # 54
        (0.0,158.0,-10.0), # 57
        (0.0,250.0,-50.0) # 8
        ])
size = (720, 1280)
focal_length = size[1]
center = (size[1] // 2, size[0] // 2)
camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype='double')
dist_coeffs = np.zeros((4, 1))

def main():
    while True:

        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.11, minNeighbors=1, minSize=(100, 100))

        if len(faces) == 1:
            x, y, w, h = faces[0, :]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
            face = dlib.rectangle(x, y, x + w, y + h)
            face_parts = face_parts_detector(gray, face)
            face_parts = face_utils.shape_to_np(face_parts)
            parts_for_estimation = np.array([
                (face_parts[30]),
                (face_parts[21]),
                (face_parts[22]),
                (face_parts[39]),
                (face_parts[42]),
                (face_parts[31]),
                (face_parts[35]),
                (face_parts[48]),
                (face_parts[54]),
                (face_parts[57]),
                (face_parts[8]),
                ],dtype='double')

            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, parts_for_estimation, camera_matrix,dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
            (nose_end_point2D, _) = cv2.projectPoints(np.array([(0.0, 0.0, 500.0)]), rotation_vector,translation_vector, camera_matrix, dist_coeffs)

            for i, ((x, y)) in enumerate(face_parts):
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                cv2.putText(frame, str(i), (x + 2, y - 2),
                        cvFont, 0.3, (0, 255, 0), 1)
            p1 = (int(parts_for_estimation[0][0]), int(parts_for_estimation[0][1]))
            p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

            cv2.arrowedLine(frame, p1, p2, (255, 0, 0), 2)
        else:
            cv2.putText(frame, "Face Detection Failed", (10, 120),
                    cvFont, 2, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == 27:
            break

if __name__ == '__main__':
    main()
    cap.release()
    cv2.destroyAllWindows()
