import cv2
import attendance
import face
import numpy as np


cap = cv2.VideoCapture(0)
face_init = face.TrainImage("images")
images, names = face_init.find_image()

known_encodings = face_init.find_encodings()

while True:
    ret, frame = cap.read()

    frameC = cv2.resize(frame, None, fx=0.25, fy=0.25)
    frameC = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    recognition_init = face.Recogniton(frame)

    for encode_face, face_loc in zip(recognition_init.faces_cur_frame, recognition_init.encodes_cur_frame):
        face_match, face_dis = recognition_init.recognize(known_encodings, encode_face)
        match_index = np.argmin(face_dis)

        if face_match[match_index]:
            name = names[match_index].upper()
            y1, x2, y2, x1 = face_loc
            y1, x2, y2, x2 = y1 * 4, x2 * 4, y2 *4, x1 * 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.rectangle(frame, (x1, y2-35), (x2, y2), (255,0,0), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y1 + 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)

    cv2.imshow('Videos', frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()




