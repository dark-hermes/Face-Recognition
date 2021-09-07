import face_recognition
import os
import cv2


class TrainImage:
    def __init__(self, path):
        self.path = path
        self.__image_dir = os.listdir(path=path)
        self.__images = []
        self.class_names = []
        self.encodings = []

    def find_image(self):
        # Use imread method from OpenCV instead of face_recogniton.load_image_file
        # to decrease execution time (faster)
        for image_name in self.__image_dir:
            image = cv2.imread(self.path + '/' + image_name)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.__images.append(image_rgb)
            self.class_names.append(os.path.splitext(image_name)[0])

        return self.__images, self.class_names

    def find_encodings(self):
        for img in self.__images:
            self.encodings.append(face_recognition.face_encodings(img)[0])

        return self.encodings

class Recogniton:
    def __init__(self, frame):
        self.faces_cur_frame = face_recognition.face_locations(frame)
        self.encodes_cur_frame = face_recognition.face_encodings(frame, self.faces_cur_frame)

    def recognize(self, encode_known, encode_now):
        self.__matches = face_recognition.compare_faces(encode_known, encode_now)
        self.__distance = face_recognition.face_distance(encode_known, encode_now)

        return self.__matches, self.__distance