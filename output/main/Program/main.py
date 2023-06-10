import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime

video_capture = cv2.VideoCapture(0)
address = "http://192.168.18.212:8080/video"
video_capture.open(address)


aijaz_image = face_recognition.load_image_file("photos/aijaz_mustafa.jpg")
aijaz_encoding = face_recognition.face_encodings(aijaz_image)[0]

dileep_image = face_recognition.load_image_file("photos/dileep_shukla.jpg")
dileep_encoding = face_recognition.face_encodings(dileep_image)[0]

harsh_image = face_recognition.load_image_file("photos/harsh_tandon.jpg")
harsh_encoding = face_recognition.face_encodings(harsh_image)[0]

karanbeer_image = face_recognition.load_image_file("photos/karanbeer.jpg")
karanbeer_encoding = face_recognition.face_encodings(karanbeer_image)[0]

kiran_image = face_recognition.load_image_file("photos/kiran_kumar.jpg")
kiran_encoding = face_recognition.face_encodings(kiran_image)[0]

manpreet_image = face_recognition.load_image_file("photos/manpreet_kaur.jpg")
manpreet_encoding = face_recognition.face_encodings(manpreet_image)[0]

palwinder_image = face_recognition.load_image_file("photos/palwinder_singh.jpg")
palwinder_encoding = face_recognition.face_encodings(palwinder_image)[0]

varnit_image = face_recognition.load_image_file("photos/varnit_sharma.jpg")
varnit_encoding = face_recognition.face_encodings(varnit_image)[0]

vipul_image = face_recognition.load_image_file("photos/vipul_sharma.jpg")
vipul_encoding = face_recognition.face_encodings(vipul_image)[0]

known_face_encoding = [
    aijaz_encoding,
    dileep_encoding,
    harsh_encoding,
    karanbeer_encoding,
    kiran_encoding,
    manpreet_encoding,
    palwinder_encoding,
    varnit_encoding,
    vipul_encoding
]

known_faces_names = [
    "Aijaz Mustafa",
    "Dileep Shukla",
    "Harsh Tandon",
    "Karanbeer LEET",
    "Kiran Kumar",
    "Manpreet Kaur",
    "Palwinder Singh",
    "Varnit Sharma",
    "Vipul Sharma"
]

students = known_faces_names.copy()

face_locations = []
face_encodings = []
face_names = []
s = True

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(current_date + '.csv', 'w+', newline='')
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
            name = ""
            face_distance = face_recognition.face_distance(known_face_encoding, face_encoding)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = known_faces_names[best_match_index]

            face_names.append(name)
            if name in known_faces_names:
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (10, 100)
                fontScale = 1.5
                fontColor = (255, 0, 0)
                thickness = 3
                lineType = 2

                cv2.putText(frame, name + ' Present',
                            bottomLeftCornerOfText,
                            font,
                            fontScale,
                            fontColor,
                            thickness,
                            lineType)

                if name in students:
                    students.remove(name)
                    print(students)
                    current_time = now.strftime("%H-%M-%S")
                    lnwriter.writerow([name, current_time])
    cv2.imshow("attendence system", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()