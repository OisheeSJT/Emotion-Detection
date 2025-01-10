import cv2
import numpy as np
from keras.models import load_model

classifier = load_model('Emotion_Detection.h5')

face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = np.expand_dims(roi.reshape(48, 48), axis=0)
            roi = np.expand_dims(roi, axis=-1)


            preds = classifier.predict(roi)[0]
            label_index = preds.argmax()
            label = class_labels[label_index]
            label_position = (x,y)
            cv2.putText(frame,label,label_position,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,(0,255,0),2)
        else:
            cv2.putText(frame,'No Face Found',(20,60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,(0,255,0),2)

    cv2.imshow('Emotion Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
