import tensorflow as tf
import time
from keras.models import load_model
from keras.preprocessing import image
import cv2
import numpy as np
from tensorflow import keras
from collections import Counter
import collections

model =  tf.keras.models.load_model("C:\\Users\\gana4\\MAJOR PROJECT\\Major_project_face_exp_rec_model.h5")
# initialize the Haar Cascade face detection model
def Facerecognization():
    face_classifier = cv2.CascadeClassifier(cv2.samples.findFile(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'))
    classifier =load_model("C:\\Users\\gana4\\MAJOR PROJECT\\Major_project_face_exp_rec_model.h5")

    emotion_labels =[
        "angry",
        "fear",
        "happy",
        "neutral",
        "sad",
        "surprise"
    ]

    cap = cv2.VideoCapture(0)


    li = []
    count = 0
    number = 10
    start_time = time.time()
    end_time = start_time + 30  # 30 seconds
    while time.time() < end_time:
        ret, frame = cap.read()
        labels = []
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray)

        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
            roi_gray = gray[y:y+h,x:x+w]
            roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray])!=0:
                roi = roi_gray.astype('float')/255.0
                roi = tf.keras.utils.img_to_array(roi)
                roi = np.expand_dims(roi,axis=0)

                prediction = classifier.predict(roi)[0]
                label=emotion_labels[prediction.argmax()]

                count +=1
                if count%number == 0:
                    li.append(label)

                coll = collections.Counter(li)
                
                label_position = (x,y)
                cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            else:
                cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            
            label = coll.most_common()[0][0]
            
            break

    cap.release()
    cv2.destroyAllWindows()
    return label
