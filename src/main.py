import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import argparse
import mediapipe as mp
import cv2
from keypoint import draw_holistics,keypoint_detection,draw_styled_landmarks,extract_keypoints,render_probabilites
import keras




if __name__ == "__main__":
    DATA_PATH = "../data/"
    actions = np.array(['hello','thank you','I love you']) ## Actions we try to detect
    no_sequences = 20  ## no_sequences == number of videos we want to collect e.g here we will collect 30 videos
    sequence_length = 20 ## Each video will be of 30 frames
    sequence = []
    sentence = []
    threshold = 0.75
    colors = [(255,0,0),(0,255,0),(0,0,255)]
    model = keras.models.load_model('../models/action.h5')
    mp_holistic,mp_drawing = draw_holistics()
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret,frame = cap.read()
            image, results = keypoint_detection(frame,holistic)
            draw_styled_landmarks(image,results)
            
            # Prediction Logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-20:]
            
            if len(sequence) == 20:
                res = model.predict(np.expand_dims(sequence,axis=0))[0] # Since model is expecting (num_sequence,20,1662)
                # print(actions[np.argmax(res)])
                
            # Visualization logic
                if res[np.argmax(res)] > threshold:
                    if len(sentence)>0:
                        if actions[np.argmax(res)]!=sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])
                if len(sentence)>1:
                    sentence = sentence[-1:]
                
                image = render_probabilites(res,actions,image,colors)
                cv2.rectangle(image,(0,0),(1020,40),(245,117,16),-1)
                cv2.putText(image," ".join(sentence),(3,30),
                            cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)

            cv2.imshow('OpenCV frame',image)
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


    


    
    

