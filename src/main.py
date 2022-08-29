import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import argparse
import mediapipe as mp
import cv2
from keypoint import draw_holistics,keypoint_detection,draw_styled_landmarks
from preprocess import train_test_data





if __name__ == "__main__":
    DATA_PATH = "../data/"
    actions = np.array(['hello','thank you','I love you']) ## Actions we try to detect
    no_sequences = 20  ## no_sequences == number of videos we want to collect e.g here we will collect 30 videos
    sequence_length = 20 ## Each video will be of 30 frames
    # create_folders(DATA_PATH,actions,no_sequences)
    
    mp_holistic,mp_drawing = draw_holistics()
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret,frame = cap.read()
            image, results = keypoint_detection(frame,holistic)
            draw_styled_landmarks(image,results)
            cv2.imshow('OpenCV frame',image)
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


    


    
    

