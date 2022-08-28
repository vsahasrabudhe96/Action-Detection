from turtle import color
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import mediapipe as mp
import cv2
from keypoint import draw_holistics,keypoint_detection,draw_styled_landmarks






if __name__ == "__main__":
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



    
    

