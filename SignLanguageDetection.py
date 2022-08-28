import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import mediapipe as mp
import cv2

## Getting Keypoints by using MP Holistics

mp_holistic = mp.solutions.holistic
mp_drawing  = mp.solutions.drawing_utils

def keypoint_detection(image,model):
    """
    - convert image to RGB
    - make the image unwriteable to save memory
    - process the image through the model
    - Mkae it writeable again
    - convert to BGR again

    Args:
        image (BGR Image): Frame that we are generatying via the opencv video 
        model (Mediapipe Model): Media Pipe model used to detect keypoints on the frame/image
    Output:
        image (BGR image with keypoints)
        results (numerical values for the location of key points)
    """
    
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    
    return image,results



## Drawing landmarks on the image

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image,results.face_landmarks,mp_holistic.FACEMESH_TESSELATION)
    mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image,results.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image,results.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS)
## Open the camera to capture
cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret,frame = cap.read()
        image, results = keypoint_detection(frame,holistic)
        draw_landmarks(image,results)
        cv2.imshow('OpenCV frame',image)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()



    
    

