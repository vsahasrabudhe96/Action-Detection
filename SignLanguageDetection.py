import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import mediapipe as mp
import cv2

## Open the camera to capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret,frame = cap.read()
    
    cv2.imshow('OpenCV frame',frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


## Getting Keypoints from 