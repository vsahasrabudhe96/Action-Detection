from email.mime import image
import os
import numpy as np
from keypoint import extract_keypoints,keypoint_detection,create_remove_folders,draw_holistics,draw_styled_landmarks
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import argparse
import mediapipe as mp
import cv2


mp_holistic,mp_drawing = draw_holistics()
DATA_PATH = os.path.join('../data/')
actions = np.array(['hello','thank you','I love you']) ## Actions we try to detect
no_sequences = 20  ## no_sequences == number of videos we want to collect e.g here we will collect 30 videos
sequence_length = 20 ## Each video will be of 30 frames
create_remove_folders(DATA_PATH,actions,no_sequences)

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:
    
    # Loop through the action
    for action in actions:
        
        # Loop through sequences/videos
        for sequence in range(no_sequences):
            
            # Loop through length
            
            for frame_num in range(sequence_length):
                
                ret,frame = cap.read()
                
                image,results = keypoint_detection(frame,holistic)
                
                draw_styled_landmarks(image,results)
                
                
                # Apply pause logic
                if frame_num ==0:
                    cv2.putText(image,'STARTING COLLECTION',(120,200),
                                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),4,cv2.LINE_AA)
                    
                    cv2.putText(image,'Collecting frames for {} Video Number'.format(action,sequence),(15,12),
                                cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1,cv2.LINE_AA)
                    
                    cv2.waitKey(2000)
                else:
                    cv2.putText(image,'Collecting frames for {} Video Number {}'.format(action,sequence),(15,12),
                                cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1,cv2.LINE_AA)
                
                ## Applying collection logic and saving the coordinates as np array
                
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH,action,str(sequence),str(frame_num))
                np.save(npy_path,keypoints)

                
                cv2.imshow("OpenCV Data Collection",image)
                
                # Break in between collecting frames
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                
    cap.release()
    cv2.destroyAllWindows()



