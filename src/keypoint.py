import mediapipe as mp
import cv2
import numpy as np
import os
def draw_holistics():
    mp_holistic = mp.solutions.holistic
    mp_drawing  = mp.solutions.drawing_utils
    return mp_holistic,mp_drawing

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

def draw_styled_landmarks(image, results):
    mp_holistic,mp_drawing = draw_holistics()
    mp_drawing.draw_landmarks(image,results.face_landmarks,mp_holistic.FACEMESH_CONTOURS,
                              mp_drawing.DrawingSpec(color=(80,110,10),thickness=1,circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80,255,120),thickness=1,circle_radius=1))
    mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80,22,10),thickness=2,circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80,40,121),thickness=2,circle_radius=2))
    mp_drawing.draw_landmarks(image,results.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(120,22,70),thickness=2,circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121,44,255),thickness=2,circle_radius=2))
    mp_drawing.draw_landmarks(image,results.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245,117,66),thickness=2,circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245,66,230),thickness=2,circle_radius=2))
    


def extract_keypoints(results):
    """Extracting the Keypoints for pose ,face, left hand and right hand. Handled the error values for giving  zero value for item not present in frame.

    Args:
        results (array): class of JSON/Dictionary format with face, pose ,left hand and right hand landmarks as methods

    Returns:
        array: concatenated flattened array of landmarks for pose, face kleft hand and right hand for that particular frame. Flattened so as to match the input formatting for LSTM model
    """
    pose = np.array([[res.x,res.y,res.z,res.visibility] for res in results.pose_landmarks.landmark]).flatten()\
        if results.pose_landmarks else np.zeros(33*4)
        
    face = np.array([[res.x,res.y,res.z] for res in results.face_landmarks.landmark]).flatten() \
        if results.face_landmarks else np.zeros(468*3)
        
    lh = np.array([[res.x,res.y,res.z] for res in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(21*3)
        
    rh = np.array([[res.x,res.y,res.z] for res in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(21*3)
        
    return np.concatenate([pose,face,lh,rh])


def create_remove_folders(DATA_PATH,actions,no_sequences):
    for action in actions:
        for sequence in range(no_sequences):
            try:
                os.makedirs(os.path.join(DATA_PATH,action,str(sequence)))
            except:
                pass




