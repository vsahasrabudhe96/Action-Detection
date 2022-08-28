import mediapipe as mp
import cv2
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
    


def extract_keypoints():
    pass