import cv2


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret,frame = cap.read()
    
    cv2.imshow('OpenCV frame',frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()