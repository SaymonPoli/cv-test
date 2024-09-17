import cv2
import numpy as np

def detect_circles(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50, param1=50, param2=30, minRadius=10, maxRadius=50)
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
            
    return frame, circles

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    processed_frame, circles = detect_circles(frame)
    
    mask = np.zeros_like(frame)
    if circles is not None:
        for (x, y, r) in circles:
            cv2.circle(mask, (x, y), r, (255, 255, 255), -1)  
    
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    _, thresh_mask = cv2.threshold(mask_gray, 1, 255, cv2.THRESH_BINARY)
    
    result = np.where(thresh_mask[:, :, None] == 255, frame, white_background)
    
    cv2.imshow('Result', result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()