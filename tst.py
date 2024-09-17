import cv2
import numpy as np

# Function to find and draw circles on the frame
def detect_circles(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Detect circles using HoughCircles
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50, param1=50, param2=30, minRadius=10, maxRadius=50)
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(frame, (x, y), r, (0, 255, 0), 4)  # Draw the circle in green
            
    return frame, circles

# Initialize the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect circles
    processed_frame, circles = detect_circles(frame)
    
    # Create a mask for the detected circles
    mask = np.zeros_like(frame)
    if circles is not None:
        for (x, y, r) in circles:
            cv2.circle(mask, (x, y), r, (255, 255, 255), -1)  # Fill the circle area with white in the mask
    
    # Convert mask to grayscale
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    # Threshold the mask
    _, thresh_mask = cv2.threshold(mask_gray, 1, 255, cv2.THRESH_BINARY)
    
    # Convert original frame to white background
    white_background = np.ones_like(frame) * 255
    
    # Combine the white background with the mask
    result = np.where(thresh_mask[:, :, None] == 255, frame, white_background)
    
    # Display the result
    cv2.imshow('Result', result)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()