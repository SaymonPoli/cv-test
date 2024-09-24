import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    gray = cv.medianBlur(gray, 5)
    
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 20,
                             param1=50, param2=30, minRadius=10, maxRadius=50)
    
    white_background = np.ones_like(frame) * 255

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv.circle(white_background, (i[0], i[0]), i[2], (0, 0, 0), 2)
            cv.circle(white_background, (i[0], i[1]), 2, (0, 0, 0), 3)

    cv.imshow('Circles on White Background', white_background)

    cv.imshow('Test preview', frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
