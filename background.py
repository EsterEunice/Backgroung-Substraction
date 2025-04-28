import numpy as np
import cv2 as cv
cap = cv.VideoCapture(r'E:\Kuliah\Semester 6\citra\background substraction\vtest.mp4')
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
fgbg = cv.createBackgroundSubtractorMOG2(detectShadows=True)
while True:
    ret, frame = cap.read()
    if frame is None:
        break
    fgmask = fgbg.apply(frame)
    fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)

    frame_resized = cv.resize(frame, None, fx=0.3, fy=0.3)
    fgmask_resized = cv.resize(fgmask, None, fx=0.3, fy=0.3)

    cv.imshow('Frame', frame_resized)
    cv.imshow('FG MASK Frame', fgmask_resized)

    key = cv.waitKey(30) & 0xFF
    if key == ord('q') or key == 27:
     break
cap.release()
cv.destroyAllWindows()
