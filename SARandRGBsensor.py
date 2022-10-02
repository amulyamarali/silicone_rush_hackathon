
import numpy as np
import cv2
 
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.startWindowThread()

cap = cv2.VideoCapture(0)


out = cv2.VideoWriter(
    'output.avi',
    cv2.VideoWriter_fourcc(*'MJPG'),
    15.,
    (640,480))
fgbg = cv2.createBackgroundSubtractorKNN(detectShadows=True)
# fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
# fgbg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
# fgbg = cv2.bgsegm.BackgroundSubtractorGMG()
# fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

while(True):
    ret, frame = cap.read()

    
    frame = cv2.resize(frame, (640, 480))
    # using a greyscale picture, also for faster detection
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    fgmask = fgbg.apply(frame)
    
    # returns the bounding boxes for the detected objects
    boxes, weights = hog.detectMultiScale(fgmask, winStride=(8,8) )

    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

    for (xA, yA, xB, yB) in boxes:
        
        cv2.rectangle(fgmask, (xA, yA), (xB, yB),
                          (256, 256, 256), 3)
    
    
    out.write(fgmask.astype('uint8'))
    cv2.imshow('Frame', fgmask)
    cv2.imshow('FG MASK Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
