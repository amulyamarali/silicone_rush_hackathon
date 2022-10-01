import cv2
import numpy as np
import playsound 

"""
three steps in total:
1> detecting fire
2> playing beep sound
3> sending mail
"""

fire_reported = 0
alarm_status = False

def play_audio():
    playsound.playsound("beep.mp3", True)

# video = cv2.VideoCapture('fire_video.mov')
video = cv2.VideoCapture(0)

while True: 
    ret, frame = video.read()
    frame = cv2.resize(frame,(1000,600))
    blur = cv2.GaussianBlur(frame, (15,15),0)
    # as the video is in BGR we hv to convert it into hsv
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # the below 2 lines of codes define the color of the image 
    lower = [18,50,50]
    upper = [35,255,255]

    # converting the above to a numpy array 
    
    lower = np.array(lower, dtype='uint8') # uint is unsigned integer 
    upper = np.array(upper, dtype='uint8')

    mask = cv2.inRange(hsv, lower, upper)
    # what the above code means is that we want the hsv colors in the given upper and lower range of colors

    output = cv2.bitwise_and(frame, hsv, mask=mask)
    # now we have created our detector 

    size = cv2.countNonZero(mask)

    # the above code counts the number pixels that actually have he fire as part of the detection 
       

    if int(size) > 15000: # if this is above 15000 then we print fire detected
        # print("fire detected")
        fire_reported+=1
        if fire_reported >=1:
            if alarm_status == False:
                play_audio()
                alarm_status = True


    if ret == False:
        break

    cv2.imshow("Output",blur)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
video.release()