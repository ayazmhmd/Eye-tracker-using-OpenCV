import cv2
import numpy as np
from pygame import mixer
import time

# init part
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.maxArea = 1500
detector = cv2.SimpleBlobDetector_create(detector_params)


def detect_faces(img, cascade):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = cascade.detectMultiScale(gray_frame, 1.3, 5)
    if len(coords) > 1:
        biggest = (0, 0, 0, 0)
        for i in coords:
            if i[3] > biggest[3]:
                biggest = i
        biggest = np.array([i], np.int32)
    elif len(coords) == 1:
        biggest = coords
    else:
        return None,None
    for (x, y, w, h) in biggest:
        frame = img[y:y + h, x:x + w]
        bbox=(x,y,w,h)
    return frame,bbox


def detect_eyes(img, cascade):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = cascade.detectMultiScale(gray_frame, 1.3, 5)  # detect eyes
    width = np.size(img, 1)  # get face frame width
    height = np.size(img, 0)  # get face frame height
    left_eye = None
    right_eye = None
    lefteye_xcenter=None
    righteye_xcenter=None
    lefteye_ycenter=None
    righteye_ycenter=None
    lefteye_bbox=None
    righteye_bbox=None
    for (x, y, w, h) in eyes:
        if y > height / 2:
            pass
        x_eyecenter = x + w / 2  # get the eye center
        y_eyecenter = y + h / 2  # get the eye center
        if x_eyecenter < width * 0.5:
            left_eye = img[y:y + h, x:x + w]
            lefteye_xcenter=x_eyecenter
            lefteye_ycenter=y_eyecenter
            lefteye_bbox=(x,y,w,h)
        else:
            right_eye = img[y:y + h, x:x + w]
            righteye_xcenter=x_eyecenter
            righteye_ycenter=y_eyecenter
            righteye_bbox=(x,y,w,h)
    eyes=(left_eye, right_eye)
    l_eyes_center=(lefteye_xcenter,lefteye_ycenter)
    r_eyes_center=(righteye_xcenter,righteye_ycenter)
    eyes_center=(l_eyes_center,r_eyes_center)
    eyes_bbox=(lefteye_bbox,righteye_bbox)
    return eyes,eyes_center,eyes_bbox

def main():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('image')
    while True:
        _, frame = cap.read()
        face_frame,bbox = detect_faces(frame, face_cascade)
        if face_frame is not None:
            eyes,eyes_center,eyes_bbox = detect_eyes(face_frame, eye_cascade)
            (x,y,w,h)=bbox
            cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), (250, 19, 10),2)
            cv2.putText(frame, "face Detected", (10, 15), cv2.LINE_4,0.4, (0,50,250), 1)
            i=0
            eyes_center=list(eyes_center)
            eyes_bbox=list(eyes_bbox)
            print(eyes_center)
            for eye in eyes:
                if eye is not None and eyes_center[i] is not None and eyes_bbox[i] is not None:
                    x_eye,y_eye=eyes_center[i]
                    (x1,y1,w1,h1)=eyes_bbox[i]
                    cv2.putText(frame, "eyes Detected", (150, 15), cv2.LINE_4,0.4, (0,50,250), 1)
                    cv2.circle(frame, (int(x+x_eye),int(y+y_eye)), 10, (0,50,250), 2)
                    i=i+1
        else:
            # Make beep sound on Windows
            mixer.init() 
            sound=mixer.Sound("bell.wav")
            sound.play()
            time.sleep(0.3)
            sound.stop()
        cv2.imshow('image', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
