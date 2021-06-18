import cv2 as cv
from cv2.xfeatures2d import matchGMS
import numpy as np

def match(img1, img2, orb):
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    matcher = cv.BFMatcher(cv.NORM_HAMMING)
    matches_all = matcher.match(des1, des2)
    matches_gms = matchGMS(img1.shape[:2], img2.shape[:2], kp1, kp2, matches_all)
    return matches_gms

def main():
    cap = cv.VideoCapture('movie.mov')
    if not cap.isOpened():
        print('Error opening video file!')
        exit(1)

    keypoints_num = 3000
    orb = cv.ORB_create(keypoints_num)
    last_frame = None

    while cap.isOpened(): 
        ret, frame = cap.read()
        if last_frame is not None:
            matched_imgs = match(last_frame, frame, orb)
            print(type(matched_imgs[0]))
        if ret:
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  
            kp = orb.detect(frame, None)
            img = cv.drawKeypoints(gray, kp, frame)
            cv.imshow('Frame', img)
            last_frame = frame
            if cv.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()

