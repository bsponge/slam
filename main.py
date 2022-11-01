#!/usr/bin/python3

import cv2 as cv
import numpy as np
import numpy as np
import sys
import math
from Matcher import *
from reconstruct import *

def main():
    cap = cv.VideoCapture('videos/movie.mp4')
    if not cap.isOpened():
        print('Error opening video file!')
        exit(1)

    last_frame = None

    width = 1920
    height = 1080

    info_cap = cv.VideoCapture('videos/movie.mp4')
    if info_cap.isOpened():
        width = info_cap.get(3)
        height = info_cap.get(4)
        info_cap.release()

    K = np.array([
        [500,   0,    width//2 ],
        [0,   500,    height//2],
        [0,     0,    1       ]
    ])

    print('K')
    print(K)

    points_file = open('points.pts', 'w')
    points_in_frame_file = open('points_in_frame.pts', 'w')
    camera_poses_file = open('camera_poses.pts', 'w')

    matcher = Matcher()

    prev_proj = np.hstack((K, np.zeros((3, 1))))

    while cap.isOpened(): 
        ret, frame = cap.read()
        if not ret:
            exit()

        if last_frame is not None: 
            prev_proj = reconstruct(last_frame, frame, K, prev_proj, matcher, points_file, points_in_frame_file)
            
        last_frame = frame

    points_file.close()
    points_in_frame.close()
    camera_poses_file.close()

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()

