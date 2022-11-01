#!/usr/bin/python3

import cv2 as cv
import numpy as np
from cv2.xfeatures2d import matchGMS
from skimage import data
from skimage.color import rgb2gray
from skimage.feature import match_descriptors, ORB, plot_matches
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
import numpy as np
import sys
import kdtree
import collections
import math
from Analyser import *
from Matcher import *
from reconstruct import *

def main():
    cap = cv.VideoCapture('movie.mp4')
    if not cap.isOpened():
        print('Error opening video file!')
        exit(1)

    last_frame = None

    width = 1920
    height = 1080

    info_cap = cv.VideoCapture('movie.mp4')
    if info_cap.isOpened():
        width = info_cap.get(3)
        height = info_cap.get(4)
        info_cap.release()

    intrinsic_matrix = np.array([
        [500,   0,    width//2 ],
        [0,   500,    height//2],
        [0,     0,    1       ]
    ])

    print('intrinsic_matrix')
    print(intrinsic_matrix)

    points_file = open('points', 'w')
    points_in_frame_file = open('points_in_frame', 'w')
    camera_pose_rotation_file = open('camera_pose_rotation', 'w')
    camera_pose_location_file = open('camera_pose_location', 'w')

    matcher = Matcher()

    first = True
    prev_proj = None

    while cap.isOpened(): 
        ret, frame = cap.read()
        if frame is not None:
            print(frame.shape)
            frame = cv.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
        kp1, kp2, matches = None, None, None
        
        if first:
            prev_proj = np.hstack((intrinsic_matrix, np.zeros((3, 1))))
            first = False

        if last_frame is not None and frame is not None: 
            prev_proj = reconstruct(last_frame, frame, intrinsic_matrix, prev_proj, matcher, points_file, points_in_frame_file, camera_pose_rotation_file, camera_pose_location_file)
            
        last_frame = frame

    points_file.close()
    points_in_frame.close()
    camera_pose_rotation_file.close()
    camera_pose_location_file.close()

    cap.release()
    cv.destroyAllWindows()



if __name__ == "__main__":
    main()

