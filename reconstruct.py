
import numpy as np
import cv2
from cv2.xfeatures2d import matchGMS
from skimage import data
from skimage.color import rgb2gray
from skimage.feature import match_descriptors, ORB, plot_matches
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
import numpy as np
import sys
import collections
import math
import sys
from Matcher import *

def save_camera_pose(camera_poses_file, camera_pose):
    for i in range(3):
        camera_poses_file.write(str(camera_pose[i, 3]))
        camera_poses_file.write(' ')

    for row in camera_pose[:, :3]:
        for elem in row:
            camera_poses_file.write(str(elem))
            camera_poses_file.write(' ')

    # add green
    camera_poses_file.write(str(0.0))
    camera_poses_file.write(' ')
    camera_poses_file.write(str(1.0))
    camera_poses_file.write(' ')
    camera_poses_file.write(str(0.0))
    camera_poses_file.write(' ')

    camera_poses_file.write('\n')

    

def reconstruct(f1, f2, K, prev_proj, matcher, points_file, points_in_frame_file):
    pts1, pts2 = matcher.match(f1, f2)

    E, mask = cv2.findEssentialMat(pts1, pts2, K, cv2.RANSAC, 0.99999, 1.0) # last 3 args , cv2.RANSAC, 0.99, 0.01
    mask = mask.squeeze()

    height, width = f1.shape[:2]

    p1_filtered = []
    p2_filtered = []
    for i in range(len(mask)):
        p1 = pts1[i]
        p2 = pts2[i]
        if mask[i] == 1 and p1[0] > 0 and p2[0] > 0 and p1[0] < width and p2[1] < height and p1[1] < height and p2[0] < width:
            p1_filtered.append(p1)
            p2_filtered.append(p2)

    pts1 = np.array(p1_filtered)
    pts2 = np.array(p2_filtered)

    ret, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)

    first_camera_matrix = prev_proj
    second_camera_matrix = np.dot(K, np.hstack((R, t.reshape((3,1)))))

    print('second_camera_matrix')
    print(second_camera_matrix)

    print('matching points')
    print(pts1.shape)

    pts1 = pts1.T
    pts2 = pts2.T

    scale = 1.0

    # got to be removed and set in js!
    # points color
    color = (1.0, 1.0, 1.0)

    triangulated_points = cv2.triangulatePoints(first_camera_matrix, second_camera_matrix, pts1, pts2).T
    for i in range(triangulated_points.shape[0]):
        triangulated_points[i] = triangulated_points[i] / np.abs(triangulated_points[i, 3])

        # coords
        points_file.write(str(scale*triangulated_points[i, 0])) # 
        points_file.write(' ')
        points_file.write(str(scale*triangulated_points[i, 1])) # 
        points_file.write(' ')
        points_file.write(str(scale*triangulated_points[i, 2])) # 
        points_file.write(' ')
        # color
        points_file.write(str(color[0])) # 
        points_file.write(' ')
        points_file.write(str(color[1])) # 
        points_file.write(' ')
        points_file.write(str(color[2])) # 
        points_file.write('\n')

    points_num = triangulated_points.shape[0]
    points_in_frame_file.write(str(points_num))
    points_in_frame_file.write('\n')

    return second_camera_matrix