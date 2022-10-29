
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
import kdtree
import collections
import math
from Analyser import *
from Matcher import *

def triangulate(pose1, pose2, pts1, pts2):
    ret = np.zeros((pts1.shape[0], 4))

    for i, p in enumerate(zip(pts1, pts2)):
        A = np.zeros((4,4))
        A[0] = p[0][0] * pose1[2] - pose1[0]
        A[1] = p[0][1] * pose1[2] - pose1[1]
        A[2] = p[1][0] * pose2[2] - pose2[0]
        A[3] = p[1][1] * pose2[2] - pose2[1]
        _, _, vt = np.linalg.svd(A)
        ret[i] = vt[3]
    
    return ret

"""
class Reconstruct:
    def __init__(self, first_frame, second_frame):
        

"""

file = open('points', 'w')
points_in_frame = open('points_in_frame', 'w')
camera_pose_file = open('camera_pose', 'w')
camera_points_file = open('camera_points', 'w')

first_img = cv2.imread('1.jpg')
second_img = cv2.imread('2.jpg')
third_img = cv2.imread('remote3.jpg')

height, width = first_img.shape[:2]
print("height and width: ", (height,width))

intrinsic_matrix = np.array([
    [600,   0,    width/2 ],
    [0,   600,    height/2],
    [0,     0,    1       ]
])

matcher = Matcher()
pts1, pts2 = matcher.match(first_img, second_img)

E, _ = cv2.findEssentialMat(pts1, pts2, intrinsic_matrix, cv2.RANSAC, 0.99, 0.01) # last 3 args , cv2.RANSAC, 0.99, 0.01
_, R, t, _ = cv.recoverPose(E, pts1, pts2, intrinsic_matrix)

first_camera_matrix = np.hstack((intrinsic_matrix, np.zeros((3,1))))
second_camera_matrix = intrinsic_matrix.dot(np.hstack((R, t.reshape((3, 1)))))

print('first camera matrix')
print(first_camera_matrix)
print('second camera matrix')
print(second_camera_matrix)

print('points 1 shape')
print(pts1.shape)
print('points 2 shape')
print(pts2.shape)

pts1 = pts1.T
pts2 = pts2.T

scale = 0.05

triangulated_points = cv2.triangulatePoints(first_camera_matrix, second_camera_matrix, pts1, pts2).T
#triangulated_points = triangulate(first_camera_matrix, second_camera_matrix, pts1, pts2)

for i in range(triangulated_points.shape[0]):
    file.write(str(scale*triangulated_points[i, 0] / triangulated_points[i,3])) # 
    file.write(' ')
    file.write(str(scale*triangulated_points[i, 1] / triangulated_points[i,3])) # 
    file.write(' ')
    file.write(str(scale*triangulated_points[i, 2] / triangulated_points[i,3])) # 
    file.write('\n')
points_num = triangulated_points.shape[0]
points_in_frame.write(str(points_num))
points_in_frame.write('\n')


##################

"""
pts1, pts2 = matcher.match(second_img, third_img)

E, _ = cv2.findEssentialMat(pts1, pts2, intrinsic_matrix, cv2.RANSAC, 0.99, 0.01)
_, R, t, _ = cv.recoverPose(E, pts1, pts2, intrinsic_matrix)

third_camera_matrix = np.hstack((R, t.reshape((3, 1))))

print('third_camera_matrix')
print(third_camera_matrix)

pts1 = pts1.T
pts2 = pts2.T

scale = 1.0

triangulated_points = cv2.triangulatePoints(second_camera_matrix, third_camera_matrix, pts1, pts2).T

for i in range(triangulated_points.shape[0]):
    file.write(str(scale*triangulated_points[i, 0])) # 
    file.write(' ')
    file.write(str(scale*triangulated_points[i, 1])) # 
    file.write(' ')
    file.write(str(scale*triangulated_points[i, 2])) # 
    file.write('\n')
points_num = triangulated_points.shape[0]
points_in_frame.write(str(points_num))
points_in_frame.write('\n')

"""

file.close()
points_in_frame.close()
camera_pose_file.close()
camera_points_file.close()
