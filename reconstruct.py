
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

"""
class Reconstruct:
    def __init__(self, first_frame, second_frame):
        

"""

file = open('points', 'w')
points_in_frame = open('points_in_frame', 'w')
camera_pose_file = open('camera_pose', 'w')
camera_points_file = open('camera_points', 'w')

first_img = cv2.imread('remote1.jpg')
second_img = cv2.imread('remote2.jpg')
third_img = cv2.imread('remote3.jpg')

height, width = first_img.shape[:2]
print((height,width))

intrinsic_matrix = np.array([
    [5500,   0,    width/2 ],
    [0,   5500,    height/2],
    [0,     0,    1       ]
])

# find better way to extract matching features and match them
'''
matcher = Matcher()
pts1, pts2 = matcher.match(first_img, second_img)
'''
orb = cv2.ORB_create(10000)
mth = cv2.BFMatcher(cv2.NORM_HAMMING)

kp1, des1 = orb.detectAndCompute(first_img, None)
kp2, des2 = orb.detectAndCompute(second_img, None)

matches = mth.knnMatch(des1, des2, k=2)

pts1 = []
pts2 = []

for m, n in matches:
    if m.distance < 0.8*n.distance:
        pts1.append(kp1[m.queryIdx].pt)
        pts2.append(kp2[m.trainIdx].pt)

pts1 = np.asarray(pts1)
pts2 = np.asarray(pts2)

E, _ = cv2.findEssentialMat(pts1, pts2, intrinsic_matrix) # last 3 args , cv2.RANSAC, 0.99, 0.01
_, R, t, _ = cv.recoverPose(E, pts1, pts2, intrinsic_matrix)

print(np.zeros((3,1)))
first_camera_matrix = np.hstack((intrinsic_matrix, np.zeros((3,1))))
second_camera_matrix = np.hstack((R, t.reshape((3, 1))))

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

scale = 1.0

triangulated_points = cv2.triangulatePoints(first_camera_matrix, second_camera_matrix, pts1, pts2).T

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
