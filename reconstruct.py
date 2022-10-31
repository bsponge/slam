
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
import sys
from Analyser import *
from Matcher import *

def add_ones(x):
  if len(x.shape) == 1:
    return np.concatenate([x,np.array([1.0])], axis=0)
  else:
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

def normalize(Kinv, pts):
  return np.dot(Kinv, add_ones(pts).T).T[:, 0:2]

def reconstruct(f1, f2, K, prev_proj, matcher):
    pts1, pts2 = matcher.match(f1, f2)

    E, mask = cv2.findEssentialMat(pts1, pts2, K, cv2.RANSAC, 0.999, 1.0) # last 3 args , cv2.RANSAC, 0.99, 0.01
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

    ret, R, t, mask = cv.recoverPose(E, pts1, pts2, K, )

    first_camera_matrix = prev_proj
    second_camera_matrix = np.dot(K, np.hstack((R, t.reshape((3,1)))))

    print('first_camera_matrix')
    print(first_camera_matrix)
    print('second_camera_matrix')
    print(second_camera_matrix)

    print('matching points')
    print(pts1.shape)

    pts1 = pts1.T
    pts2 = pts2.T

    

    scale = 1.0

    triangulated_points = cv2.triangulatePoints(first_camera_matrix, second_camera_matrix, pts1, pts2).T
    for i in range(triangulated_points.shape[0]):
        triangulated_points[i] = triangulated_points[i] / np.abs(triangulated_points[i, 3])

        points_file.write(str(scale*triangulated_points[i, 0])) # 
        points_file.write(' ')
        points_file.write(str(scale*triangulated_points[i, 1])) # 
        points_file.write(' ')
        points_file.write(str(scale*triangulated_points[i, 2])) # 
        points_file.write('\n')

    points_num = triangulated_points.shape[0]
    #points_num = pts1.shape[1] + pts2.shape[1]
    points_in_frame.write(str(points_num))
    points_in_frame.write('\n')

    return second_camera_matrix

np.set_printoptions(suppress=True)

points_file = open('points', 'w')
points_in_frame = open('points_in_frame', 'w')
camera_pose_file = open('camera_pose', 'w')
camera_points_file = open('camera_points', 'w')

i1 = cv2.imread('room/r1.jpg')
i2 = cv2.imread('room/r2.jpg')
i3 = cv2.imread('room/r3.jpg')
i4 = cv2.imread('room/r4.jpg')
i5 = cv2.imread('room/r5.jpg')

height, width = i1.shape[:2]
print("height and width: ", (height,width))

# ipone 12 camera focal length in pixels
F = 3320

if len(sys.argv) > 1:
    F = int(sys.argv[1])
    print('setting F to: ', F)

print('using w, h, F: ', width, height, F)

K = np.array([
    [3320,   0,    width//2 ],
    [0,   3337,    height//2],
    [0,   0,    1        ]
])

matcher = Matcher()

print('K matrix')
print(K)

first_pose = np.hstack((K,np.zeros((3,1))))

proj = reconstruct(i1, i2, K, first_pose, matcher)
proj = reconstruct(i2, i3, K, proj, matcher)
proj = reconstruct(i3, i4, K, proj, matcher)
proj = reconstruct(i4, i5, K, proj, matcher)

points_file.close()
points_in_frame.close()
camera_pose_file.close()
camera_points_file.close()

'''
# draw keypoints on image
img = cv.drawKeypoints(second_img, kp2, second_img)
cv2.imshow('frame', img)
cv.waitKey(25)
while True:
    pass
'''