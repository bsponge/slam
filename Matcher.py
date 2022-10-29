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

class Matcher:
    def __init__(self):
        pass

    def match(self, first_frame, second_frame):
        orb = cv2.ORB_create()
        mth = cv2.BFMatcher(cv2.NORM_HAMMING)

        pts1 = cv2.goodFeaturesToTrack(np.mean(first_frame, axis=2).astype(np.uint8), 10000, qualityLevel=0.001, minDistance=2)
        pts2 = cv2.goodFeaturesToTrack(np.mean(second_frame, axis=2).astype(np.uint8), 10000, qualityLevel=0.001, minDistance=2)

        kps1 = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=10) for f in pts1]
        kps2 = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=10) for f in pts2]

        kps1, des1 = orb.compute(first_frame, kps1)
        kps2, des2 = orb.compute(second_frame, kps2)

        matches_all = mth.knnMatch(des1, des2, k=2)
        p1 = []
        p2 = []
        for m,n in matches_all:
            if m.distance < 0.8*n.distance:
                p1.append(kps1[m.queryIdx].pt)
                p2.append(kps2[m.trainIdx].pt)

        p1 = np.asarray(p1)
        p2 = np.asarray(p2)

        return p1, p2


