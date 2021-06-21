import kdtree
import numpy as np
import cv2 as cv
from cv2.xfeatures2d import matchGMS

class Analyser:
    def __init__(self, _intrinsic_matrix, _extrinsic_matrix):
        self.frames = []
        self.tree = kdtree.create(dimensions=3)
        self.intrinsic_matrix = _intrinsic_matrix
        self.inv_intrinsic = np.linalg.inv(_intrinsic_matrix)
        self.extrinsic_matrix = _extrinsic_matrix

    def analyse(self, frame, kp1, kp2, matches):
        for m in matches:
            #x, y, z = kp2[m.queryIdx].pt

            #self.tree.add((x, y, z))
            pass

