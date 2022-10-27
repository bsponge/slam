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
from Analyser import *
from Matcher import *

def normalize(pts, width, height):
    pts[0,:] /= width
    pts[1,:] /= height
    return pts


def main():
    Point = collections.namedtuple('Point', 'x y z')
    
    

    cap = cv.VideoCapture('movie.mov')
    if not cap.isOpened():
        print('Error opening video file!')
        exit(1)

    keypoints_num = 10000
    last_frame = None
    # focal length
    F = 900

    # test, should be obtained from frame info
    width = 1920 / 2
    height = 1080 / 2

    intrinsic_matrix = np.array([
        [F/width,   0,          width/2],
        [0,         F/height,   height/2],
        [0,         0,          1]
    ])

    # camera location
    IRt = np.eye(4)[:3]

    camera_pose = IRt
    rotation_matrix = np.identity(3)

    """
    One way to get a 3D position from a pair of matching points from two images is to take the fundamental matrix,
    compute the essential matrix, and then to get the rotation and translation between the cameras from the essential matrix.
    This, of course, assumes that you know the intrinsics of your camera.
    Also, this would give you up-to-scale reconstruction, with the translation being a unit vector.
    """

    # camera coordinate system X,Y,Z    <--------
    #                                           |
    # image coordinates system U,V              |
    # |u'|                       |X|            V
    # |v'| = intrinsic_matrix *  |Y| <-- (camera coordinate system)
    # |w'|                       |Z|
    #
    # u = u'/w'
    # v = v'/w'

    # relation between camera coordinates system and world coordinates system
    # is camera_coord_system = rotation_matrix * world_coord_system - rotation_matrix * camera_pose

    # 3D points in world coordinate system should be extrinsic_matrix**(-1) * (u, v, 1)

    file = open('points', 'w')
    points_in_frame = open('points_in_frame', 'w')
    camera_pose_file = open('camera_pose', 'w')
    camera_points_file = open('camera_points', 'w')

    center = None

    position = np.array([0.0, 0.0, 0.0])
    rotations = []
    translations = []
    projections = []

    orb = cv.ORB_create(keypoints_num, fastThreshold=0)
    mth = cv.BFMatcher(cv.NORM_HAMMING)
    matcher = Matcher(mth)
    

    while cap.isOpened(): 
        # get frame
        ret, frame = cap.read()
        if frame is not None:
            print(frame.shape)
            frame = cv.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
        kp1, kp2, matches = None, None, None
        

        if last_frame is not None and frame is not None: 
            points_num = 1
            kp1, des1 = orb.detectAndCompute(last_frame, None)
            kp2, des2 = orb.detectAndCompute(frame, None)


            matches = matcher.match(kp1, kp2, des1, des2)
            pts1 = []
            pts2 = []

            for m in matches:
                if m.distance < 10:
                    pts1.append(kp1[m.queryIdx].pt)
                    pts2.append(kp2[m.trainIdx].pt)

            pts1 = np.asarray(pts1)
            pts2 = np.asarray(pts2)

            if pts1.shape[0] != 0 and pts2.shape[0] != 0:
                E, mask = cv.findEssentialMat(pts1, pts2, intrinsic_matrix, cv.LMEDS, 0.999999, 1.0)
                ptss1 = []
                ptss2 = []
                for i in range(len(mask)):
                    if mask[i][0] == 0:
                        ptss1.append(pts1[i])
                        ptss2.append(pts2[i])
                pts1 = np.asarray(ptss1)
                pts2 = np.asarray(ptss2)
                retval, R, t, mask = cv.recoverPose(E, pts1, pts2, intrinsic_matrix)
                ptss1 = []
                ptss2 = []
                for i in range(len(mask)):
                    if mask[i][0] == 0:
                        ptss1.append(pts1[i])
                        ptss2.append(pts2[i])
               
                rotations.append(R)
                translations.append(t)
                projections.append(intrinsic_matrix.dot(np.hstack((R, t))))

                """
                pts1[:,0] /= width
                pts1[:,1] /= height
                pts2[:,0] /= width
                pts2[:,1] /= height
                """


                #t = R.dot(t.flatten())
                #position += t
                print('t')
                print(t)
                
                
                p1 = np.hstack((R,t))
                print('p1')
                print(p1)

                """
                pts1 = cv.undistortPoints(pts1, intrinsic_matrix, None)
                pts2 = cv.undistortPoints(pts2, intrinsic_matrix, None)
                print(pts1.shape)
                """

                pts1 = pts1.reshape(pts1.shape[1], pts1.shape[0])
                pts2 = pts2.reshape(pts2.shape[1], pts2.shape[0])

                pts1 = normalize(pts1, width, height)
                pts2 = normalize(pts2, width, height)
                scale = 1
                position += t.flatten()
                

                camera_pose_file.write(str(scale*position[0]))
                camera_pose_file.write(' ')
                camera_pose_file.write(str(scale*position[1]))
                camera_pose_file.write(' ')
                camera_pose_file.write(str(scale*position[2]))
                camera_pose_file.write('\n')

                camera_points_file.write(str(1))
                camera_points_file.write('\n')

                
                if len(projections) > 1:
                    triangulated_points = cv.triangulatePoints(IRt, p1, pts1, pts2)
                    print(triangulated_points.shape)
                    for i in range(triangulated_points.shape[1]):
                        file.write(str(scale*triangulated_points[0,i]/triangulated_points[3,i]))
                        file.write(' ')
                        file.write(str(scale*triangulated_points[1,i]/triangulated_points[3,i]))
                        file.write(' ')
                        file.write(str(scale*triangulated_points[2,i]/triangulated_points[3,i]))
                        file.write('\n')
                    points_num = triangulated_points.shape[1]
                    points_in_frame.write(str(points_num))
                    points_in_frame.write('\n')
                    print(triangulated_points.shape[1])
            
            
        # display frames of the video
        
        if ret:
            """
            kp = orb.detect(frame, None)
            if frame is not None and kp2 is not None:
                img = cv.drawKeypoints(frame, [kp2[m.trainIdx] for m in matches], frame)
            img = frame
            if img is not None and matches is not None:
                for m in matches[100:110]:
                    cv.line(img, (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1])), (int(kp2[m.trainIdx].pt[0]), int(kp2[m.trainIdx].pt[1])), (255,0,0), 1)
                    d = kp1[m.queryIdx].pt[0] - kp2[m.trainIdx].pt[0]
                    if d != 0.0:
                        img = cv.putText(img, str((kp1[m.trainIdx].pt[1]-kp2[m.trainIdx].pt[1])/d), (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1])), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1, cv.LINE_AA)
            # perspective lines
            cv.imshow('Frame',img)
            #sys.stdin.readline()
            last_frame = frame
            """
            if cv.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
        last_frame = frame
    file.close()

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()

