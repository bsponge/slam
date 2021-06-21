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


def main():
    Point = collections.namedtuple('Point', 'x y z')
    
    

    cap = cv.VideoCapture('movie1.mp4')
    if not cap.isOpened():
        print('Error opening video file!')
        exit(1)

    keypoints_num = 1000 
    last_frame = None
    # focal length
    F = 300

    # test, should be obtained from frame info
    width = 1920 / 2
    height = 1080 / 2

    intrinsic_matrix = np.array([
        [F/width,   0,          width/2],
        [0,         F/height,   height/2],
        [0,         0,          1]
    ])

    # camera location
    camera_pose = np.empty((3,1))
    rotation_matrix = np.identity(3)
    
    # extrinsic_matrix aka view_matrix (OpenGL)
    extrinsic_matrix = np.hstack((rotation_matrix, camera_pose))

    # in pixel coords
    fundamental_matrix = None
    # normalized image coords
    essential_matrix = None

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

    center = None

    position = 0.5, 0., 0.0

    orb = cv.ORB_create(keypoints_num, fastThreshold=0)
    mth = cv.BFMatcher(cv.NORM_HAMMING)
    matcher = Matcher(mth)
    

    while cap.isOpened(): 
        # get frame
        ret, frame = cap.read()
        if frame is not None:
            frame = cv.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
        kp1, kp2, matches = None, None, None

        if last_frame is not None and frame is not None: 
            points_num = 0
            kp1, des1 = orb.detectAndCompute(last_frame, None)
            kp2, des2 = orb.detectAndCompute(frame, None)

            matches = matcher.match(kp1, kp2, des1, des2)
            pts1 = []
            pts2 = []

            for m in matches:
                if m.distance < 30:
                    pts1.append(kp1[m.queryIdx].pt)
                    pts2.append(kp2[m.trainIdx].pt)

            pts1 = np.asarray(pts1)
            pts2 = np.asarray(pts2)
            print(f'pts1: {pts1.shape}, pts2: {pts2.shape}')

            print(f'lastframeshape: {last_frame.shape}, frameshape: {frame.shape}')

            if pts1.shape[0] != 0 and pts2.shape[0] != 0:
                F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_8POINT)
                inv = np.linalg.inv(intrinsic_matrix)
                E = inv.transpose().dot(F).dot(intrinsic_matrix)
                print('fundamental')
                print(F)
                print('essential')
                print(E)
                u, s, vh = np.linalg.svd(E)


            """
            for m in matches:
                #if m.distance < 80:
                if True:
                    # y = mx + b
                    div1 = center[0]-kp1[m.queryIdx].pt[0]
                    if div1 != 0.0:
                        m1 = (center[1]-kp1[m.queryIdx].pt[1]) / div1
                    div2 = center[0]-kp2[m.queryIdx].pt[0]
                    if div2 != 0.0:
                        m2 = (center[1]-kp2[m.trainIdx].pt[1]) / div2
                    # if slope difference of linear functions defined by center point and keypoint is higher than 10 degrees then discard
                    #if abs(m2-m1) < 0.20:
                    dx = abs(kp1[m.queryIdx].pt[0] - kp2[m.trainIdx].pt[0])
                    slope = 0

                    if dx != 0.0:
                        slope = (kp1[m.queryIdx].pt[1]-kp2[m.trainIdx].pt[1]) / dx


                    # ? probably doesn't matter
                    if slope < 14 and slope > -14:
                        # distance between center and keypoint from current frame
                        x, y = kp2[m.trainIdx].pt
                        distance = ((center[0] - kp2[m.trainIdx].pt[0])**2+(center[1] - kp2[m.trainIdx].pt[1])**2)**(1/2)
                        max_distance = ((center[0])**2+(center[1])**2)**(1/2)
                        m.distance > frame.shape[1]//2
                        
                        img_pt = np.zeros((3,1), dtype=float)
                        img_pt[0][0] = float(x)
                        img_pt[1][0] = float(y)
                        img_pt[2][0] = 1.0
                        pt = np.linalg.inv(intrinsic_matrix)
                        pt = pt.dot(img_pt)

                        #
                        # z = ?
                        #
                        pt[2][0] *= width*20/m.distance

                        file.write(str(pt[0][0]))
                        file.write(' ')
                        #file.write(str(3*y*(40.0/m.distance)))
                        file.write(str(pt[1][0]))
                        file.write(' ')
                        file.write(str(pt[2][0] + position[2]))
                        file.write('\n')
                        points_num = points_num + 1
            """
            
            """
            last_frame, frame = map(rgb2gray, (last_frame, frame))
            descriptor_extractor = ORB()
            print(last_frame.shape)
            descriptor_extractor.detect_and_extract(last_frame)
            keypoints_left = descriptor_extractor.keypoints
            descriptors_left = descriptor_extractor.descriptors

            descriptor_extractor.detect_and_extract(frame)
            keypoints_right = descriptor_extractor.keypoints
            descriptors_right = descriptor_extractor.descriptors
            matches = match_descriptors(descriptors_left, descriptors_right,
                            cross_check=True)

            model, inliers = ransac((keypoints_left[matches[:, 0]],
                                     keypoints_right[matches[:, 1]]),
                                     FundamentalMatrixTransform, min_samples=8,
                                     residual_threshold=1, max_trials=5000)

            inlier_keypoints_left = keypoints_left[matches[inliers, 0]]
            inlier_keypoints_right = keypoints_right[matches[inliers, 1]]

            print(f"Number of matches: {matches.shape[0]}")
            print(f"Number of inliers: {inliers.sum()}")
            """
            points_in_frame.write(str(points_num))
            points_in_frame.write('\n')
            position = position[0], position[1], position[2] + 5
        # display frames of the video
        last_frame = frame
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
    file.close()

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()

