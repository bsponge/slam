import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from cv2.xfeatures2d import matchGMS
import numpy as np
import sys
import kdtree
import collections

class Mapper:
    def __init__(self):
        self.val = 'sijsflkdjfjj'

    def print(self):
        print('siema')


def match(kp1, kp2, des1, des2, shape1, shape2, orb, matcher):
    matches_all = matcher.knnMatch(des1, des2, k=2)
    #matches_all = matcher.match(des1, des2)
    #matches_gms = matchGMS(shape1[:2], shape2[:2], kp1, kp2, matches_all)
    #matches_gms = sorted(matches_gms, key = lambda x: x.distance)
    good = []
    for m,n in matches_all:
        # 0.50 ???
        if m.distance < 0.50*n.distance and m.distance > 0.05*n.distance:
            good.append(m)
    return good

def main():
    Point = collections.namedtuple('Point', 'x y z')
    tree = kdtree.create(dimensions=3)
    mapper = Mapper()
    mapper.print()
    matcher = cv.BFMatcher(cv.NORM_HAMMING)
    cap = cv.VideoCapture('movie1.mp4')

    if not cap.isOpened():
        print('Error opening video file!')
        exit(1)

    keypoints_num = 1000
    orb = cv.ORB_create(keypoints_num, fastThreshold=0)
    last_frame = None
    F = 910

    # test, should be obtained from frame info
    width = 1920 / 2
    height = 1080 / 2

    intrinsic_matrix = np.zeros((3,3))
    intrinsic_matrix[0][0] = F/width
    intrinsic_matrix[1][1] = F/height
    intrinsic_matrix[2][2] = F
    intrinsic_matrix[0][1] = 0
    intrinsic_matrix[0][2] = width / 2
    intrinsic_matrix[1][2] = height / 2

    # camera location
    camera_pose = np.empty((3,1))

    rotation_matrix = np.identity(3)
    
    # extrinsic_matrix aka view_matrix (OpenGL)
    extrinsic_matrix = np.hstack((rotation_matrix, camera_pose))

    

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

    print(intrinsic_matrix)

    file = open('points', 'w')
    points_in_frame = open('points_in_frame', 'w')

    center = None

    position = 0.0, 0.0, 0.0
    

    while cap.isOpened(): 
        # get frame
        ret, frame = cap.read()
        frame = cv.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
        kp1, kp2, matches = None, None, None

        frames = []
        # analyze first 6 frames to locate points in space with kdtree
        """
        for i in range(6):
            frames.append(cap.read()[1])

        for i in range(len(frames)-1):
            prev = frames[i]
            now = frames[i+1]
            kp1, des1 = orb.detectAndCompute(prev, None)
            kp2, des2 = orb.detectAndCompute(now, None)
            matches = match(kp1, kp2, des1, des2, prev.shape, now.shape, orb, matcher)
            
        """


        if last_frame is not None and frame is not None:
            
            # get keypoints and descriptors
            kp1, des1 = orb.detectAndCompute(last_frame, None)
            kp2, des2 = orb.detectAndCompute(frame, None)
            # get matches
            matches = match(kp1, kp2, des1, des2, last_frame.shape, frame.shape, orb, matcher)
            # iterate through matches and save matching points to the file
            points_num = 0
            for m in matches:
                if m.distance < 80:
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

                        

                        #pt[1][0] = pt[1][0] * m.distance * abs(m2) * 0.1

                        z = m.distance * abs(m2) + position[2]

                        file.write(str(pt[0][0]))
                        file.write(' ')
                        #file.write(str(3*y*(40.0/m.distance)))
                        file.write(str(pt[1][0]))
                        file.write(' ')
                        file.write(str(pt[2][0] + position[2]))
                        file.write('\n')
                        points_num = points_num + 1
                        tree.add(Point(pt[0][0], pt[1][0], z))
            points_in_frame.write(str(points_num))
            points_in_frame.write('\n')
            position = position[0], position[1], position[2] + 70
        # display frames of the video
        if ret:
            if frame is not None:
                center = (frame.shape[1]//2, frame.shape[0])
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  
            kp = orb.detect(frame, None)
            if frame is not None and kp2 is not None:
                img = cv.drawKeypoints(frame, [kp2[m.trainIdx] for m in matches], frame)
            img = frame
            if img is not None and matches is not None:
                for m in matches[100:120]:
                    cv.line(img, (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1])), (int(kp2[m.trainIdx].pt[0]), int(kp2[m.trainIdx].pt[1])), (255,0,0), 1)
                    d = kp1[m.queryIdx].pt[0] - kp2[m.trainIdx].pt[0]
                    if d != 0.0:
                        img = cv.putText(img, str((kp1[m.trainIdx].pt[1]-kp2[m.trainIdx].pt[1])/d), (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1])), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv.LINE_AA)
            # perspective lines
            cv.imshow('Frame',img)
            #sys.stdin.readline()
            last_frame = frame
            if cv.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    file.close()

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()

