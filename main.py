import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from cv2.xfeatures2d import matchGMS
import numpy as np

def match(kp1, kp2, des1, des2, shape1, shape2, orb, matcher):
    matches_all = matcher.match(des1, des2)
    matches_gms = matchGMS(shape1[:2], shape2[:2], kp1, kp2, matches_all)
    return matches_gms

def main():
    matcher = cv.BFMatcher(cv.NORM_HAMMING)
    cap = cv.VideoCapture('movie.mov')

    if not cap.isOpened():
        print('Error opening video file!')
        exit(1)

    keypoints_num = 2000
    orb = cv.ORB_create(keypoints_num, fastThreshold=0)
    last_frame = None
    F = 400

    # test, should be obtained from frame info
    width = 1920 / 2
    height = 1080 / 2

    intrinsic_matrix = np.zeros((3,3))
    intrinsic_matrix[0][0] = F/width
    intrinsic_matrix[1][1] = F/height
    intrinsic_matrix[2][2] = 1
    intrinsic_matrix[0][1] = 0
    intrinsic_matrix[0][2] = width / 2
    intrinsic_matrix[1][2] = height / 2

    # camera location
    camera_pose = np.empty((3,1))

    rotation_matrix = np.identity(3)
    
    # extrinsic_matrix aka view_matrix (OpenGL)
    extrinsic_matrix = np.hstack((rotation_matrix, camera_pose))

    

    # camera coordinate system X,Y,Z
    #                                           |
    # image coordinates system U,V              |
    # |u'|                       |X|             V
    # |v'| = intrinsic_matrix *  |Y| <-- (camera coordinate system)
    # |w'|                       |Z|
    #
    # u = u'/w'
    # v = v'/w'

    # relation between camera coordinates system and world coordinates system
    # is camera_coord_system = rotation_matrix * world_coord_system - rotation_matrix * camera_pose

    # 3D points in reality should be extrinsic_matrix**(-1) * (u, v, 1)

    print(intrinsic_matrix)

    file = open('points', 'w')
    points_in_frame = open('points_in_frame', 'w')

    center = None

    position = 0.0, 0.0, 0.0
    

    while cap.isOpened(): 
        # get frame
        ret, frame = cap.read()
        frame = cv.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
        if last_frame is not None and frame is not None:
            
            # get keypoints and descriptors
            kp1, des1 = orb.detectAndCompute(last_frame, None)
            kp2, des2 = orb.detectAndCompute(frame, None)
            # get matches
            matches = match(kp1, kp2, des1, des2, last_frame.shape, frame.shape, orb, matcher)
            # iterate through matches and save matching points to the file
            points_num = 0
            for m in matches:
                if m.distance < 40:
                    dx, dy = kp1[m.trainIdx].pt[0] - kp2[m.queryIdx].pt[0], kp1[m.trainIdx].pt[1] - kp2[m.queryIdx].pt[1]
                    # y = mx + b
                    div1 = center[0]-kp1[m.trainIdx].pt[0]
                    if div1 != 0.0:
                        m1 = (center[1]-kp1[m.trainIdx].pt[1]) / div1
                    div2 = center[0]-kp2[m.queryIdx].pt[0]
                    if div2 != 0.0:
                        m2 = (center[1]-kp2[m.queryIdx].pt[1]) / div2
                    # if slope difference of linear functions defined by center point and keypoint is higher than 10 degrees then discard
                    if abs(m2-m1) < 0.36:
                        # distance between center and keypoint from current frame
                        x, y = kp2[m.queryIdx].pt
                        distance = ((center[0] - kp2[m.queryIdx].pt[0])**2+(center[1] - kp2[m.queryIdx].pt[1])**2)**(1/2)
                        max_distance = ((center[0])**2+(center[1])**2)**(1/2)
                        m.distance > frame.shape[1]//2
                        z = m.distance * distance + position[2]
                        #file.write(' '.join(str(f*10.0) for f in kp2[m.queryIdx].pt))
                        img_pt = np.zeros((3,1), dtype=np.float)
                        img_pt[0][0] = float(x)
                        img_pt[1][0] = float(y)
                        img_pt[2][0] = 1.0
                        pt = np.linalg.inv(intrinsic_matrix)
                        pt = pt.dot(img_pt)

                        file.write(str(pt[0][0]))
                        file.write(' ')
                        #file.write(str(3*y*(40.0/m.distance)))
                        file.write(str(pt[1][0]))
                        file.write(' ')
                        file.write(str(z))
                        file.write('\n')
                        points_num = points_num + 1
            points_in_frame.write(str(points_num))
            points_in_frame.write('\n')
            position = position[0], position[1], position[2] + 50
        # display frames of the video
        if ret:
            center = (frame.shape[1]//2, frame.shape[0]-F)
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  
            kp = orb.detect(frame, None)
            img = cv.drawKeypoints(frame, kp, frame)
            # perspective lines
            cv.line(img, (0,img.shape[0]-F), (img.shape[1], img.shape[0]-F), (255,0,0), 1)
            cv.line(img, (img.shape[1]//2,0), (img.shape[1]//2, img.shape[0]), (255,0,0), 1)
            cv.line(img, (0,0), center, (255,0,0), 1)
            cv.line(img, (img.shape[1],0), center, (255,0,0), 1)
            
            cv.imshow('Frame', img)
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

