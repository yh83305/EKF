import pyrealsense2 as rs
import apriltag
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

if __name__ == "__main__":
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
    # Start streaming
    pipeline.start(config)

    mtx = np.array([[605.932251, 0.000000, 320.847046],
                    [0.000000, 605.483582, 240.420425],
                    [0.000000, 0.000000, 1.000000]])
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        atd = apriltag.Detector(apriltag.DetectorOptions(families='tag36h11 tag25h9'))
        tags = atd.detect(gray)

        print(tags)

        if len(tags) == 0:
            cv2.imshow("result", color_image)
            cv2.waitKey(1)
            continue

        homo = tags[0].homography
        num, Rs, Ts, Ns = cv2.decomposeHomographyMat(homo, mtx)
        r = R.from_matrix(Rs[0].T)
        euler = r.as_euler('xyz').T * 180 / np.pi
        angle = euler[2]
        for c in tags[0].corners:
            cv2.circle(color_image, tuple(c.astype(int)), 4, (255, 0, 0), 2)
        cc = tags[0].center
        cv2.circle(color_image, tuple(cc.astype(int)), 6, (20, 200, 120), 2)
        ARROW_LENGTH = 150
        shiftx = np.sin(angle * np.pi / 180) * ARROW_LENGTH
        shifty = ARROW_LENGTH / 2 * np.cos(angle * np.pi / 180)
        newcenter = np.array([shiftx, shifty]) + cc
        cv2.circle(color_image, tuple(newcenter.astype(int)), 8, (0, 0, 255), 5)
        cv2.line(color_image, tuple(newcenter.astype(int)), tuple(cc.astype(int)), (0, 0, 255), 2)
        cv2.imshow("result", color_image)
        cv2.waitKey(1)
