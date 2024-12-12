# serves to conduct actual camera calibration
import cv2 as cv
import glob
import numpy as np
import os
def calibrate(checkerboard_x, checkerboard_y, image_dir):
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((checkerboard_x * checkerboard_y, 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_x, 0:checkerboard_y].T.reshape(-1, 2)
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    images = os.listdir(image_dir)
    if len(images) == 0:
        print("No images found!")
        return [0,0,0,0]
    for fname in images:
        print("Processing image: " + image_dir + "\\" + fname)
        img = cv.imread(image_dir + "\\" + fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (checkerboard_x, checkerboard_y), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)    
            globmtx = mtx
            globdist = dist
        else:
            print("Couldn't find checkerboard!")
    if globmtx is None:
        print("Couldn't perform checkerboard camera calibration!")
        print("Please check your images and try again.")
    else:
        print("Camera Calibration Done!")
        return [mtx, dist, rvecs, tvecs]