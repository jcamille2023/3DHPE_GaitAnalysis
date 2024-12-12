import numpy as np
import cv2 as cv
import os
import glob
import sys
import re
import shutil
from pathlib import Path 
import math
#import argparse

'''
This script is used to undistort a video using a checkerboard calibration. This based on the OpenCV tutorial:
https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
The script can be used in 3 different ways:
1. Calibrate the camera using a checkerboard pattern
2. Undistort a video using the calibration
3. Create a video from a folder of frames
'''

file_pattern = re.compile(r'.*?(\d+).*?')
def get_order(file):
    match = file_pattern.match(Path(file).name)
    if not match:
        return math.inf
    return int(match.groups()[0])

def calibrate(matrix_x, matrix_y, image_dir, output_folder):
    global globmtx
    global globdist
    globmtx = None
    globdist = None
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    matrix_x = int(matrix_x)
    matrix_y = int(matrix_y)
    objp = np.zeros((matrix_x * matrix_y, 3), np.float32)
    objp[:, :2] = np.mgrid[0:matrix_x, 0:matrix_y].T.reshape(-1, 2)
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    images = os.listdir(image_dir)
    if len(images) == 0:
        print("No images found!")
        exit()
    for fname in images:
        print("Processing image: " + fname)
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (matrix_x, matrix_y), None)
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
        print("Saving calibration...")
        np.savetxt(output_folder + "mtx.txt", globmtx)
        np.savetxt(output_folder + "dist.txt", globdist)
        print("Camera Calibration Done!")


def videoprocessing(video_path, output_folder, name, crop=False):
    print("Processing video...")
    cap = cv.VideoCapture(video_path)
    fps = cap.get(cv.CAP_PROP_FPS)
    size = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
    path = os.path.join(output_folder, name + ".mp4")
    writer = cv.VideoWriter(path, cv.VideoWriter_fourcc(*'avc1'), int(fps), size)
    print("Calibrating...")
    try: 
        globdist
        globmtx
    except:
        dist_path = os.path.join(output_folder, "dist.txt")
        # Create an empty file
        globdist = np.loadtxt(dist_path)
        mtx_path = os.path.join(output_folder, "mtx.txt")
        globmtx = np.loadtxt(mtx_path)
    if not cap.isOpened():
        print("Error opening video stream or file")
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            #Find new camera matrix and undistort
            newcameramtx, roi = cv.getOptimalNewCameraMatrix(globmtx, globdist, (w, h), 1, (w, h))
            dst = cv.undistort(frame, globmtx, globdist, None, newcameramtx)
            # crop the image
            # if crop:
            #     x, y, w, h = roi
            #     dst = dst[y:y + h, x:x + w]
            # save the frame
            print("Frame " + str(cap.get(cv.CAP_PROP_POS_FRAMES)) + " processed!")
            writer.write(dst)
            # cv.imwrite(frame_path + "/frame" + str(i) + ".jpg", dst)
            # i += 1
            # print("Frame " + str(i) + " processed!")
            
        else:
            break
    cap.release()
    writer.release()
    cv.destroyAllWindows()
    print("Video Processing Done!")
    #frames2video(output_folder, fps, name)

def frames2video(path, fps, name):
    print("Creating video...")
    img_array = []
    size = None
    frame_path = os.path.join(path, "frames/*.jpg")
    for filename in sorted(glob.glob(frame_path), key=get_order):   
        print("Processing frame: " + filename )    
        img = cv.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)
    path = os.path.join(path, name)
    out = cv.VideoWriter(path + '.mp4', cv.VideoWriter_fourcc(*'avc1'), int(fps), size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    print("Video Created!")


# parser = argparse.ArgumentParser(description='Calibrate and process video')
# parser.add_argument('--all', nargs=5, help='Calibrate and process video')
# parser.add_argument('--calibrate', nargs=5, help='Calibrate camera')
# parser.add_argument('--videoprocessing', nargs=5, help='Process video')
# parser.add_argument('--frame2videos', nargs=4, help='Convert frames to video')
# args = parser.parse_args()
# if args.calibrate:
#     calibrate(args.calibrate[0], args.calibrate[1], args.calibrate[2], args.calibrate[3])
# elif args.videoprocessing:
#     videoprocessing(args.videoprocessing[0], args.videoprocessing[1], args.videoprocessing[2], args.videoprocessing[3])
# elif args.frame2videos:
#     frames2video(args.frame2videos[0], args.frame2videos[1], args.frame2videos[2], args.frame2videos[3])
# elif args.all:
#     calibrate(args.all[0], args.all[1], args.all[2], args.all[3])
#     videoprocessing(args.all[4], args.all[3], args.all[2])
#     
if sys.argv[1] == "--calibrate":
    calibrate(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
elif sys.argv[1] == "--videoprocessing":
    videoprocessing(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
elif sys.argv[1] == "--frames2video":
    frames2video(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
elif sys.argv[1] == "--all":
    print(sys.argv)
    calibrate(sys.argv[3], sys.argv[4], sys.argv[2], sys.argv[5])
    videoprocessing(sys.argv[6], sys.argv[5], sys.argv[7], sys.argv[8])
    exit(0)
# else:
#     print("Command not found. Here are a list of commands:")
#     print("--all CHECKERBOARD_FOLDER CHECKERBOARD_SIZE_X CHECKERBOARD_SIZE_Y OUTPUT_FOLDER VIDEO_INPUT:     Calibrate the camera, process a video, and create a video from a folder of frames")
#     print("--calibrate CHECKERBOARD_FOLDER_PATH CHECKERBOARD_SIZE_X CHECKERBOARD_SIZE_Y OUTPUT_FOLDER:     Calibrate the camera and produce an intrinsic matrix and distortion coefficients")
#     print("--videoprocessing VIDEO_PATH OUTPUT_FOLDER NAME CROP:     Process a video with an existing intrinsic matrix and distortion coefficients")
#     print("--frames2video FRAMES_PATH FPS NAME:     Create a video from a folder of frames")
#     exit()



#calibrate(CHECKERBOARD_SIZE_X, CHECKERBOARD_SIZE_Y, CHECKERBOARD_FOLDER)
#videoprocessing(FILE_INPUT, OUTPUT_FOLDER)
#frames2video(OUTPUT_FOLDER)


