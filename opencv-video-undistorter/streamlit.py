import streamlit as st
import cv2 as cv
import numpy as np
from PIL import Image
import io
import os
import streamlit_ext as ste
import pyheif
import pathlib

def saveheif(image):
    imread = pyheif.read_heif(image)
    save = Image.frombytes(imread.mode, imread.size, imread.data)
    save.save("image.jpg")

def calibrate(matrix_x, matrix_y, calibration_img, output, video, size_sq):
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objpoints = []
    imgpoints = []
    globmtx = None
    globdist = None
    matrix_x *= size_sq
    matrix_y *= size_sq
    objp = np.zeros((matrix_x * matrix_y, 3), np.float32)
    objp[:, :2] = np.mgrid[0:matrix_x, 0:matrix_y].T.reshape(-1, 2)
    for calib_img in calibration_img:
        global runcount
        runcount = True
        st.write("Processing image: " + calib_img.name)
        fmt = pathlib.Path(calib_img.name).suffix[1:]
        print(fmt)
        if fmt.lower() in ["heic", "heif", 'avif']:
            saveheif(calib_img)
        else:
            image = Image.open(calib_img)
            image = image.save("image.jpg")
        calib_img = cv.imread("image.jpg")
        gray = cv.cvtColor(calib_img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (matrix_x, matrix_y), None)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            ret, mtx, dist, _, _ = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)    
            globmtx = mtx
            globdist = dist
            st.write(globmtx)
    os.remove("image.jpg")
    st.session_state.runcount = True
    if output:
        if globmtx is not None:
            st.balloons()
            st.write("### Calibration successful!")
            st.write("You can now download the intrinsic matrix and distortion coefficients.")
            with io.BytesIO() as mtx_file:
                np.savetxt(mtx_file, globmtx)
                mtx_file.seek(0)
                #st.download_button("Download Intrinsic Matrix", mtx_file, file_name="mtx.txt")
                ste.download_button("Download Intrinsic Matrix", mtx_file, file_name="mtx.txt")
            with io.BytesIO() as dist_file:
                np.savetxt(dist_file, globdist)
                dist_file.seek(0)
                #st.download_button("Download Distortion Coefficients", dist_file, file_name="dist.txt")
                ste.download_button("Download Distortion Coefficients", dist_file, file_name="dist.txt")
        else:
            st.write("### Calibration failed!")
            st.write("Please try again with different images.")
    else:
        if globmtx is not None:
            undistort_video(video, globmtx, globdist, True)
        else:
            st.write("### Calibration failed!")
            st.write("Please try again with different images.")

def write_bytesio_to_file(bytesio, file_name):
    with open(file_name, "wb") as f:
        f.write(bytesio.getbuffer())

def upload_mode(video, mtx, dist):
    mtx = np.loadtxt(mtx)
    dist = np.loadtxt(dist)
    undistort_video(video, mtx, dist, False)

def undistort_video(video, mtx, dist, crop):
    st.session_state.runcount = True
    print(type(mtx))
    st.write("Processing video...")
    print(mtx)
    name = "./undistort.mp4"
    write_bytesio_to_file(video, "video.mp4")
    cap = cv.VideoCapture("video.mp4")
    fps = cap.get(cv.CAP_PROP_FPS)
    size = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
    writer = cv.VideoWriter(name, cv.VideoWriter_fourcc(*'avc1'), int(fps), size)
    
    total_frames = cap.get(cv.CAP_PROP_FRAME_COUNT)
    total_frames = int(total_frames)
    print(total_frames)
    st.write("Calibrating video...")
    progress_bar = st.progress(0)
    if not cap.isOpened():
        st.write("Error opening video stream or file")
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            h,  w = frame.shape[:2]
            progress_bar.progress((int(cap.get(cv.CAP_PROP_POS_FRAMES)) / total_frames))
            newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
            dst = cv.undistort(frame, mtx, dist, None, newcameramtx)
            # if crop:
            #     x, y, w, h = roi
            #     dst = dst[y:y+h, x:x+w]
            writer.write(dst)
        else:
            break
    cap.release()
    cv.destroyAllWindows()
    #subprocess.run("ffmpeg -i undistort.mp4 -vcodec libx264 undistortcodec.mp4")
    #subprocess.call(args=f"ffmpeg -y -i {name} -c:v libx264 {convertedVideo}".split(" "))
    os.remove("video.mp4")
    writer.release()
    st.write("Video processed!")
    st.balloons()
    st.write("You can now download the processed video.")
    ste.download_button("Download Processed Video", name, file_name="undistort.mp4")
    st.video(name)

if "runcount" not in st.session_state:
    st.write("""
    # Video Calibration
    ### Undo the effects of camera distortion on videos. See this project on [GitHub](https://github.com/Piflyer/opencv-video-undistorter).
    """)
    tab1, tab2, tab3 = st.tabs(["Calibrate Camera", "Calibrate + Process Video", "Process Video with Matrices"])
    with tab1:
        st.write("""
        ## Calibrate Camera
        ### Calibrate the camera and produce files for intrinsic matrix and distortion coefficients
        """)
        with st.container():
            st.write("""
            #### Step 1: Set the number of squares in the checkerboard
            Set the number of square to calibrate the camera. It should be the number of square on each side of the checkerboard minus one.
            """)
            size_sq = st.number_input("Size of square in CM", min_value=0, max_value=10000, key="size_sq")
            matrix_x = st.number_input("Number of squares in the x direction", min_value=1, max_value=100, key="matrix_x")
            matrix_y = st.number_input("Number of squares in the y direction", min_value=1, max_value=100, key="matrix_y")
        with st.container():
            st.write("""
            #### Step 2: Upload images of a checkerboard
            Upload images of a checkerboard. The checkerboard should be fully seen in each image. The images should be taken from different angles and distances from the camera. 10-12 images should be enough.
            """)
            calibration_img = st.file_uploader("Upload images of a checkerboard", type=["jpg", "png", "jpeg", "heif", "heic"], accept_multiple_files=True, key="video_process")
            st.button("Calibrate Camera", on_click=calibrate, args=(matrix_x, matrix_y, calibration_img, True, None, size_sq), key="videobutton")

    with tab2:
        st.write("""
        ## Calibrate + Process Video
        ### Calibrate the camera and process a video
        """)
        #st.write("Coming soon...")
        with st.container():
            st.write("""
            #### Step 1: Set the number of squares in the checkerboard
            Set the number of square to calibrate the camera. It should be the number of square on each side of the checkerboard minus one.
            """)
            size_sq = st.number_input("Size of square in CM", min_value=0, max_value=10000)
            matrix_x = st.number_input("Number of squares in the x direction", min_value=1, max_value=100)
            matrix_y = st.number_input("Number of squares in the y direction", min_value=1, max_value=100)
        with st.container():
            st.write("""
            #### Step 2: Upload images of a checkerboard
            Upload images of a checkerboard. The checkerboard should be fully seen in each image. The images should be taken from different angles and distances from the camera. 10-12 images should be enough.
            """)
            calibration_img = st.file_uploader("Upload images of a checkerboard", type=["jpg", "png", "jpeg"], accept_multiple_files=True, key="calibrate_button")
        with st.container():
            st.write("""
            #### Step 2: Upload Video to Process
            Upload a video to process. The video should be taken from the same camera as the images used to calibrate the camera.
            """)
            video = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"], key="video_button")
            st.button("Process Video", on_click=calibrate, args=(matrix_x, matrix_y, calibration_img, False, video, size_sq), key="button")

    with tab3:
        st.write("""
        ## Process Video with Matrices
        ### Process a video with an existing intrinsic matrix and distortion coefficients""")
        with st.container():
            st.write("""
            #### Step 1: Upload intrinsic matrix file
            Upload a file containing the intrinsic matrix. The file should be a .txt file containing the intrinsic matrix.
            """)
            mtx = st.file_uploader("Upload intrinsic matrix file", type=["txt"], key="mtx")
        with st.container():
            st.write("""
            #### Step 2: Upload distortion coefficients file
            Upload a file containing the distortion coefficients. The file should be a .txt file containing the distortion coefficients.
            """)
            dist = st.file_uploader("Upload distortion coefficients file", type=["txt"], key="dist")
        with st.container():
            st.write("""
            #### Step 3: Upload Video to Process
            Upload a video to process. The video should be taken from the same camera as the images used to calibrate the camera.
            """)
            video = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"], key="videodist")
            st.button("Process Video", on_click=upload_mode, args=(video, mtx, dist), key="vidbutt")
