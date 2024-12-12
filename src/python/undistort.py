import cv2 as cv
import os
def videoprocessing(mtx,dist,video_path, output_folder, name, crop=False):
    print("Processing video...")
    cap = cv.VideoCapture(video_path)
    fps = cap.get(cv.CAP_PROP_FPS)
    size = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
    os.makedirs(output_folder, exist_ok=True)
    path = os.path.join(output_folder+"\\", name)
    writer = cv.VideoWriter(path, cv.VideoWriter_fourcc(*'avc1'), int(fps), size)
    print("Calibrating...")
    if not cap.isOpened():
        print("Error opening video stream or file")
        return mtx
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            #Find new camera matrix and undistort
            newcameramtx = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))[0]
            dst = cv.undistort(frame, mtx, dist, None, newcameramtx)
            dst = cv.resize(dst, (w, h))  # Ensure output matches expected size
            print("Frame " + str(cap.get(cv.CAP_PROP_POS_FRAMES)) + " processed!")
            writer.write(dst)
        else:
            break
    cap.release()
    writer.release()
    cv.destroyAllWindows()
    print("Video Processing Done!")
    return newcameramtx