from checkerboard import video_split
from calibration import calibrate
from undistort import videoprocessing
from pose import blazepose
from processing import data_processing
from gait_measures import gait_measures
import glob
import os
#checkerboard_path = input("Paste ABSOLUTE path to checkerboard video.\n")
videos_path = input("Paste ABSOLUTE path to gait videos.\n") + "\\"
checkerboard_corner_x = int(input("Enter number of checkerboard corners on x axis.\n"))
checkerboard_corner_y = int(input("Enter number of checkerboard corners on y axis.\n"))
print(os.getcwd())
checkerboard_frames_path = os.getcwd()+"\\frames"
#checkerboard_frames_path = video_split(checkerboard_path)
mtx, dist, rvecs, tvecs = calibrate(checkerboard_corner_x,checkerboard_corner_y,checkerboard_frames_path)
video_output_folder = os.path.join(videos_path,"..\\undistorted_videos")
video_gait_data = []

for fname in os.listdir(videos_path):
    mtx = videoprocessing(mtx,dist,videos_path+fname,video_output_folder,fname[:-3]+"undistorted.mp4")
    joint_data = blazepose(video_output_folder + "\\" + fname[:-3]+"undistorted.mp4")
    path_length = float(input("Enter the length of the path traveled relative to the camera. Negative if away, positive if approaching."))                
    new_joint_data = data_processing(joint_data,mtx,path_length)
    measures = gait_measures(new_joint_data,"measures.xlsx")
    video_gait_data.append({"video":fname,
                            "measures":measures})
    print(f'Data for video {fname}: ', measures)
for i, data in enumerate(video_gait_data):
    print(f'Data for video {i}:', data.measures)
    


