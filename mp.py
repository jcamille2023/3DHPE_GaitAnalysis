import os
from fileinput import filename
import mediapipe as mp
import cv2
from openpyxl import Workbook
# imports for Google code
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import sys
from fractions import Fraction
model_path = os.path.join(os.getcwd(),"pose_landmarker_full.task")
video_path = sys.argv[1]
filename = sys.argv[2]
f_x = float(sys.argv[3])
f_y = float(sys.argv[4])
distance = 3


def save_to_spreadsheet(arr,length,img_height,img_width):
    print("Length: ", length)
    wb = Workbook()
    s = wb.active
    distance = float(input("Enter the distance of this person in meters/second (negative for approaching, positive for facing away from camera)"))
    s.append(["timestamp","joint","x","y","z"])
    if distance > 0:
        dist = 0
    else:
        dist = -distance
    speed = distance / length
    i_tsp =  arr[0].timestamp
    # get image center
    center = (img_width/2, img_height/2)
    # image aspect ratio
    asp_rat = Fraction(img_width,img_height)
    # calculate sensor diagonal -- FOR IPHONE 11 -- units in mm
    s_diag = (1/2.55) * 25.4
    # image diagonal
    i_diag = np.sqrt(asp_rat.numerator**2 + asp_rat.denominator**2)
    # calculating sensor scale factor
    ss_fac = s_diag / i_diag
    # calculating sensor width and height
    s_width = ss_fac*asp_rat.numerator
    s_height = ss_fac*asp_rat.denominator
    # calculating coord scale factor for width and height
    sc_fac_w = s_width / img_width
    sc_fac_h = s_height / img_height

    for idx, i in enumerate(arr):
        if len(i.pose_landmarks) == 0:
            continue
        landmarks = i.pose_landmarks[0]
        world_landmarks = i.pose_world_landmarks[0]
        tsp = i.timestamp
        for tdx, t in enumerate(landmarks):
            # calculate joint positions in pixels
            pixel_x = center[0] - t.x*img_width
            pixel_y = center[1] - t.y*img_height
            # calculate estimated depth by using avg velocity and distance
            est_z = dist + speed*(tsp - i_tsp)
            # calculate joint depth for this joint
            joint_z = est_z + (world_landmarks[tdx].z) + 1
            # calculate pixel depth for each joint for x and y axis
            pixel_depth_x = f_x * joint_z*1000 / s_width
            pixel_depth_y = f_y * joint_z*1000 / s_height
            # calculating absolute x and y coords (relative to camera)
            abs_x = (pixel_x)*pixel_depth_x/f_x
            abs_y = (pixel_y)*pixel_depth_y/f_y
            # calculating real x and y coords using scale factor
            real_x = abs_x * sc_fac_w
            real_y = abs_y * sc_fac_h
            s.append([tsp, # timestamp
                   tdx, # joint number
                   real_x/1000, # scaled x coord
                   real_y/1000, # scaled y coord
                   joint_z]) # absolute z coord
    wb.save(filename)
    print("Saved to ", filename, "!")
def get_mp_image(arr):
    return mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data = arr
    )
# Google code
def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image

# my code
fps = 0
def main():
    base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO
    )
    pose_landmarks_frames = []
    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
        i = 0

        landmarker = landmarker
        # Use OpenCV’s VideoCapture to load the input video.
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # get frames/sec
            fps = cap.get(cv2.CAP_PROP_FPS)
            # get frame count
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            # convert frame to mediapipe image class
            mp_image = get_mp_image(frame)
            #get image height/width
            height, width = frame.shape[:2]
            # detect landmarks then add results to list for each frame
            pose_landmarks_frames.append(landmarker.detect_for_video(mp_image,int((i/fps)*1e6)))
            pose_landmarks_frames[i].timestamp = i/fps
            # show each frame with predicted pose landmarks
            cv2.imshow('MediaPipe Pose', draw_landmarks_on_image(frame,pose_landmarks_frames[i]))
            if cv2.waitKey(5) & 0xFF == 27:
                break
            i += 1
    print("Saving data to spreadsheet...")
    length = frame_count / fps
    save_to_spreadsheet(pose_landmarks_frames,length, height, width)
    return 0
main()