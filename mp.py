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

model_path = os.path.join(os.getcwd(),"pose_landmarker_full.task")
video_path = sys.argv[1]
filename = sys.argv[2]
distance = 3


def save_to_spreadsheet(arr,length):
    print("Length: ", length)
    wb = Workbook()
    s = wb.active
    speed = float(input("Enter the speed of this person in meters/second (negative for approaching, positive for facing away from camera)"))
    s.append(["timestamp","joint","x","y","z"])
    i_tsp =  arr[0].timestamp
    for idx, i in enumerate(arr):
        if len(i.pose_landmarks) == 0:
            continue
        landmarks = i.pose_world_landmarks[0]

        tsp = i.timestamp
        for tdx, t in enumerate(landmarks):
            print(t.z)
            arr = [tsp, # timestamp
                   tdx, # joint number
                   t.x, # scaled x coord
                   t.y, # scaled y coord
                   distance - speed*(tsp - i_tsp) + (t.z)] # absolute z coord
            s.append(arr)
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
            # convert frame to mediapipe image class
            mp_image = get_mp_image(frame)
            # detect landmarks then add results to list for each frame
            pose_landmarks_frames.append(landmarker.detect_for_video(mp_image,int((i/fps)*1e6)))
            pose_landmarks_frames[i].timestamp = i/fps
            # show each frame with predicted pose landmarks
            cv2.imshow('MediaPipe Pose', draw_landmarks_on_image(frame,pose_landmarks_frames[i]))
            if cv2.waitKey(5) & 0xFF == 27:
                break
            i += 1
    print("Saving data to spreadsheet...")
    length = fps/i
    save_to_spreadsheet(pose_landmarks_frames,length)
    return 0
main()