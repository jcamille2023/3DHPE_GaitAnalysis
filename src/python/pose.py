# split into two: mediapipe processing and post processing
import mediapipe as mp
import cv2
import os
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
model_path = os.path.join(os.getcwd(),"pose_landmarker_full.task")
def get_mp_image(arr):
    return mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data = arr
    )

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


def blazepose(video_path):
    print(video_path)
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
            #get image height/width
            height, width = frame.shape[:2]
            # detect landmarks then add results to list for each frame
            pose_landmarks_frames.append({
                "prediction": landmarker.detect_for_video(mp_image,int((i/fps)*1e6)),
                "timestamp": i/fps
            })
            # show each frame with predicted pose landmarks
            cv2.imshow('MediaPipe Pose', draw_landmarks_on_image(frame,pose_landmarks_frames[i]["prediction"]))
            if cv2.waitKey(5) & 0xFF == 27:
                break
            i += 1
            
    print("Mediapipe analysis complete..")
    length = pose_landmarks_frames[-1]["timestamp"]
    data = {"frames":pose_landmarks_frames,
            "length":length,
            "video_height":height,
            "video_width":width}
    return data

