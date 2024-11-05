import cv2
import os
import sys
# Path to the video file
video_path = sys.argv[0]

# Create a folder to store frames if it doesn't exist
output_folder = "C:/Users/jcamille2023/PycharmProjects/GhoraaniLab/Mediapipe/frames"
os.makedirs(output_folder, exist_ok=True)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Get the frames per second (fps) of the video
fps = cap.get(cv2.CAP_PROP_FPS)

# Calculate the frame interval to extract every 2 seconds
frame_interval = int(fps * 2)  # 2 seconds worth of frames

frame_count = 0
saved_frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()

    # Break the loop if we have reached the end of the video
    if not ret:
        break

    # Check if the current frame is at the specified interval
    if frame_count % frame_interval == 0:
        # Save the frame as an image file
        frame_path = os.path.join(output_folder, f"frame_{saved_frame_count:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        print(f"Saved {frame_path}")
        saved_frame_count += 1

    frame_count += 1

# Release the video capture object
cap.release()
print("All frames have been extracted and saved.")
