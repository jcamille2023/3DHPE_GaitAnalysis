# Serves to conduct camera calibration with the checkerboard
import os
import cv2

# splits checkerboard video into frames
def video_split(checkerboard_path):
    # Create a folder to store frames if it doesn't exist
    output_folder = os.path.join(os.getcwd(),"..\\..\\frames")
    os.makedirs(output_folder, exist_ok=True)
    
    print("Turning checkerboard videos into frames....")
    cap = cv2.VideoCapture(checkerboard_path)

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
    return output_folder
