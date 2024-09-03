import mediapipe as mp
import cv2
model_path = "C:/Users/jcamille2023/learningAI/pythonProject/pose_landmarker_full.task"

# Use OpenCV’s VideoCapture to load the input video.

def get_mp_image(arr):
    return mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data = arr
    )
def main():
    video_path = "C:/Users/jcamille2023/learningAI/pythonProject/videoplayback.mp4"
    base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO
    )
    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
        i = 0
        landmarker = landmarker
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            fps = cap.get(cv2.CAP_PROP_FPS)
            timestamp = mp.Timestamp(seconds=i/fps)
            mp_image = get_mp_image(frame)
            results = landmarker.detect_for_video(mp_image,timestamp)
            print(results)
            cv2.imshow('MediaPipe Pose', frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break
            i += 1


    return 0
main()
