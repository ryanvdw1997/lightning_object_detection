# (Run this once per video to dump frames)
import cv2, os
video_num = 4

video_path = f"../data/videos/lightning_video{video_num}.mp4"
out_dir = f"../data/frames/video{video_num}_frames"
os.makedirs(out_dir, exist_ok=True)
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
sample_rate = int(fps / 5)  # e.g. if video is 30 fps, dump every 6th frame
print("We have opened the video")
idx = 0
saved = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if idx % sample_rate == 0:
        cv2.imwrite(f"{out_dir}/frame_{saved:05d}.jpg", frame)
        saved += 1
    idx += 1
print(f"Total # of Frames Saved for Video {video_num}: {saved}")
cap.release()
