# (Run this once per video to dump frames)
import cv2, os, json
video_num = 5

train_test_dict = {'train': [2, 5, 6],
                   'val': [1, 4]}
for key in train_test_dict.keys():
    for vid_num in train_test_dict[key]:
        annotations_path = f'../data/annotations_v3/frames_video{vid_num}_annotations.json'
        with open(annotations_path, 'r') as f:
            coco = json.load(f)
        num_annotated_images = len({img['id'] for img in coco['images']})
        fname_example = coco['images'][0]['file_name']
        num_digits_fname = len(fname_example.split('.')[0].split('_')[1])
        video_path = f"../data/videos/lightning_video{vid_num}.mp4"
        out_dir = f"../data/frames/{key}"
        os.makedirs(out_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = max(1, total_frames//num_annotated_images)
        print(f"We have opened the video {vid_num}")
        idx = 0
        saved = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if idx % interval == 0 and saved < num_annotated_images:
                if num_digits_fname == 5:
                    cv2.imwrite(f"{out_dir}/video{vid_num}_frame_{saved:05d}.jpg", frame)
                elif num_digits_fname == 6:
                    cv2.imwrite(f"{out_dir}/video{vid_num}_frame_{saved:06d}.jpg", frame)
                saved += 1
            idx += 1
        print(f"Total # of Frames Saved for Video {vid_num}: {saved}")
        cap.release()
