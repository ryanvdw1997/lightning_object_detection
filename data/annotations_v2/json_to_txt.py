import json
import os

train_test_dict = {'train': ['frames_video2_annotations_upsampled.json',
                             'frames_video5_annotations_upsampled.json',
                             'frames_video6_annotations_upsampled.json'],
                   'val': ['frames_video1_annotations.json',
                           'frames_video4_annotations.json']}
for k in train_test_dict.keys():
    for annot in train_test_dict[k]:
        # 1) Load COCO JSON
        with open(annot) as f:
            coco = json.load(f)

        # 2) Build a map: image_id → (width, height, file_name)
        img_info = {img["id"]: img for img in coco["images"]}

        # 3) Prepare output folder for label TXT files
        label_dir = f"../labels/{k}"
        os.makedirs(label_dir, exist_ok=True)

        # 4) Iterate annotations and group them by image_id
        annotations_by_image = {}
        for ann in coco["annotations"]:
            img_id = ann["image_id"]
            annotations_by_image.setdefault(img_id, []).append(ann)

        # 5) For each image_id (even if it has no annotations), write a .txt
        print("Going to textify the json annotations (including empty files)...")
        for img_id, info in img_info.items():
            img_w, img_h = info["width"], info["height"]
            filename = info["file_name"]                # e.g. "frame_00005.jpg"
            base = os.path.splitext(filename)[0]        # "frame_00005"
            txt_path = os.path.join(label_dir, f"{base}.txt")

            # Get all annotations for this image_id, or empty list if none
            anns = annotations_by_image.get(img_id, [])

            with open(txt_path, "w") as f:
                for ann in anns:
                    x_min, y_min, w, h = ann["bbox"]
                    x_center = x_min + w / 2
                    y_center = y_min + h / 2
                    xc_norm = x_center / img_w
                    yc_norm = y_center / img_h
                    w_norm = w / img_w
                    h_norm = h / img_h
                    class_id = ann["category_id"] - 1

                    f.write(f"{class_id} {xc_norm:.6f} {yc_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")
            # If anns was empty, the file is created but remains empty.

        print("Done. Every image now has a corresponding .txt (possibly empty).")
