import os
import json
import shutil
from pathlib import Path
from copy import deepcopy
from copy import deepcopy
from pathlib import Path
import os, json, shutil

def duplicate_coco_images_with_objects(
    image_dir,
    output_image_dir,
    num_copies=5
):
    training_videos = ['video2', 'video5', 'video6']
    for vid in training_videos:
        coco_json_path = f'../data/annotations_v3/frames_{vid}_annotations.json'
        # Load COCO annotations
        with open(coco_json_path, 'r') as f:
            coco = json.load(f)

        os.makedirs(output_image_dir, exist_ok=True)

        # Group annotations by image_id
        anns_by_image = {}
        for ann in coco['annotations']:
            anns_by_image.setdefault(ann['image_id'], []).append(ann)

        # Start building new COCO dict
        new_coco = deepcopy(coco)
        # 1) Copy all original images & annotations in full
        new_coco['images']      = deepcopy(coco['images'])
        new_coco['annotations'] = deepcopy(coco['annotations'])

        max_image_id = max(img['id'] for img in coco['images'])
        max_ann_id   = max(ann['id'] for ann in coco['annotations'])

        img_id_counter = max_image_id + 1
        ann_id_counter = max_ann_id + 1

        # 2) Only loop over images that have at least one annotation
        for img in coco['images']:
            image_id = img['id']
            filename = img['file_name']

            if image_id not in anns_by_image:
                # no upsampling for background-only images
                continue

            # Duplicate this positive image N times
            for i in range(num_copies):
                # --- new image entry ---
                new_img = deepcopy(img)
                new_filename = f"{Path(filename).stem}_dup{i}{Path(filename).suffix}"
                new_img['file_name'] = new_filename
                new_img['id'] = img_id_counter
                new_coco['images'].append(new_img)

                # copy the file
                src = os.path.join(image_dir, filename)
                dst = os.path.join(output_image_dir, new_filename)
                shutil.copy(src, dst)

                # --- new annotation entries for this image ---
                for ann in anns_by_image[image_id]:
                    new_ann = deepcopy(ann)
                    new_ann['id'] = ann_id_counter
                    new_ann['image_id'] = img_id_counter
                    new_coco['annotations'].append(new_ann)
                    ann_id_counter += 1

                img_id_counter += 1

        # Save new COCO JSON
        out_json = f'../data/annotations_v3/frames_{vid}_annotations_upsampled.json'
        with open(out_json, 'w') as f:
            json.dump(new_coco, f, indent=2)

        print(f"✅ Duplicated positive images and updated COCO JSON saved to: {out_json}")


if __name__ == '__main__':
    duplicate_coco_images_with_objects(
    image_dir='../data/images/train/',
    output_image_dir='../data/images/train/',
    num_copies=5
)
