import numpy as np
import cv2
import json
import os
import re


# Helper: Create a simple lookup to copy over annotation fields we need
def copy_annotation_base(ann: dict, new_id: int, new_image_id: int, new_bbox: list[float]):
    """
    Copy everything except id, image_id, and bbox (we overwrite those).
    Recompute area = width * height.
    """
    width, height = new_bbox[2], new_bbox[3]
    area = width * height

    return {
        "id": new_id,
        "image_id": new_image_id,
        "category_id": ann["category_id"],
        "bbox": [round(float(x), 2) for x in new_bbox],
        "area": round(float(area), 2),
        "iscrowd": ann.get("iscrowd", 0),
        # If you have segmentation polygons, you could copy & transform them too,
        # but here we assume bounding‐box‐only. If your ann has "segmentation",
        # you could copy ann["segmentation"] verbatim (it won’t be correct spatially
        # unless you run a polygon‐transform routine).
        "segmentation": ann.get("segmentation", []),
        "ignore": ann.get("ignore", 0),
    }

def load_image_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at: {path}")
    return img

def save_image_bgr(path: str, img: np.ndarray) -> None:
    cv2.imwrite(path, img)

def apply_gamma(image_bgr: np.ndarray, gamma: float) -> np.ndarray:
    """
    Apply gamma correction. A gamma < 1 darkens; gamma > 1 brightens.
    We'll build a lookup table once per gamma for speed.
    """
    inv_gamma = 1.0 / gamma
    table = (np.arange(256, dtype=np.float32) / 255.0) ** inv_gamma
    table = np.clip((table * 255.0), 0, 255).astype(np.uint8)
    return cv2.LUT(image_bgr, table)

def amend_wrong_fn_in_annotations():
    src_dir = '../data/annotations_v3'
    for annot in os.listdir(src_dir):
        if '.json' not in annot:
            continue 
        # Load the original JSON file
        annot_path = os.path.join(src_dir, annot)
        print(annot_path)
        with open(annot_path, 'r') as f:
            data = json.load(f)

        # Update file_name for each image
        for image in data.get("images", []):
            if image["file_name"].endswith(".png"):
                image["file_name"] = image["file_name"].replace(".png", ".jpg")

        # Save the modified JSON to a new file
        with open(annot_path, 'w') as f:
            json.dump(data, f, indent=2)

def amend_wrong_fn_in_aug_annots(specific_file=None):
    src_dir = '../data/annotations_v3'
    for annot in os.listdir(src_dir):
        if specific_file:
            if annot != specific_file:
                continue
        # elif 'augmented.json' not in annot:
        #     continue 
        # Load the original JSON file
        annot_path = os.path.join(src_dir, annot)
        print(annot_path)
        with open(annot_path, 'r') as f:
            data = json.load(f)
        print(annot)
        video_num = re.search(r"(video\d)", annot).group(1)
        # Update file_name for each image
        for image in data.get("images", []):
            if not image["file_name"].startswith(f"{video_num}"):
                image["file_name"] = video_num+"_"+image["file_name"]
        f.close()
        # Save the modified JSON to a new file
        with open(annot_path, 'w') as f:
            json.dump(data, f, indent=2)
        f.close()

if __name__ == '__main__':
    amend_wrong_fn_in_annotations()
