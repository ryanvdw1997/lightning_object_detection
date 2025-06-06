import os
import json
import cv2
import albumentations as A


import os
import json
import cv2
import numpy as np
import albumentations as A

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# 1) Paths to your original data
COCO_INPUT_JSON = "../data/annotations/frames_4_annotations.json"
ORIG_IMG_DIR    = "../data/frames/processed/images/train"

# 2) Where to write augmented images and the new JSON
OUTPUT_IMG_DIR  = "../data/frames/processed/images/train"
OUTPUT_JSON     = "../data/annotations/frames_4_annotations_augmented.json"

# 3) Gamma values to apply (all < 1.0 to darken). You can add/remove as needed.
GAMMA_VALUES    = [0.8, 0.5, 0.3]

# 4) Scale limits for “scale‐only” augment. Here ±10% scale around the center.
#    We set shift_limit=0 and rotate_limit=0 → only scaling.
SCALE_LIMIT = 0.1  # means random scale ∈ [0.9, 1.1]
ROTATE_LIMIT = 45
UP_SAMPLE_FACTOR = 30

# 5) Albumentations pipeline for “flip + scale” (applied only on images with objects)
FLIP_SCALE_PIPELINE = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.0,         # no shift
            scale_limit=SCALE_LIMIT, # only scaling
            rotate_limit=0,          # no rotation
            border_mode=cv2.BORDER_REFLECT_101,
            p=.5,
        ),
        A.Rotate(limit=ROTATE_LIMIT, p=.5)
    ],
    bbox_params=A.BboxParams(
        format="coco",            # [x_min, y_min, width, height]
        label_fields=["category_ids"],
        min_visibility=0.3,       # drop boxes if < 30% visible after transform
    ),
)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Load original COCO JSON and build lookups
# ─────────────────────────────────────────────────────────────────────────────

with open(COCO_INPUT_JSON, "r") as f:
    coco = json.load(f)

# Copy the original top‐level fields so we don’t mutate them in place
original_images      = coco["images"]
original_annotations = coco["annotations"]
categories           = coco["categories"]  # copy over as‐is
info                 = coco.get("info", {})
licenses             = coco.get("licenses", [])

# Build a lookup: image_id → image_dict
img_id_to_info = { img["id"]: img for img in original_images }

# Group annotations by image_id
anns_by_image = {}
for ann in original_annotations:
    iid = ann["image_id"]
    anns_by_image.setdefault(iid, []).append(ann)

print(f"NUMBER OF IMAGES WITH AN OBJECT: {len(list(anns_by_image.keys()))}")

# Find the next free IDs for new images & annotations
max_image_id      = max(img["id"] for img in original_images)
max_annotation_id = max(ann["id"] for ann in original_annotations)

next_image_id      = max_image_id + 1
next_annotation_id = max_annotation_id + 1

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Prepare output directories
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

# We’ll build new lists for augmented images & annotations, then append originals
new_image_entries      = []
new_annotation_entries = []

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


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Define helper functions for augmentations
# ─────────────────────────────────────────────────────────────────────────────

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

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Loop over every original image, apply gamma & (if needed) flip/scale
# ─────────────────────────────────────────────────────────────────────────────

for orig_img_dict in original_images:
    orig_id       = orig_img_dict["id"]
    orig_fname    = orig_img_dict["file_name"]
    orig_width    = orig_img_dict["width"]
    orig_height   = orig_img_dict["height"]
    orig_path     = os.path.join(ORIG_IMG_DIR, orig_fname)
    base_name, ext = os.path.splitext(orig_fname)

    # Load BGR image once
    img_bgr = load_image_bgr(orig_path)

    # ── 4a) Apply gamma to *all* images ──────────────────────────────────────
    for gamma in GAMMA_VALUES:
        aug_img = apply_gamma(img_bgr, gamma)
        gamma_str = str(gamma).replace(".", "_")  # for filename

        new_fname = f"{base_name}_gamma{gamma_str}{ext}"
        new_path  = os.path.join(OUTPUT_IMG_DIR, new_fname)

        # Save darkened image
        save_image_bgr(new_path, aug_img)

        # Create a new COCO image entry
        new_img_id = next_image_id
        next_image_id += 1

        new_image_entries.append({
            "id": new_img_id,
            "width": orig_width,
            "height": orig_height,
            "file_name": new_fname,
        })

        # Copy original annotations (bbox & category) unchanged,
        # but with new image_id and new annotation_ids
        if orig_id in anns_by_image:
            for orig_ann in anns_by_image[orig_id]:
                new_ann = copy_annotation_base(
                    ann=orig_ann,
                    new_id=next_annotation_id,
                    new_image_id=new_img_id,
                    new_bbox=orig_ann["bbox"]  # unchanged for gamma
                )
                next_annotation_id += 1
                new_annotation_entries.append(new_ann)

    # ── 4b) If this image has at least one annotation (object present),
    #         also apply horizontal flip + scale augmentations ────────────
    if orig_id in anns_by_image:
        # Build lists of bboxes and category_ids for Albumentations
        bboxes      = []
        category_ids= []
        for ann in anns_by_image[orig_id]:
            # COCO‐format: [x_min, y_min, width, height]
            bboxes.append(ann["bbox"])
            category_ids.append(ann["category_id"])
        # Repeat the same pipeline 5 times
        for repeat_idx in range(UP_SAMPLE_FACTOR):
            augmented = FLIP_SCALE_PIPELINE(
                image=img_bgr,
                bboxes=bboxes,
                category_ids=category_ids,
            )

            aug_img_bgr   = augmented["image"]
            aug_bboxes    = augmented["bboxes"]
            aug_cat_ids   = augmented["category_ids"]

            # Use repeat_idx to create a unique filename, e.g. "frame123_flipscale0.jpg"
            new_fname = f"{base_name}_flipscale{repeat_idx}{ext}"
            new_path  = os.path.join(OUTPUT_IMG_DIR, new_fname)
            save_image_bgr(new_path, aug_img_bgr)

            # Create a new COCO image entry
            new_img_id = next_image_id
            next_image_id += 1

            new_image_entries.append({
                "id": new_img_id,
                "width": orig_width,
                "height": orig_height,
                "file_name": new_fname,
            })

            # For each updated bbox, create a new annotation entry.
            for bbox_coco, cid in zip(aug_bboxes, aug_cat_ids):
                new_ann = {
                    "id": next_annotation_id,
                    "image_id": new_img_id,
                    "category_id": cid,
                    "bbox": [round(float(x), 2) for x in bbox_coco],
                    "area": round(float(bbox_coco[2] * bbox_coco[3]), 2),
                    "iscrowd": 0,
                    "segmentation": [],
                    "ignore": 0,
                }
                next_annotation_id += 1
                new_annotation_entries.append(new_ann)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Write out the combined COCO JSON
# ─────────────────────────────────────────────────────────────────────────────

# Combine originals + new entries
combined = {
    "info": info,
    "licenses": licenses,
    "images": original_images + new_image_entries,
    "annotations": original_annotations + new_annotation_entries,
    "categories": categories
}

# Make sure the output folder exists
os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

with open(OUTPUT_JSON, "w") as f:
    json.dump(combined, f, indent=2)

print(f"All augmented images saved to directory:\n  {OUTPUT_IMG_DIR}")
print(f"New COCO JSON with updated annotations written to:\n  {OUTPUT_JSON}")

with open(OUTPUT_JSON, "r") as f:
    new_coco = json.load(f)

# Copy the original top‐level fields so we don’t mutate them in place
new_images      = new_coco["images"]
new_annotations = new_coco["annotations"]
categories           = new_coco["categories"]  # copy over as‐is
info                 = new_coco.get("info", {})
licenses             = new_coco.get("licenses", [])

# Build a lookup: image_id → image_dict
img_id_to_info = { img["id"]: img for img in new_images }

# Group annotations by image_id
anns_by_image = {}
for ann in new_annotations:
    iid = ann["image_id"]
    anns_by_image.setdefault(iid, []).append(ann)

print(f"# OF IMAGES WITH OBJECTS NOW: {len(list(anns_by_image.keys()))}")
