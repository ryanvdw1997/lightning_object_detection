import cv2
import torch
import numpy as np

WEIGHTS_PATH = 'latest_training_run/lightning_run6/weights/best.pt'

VIDEO_SOURCE = 'data/videos/lightning_video3.mp4' 

model = torch.hub.load("ultralytics/yolov5", "custom", path=WEIGHTS_PATH)

APPLY_GAMMA = False
GAMMA = .2

CONF_THRESHOLD = .3


def adjust_gamma(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Apply gamma correction to a BGR or RGB image (uint8).
    The same function should be used at train time and inference if you trained with gamma != 1.0.
    """
    invGamma = 1.0 / gamma
    table = (np.array([((i / 255.0) ** invGamma) * 255
                       for i in np.arange(256)]).astype("uint8"))
    return cv2.LUT(image, table)

def load_model(weights_path: str):
    """
    Load a custom YOLOv5 model via TorchHub. 
    Returns a model object configured to run on GPU if available.
    """
    model = torch.hub.load("ultralytics/yolov5", "custom", path=weights_path)
    model.conf = CONF_THRESHOLD  # confidence threshold
    model.iou = 0.45             # NMS IoU threshold (default)
    return model

def main():
    # (1) Load YOLOv5 model
    model = load_model(WEIGHTS_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # (2) Open the video source
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"❌ ERROR: cannot open video source {VIDEO_SOURCE}")
        return

    # (3) Prepare display window
    cv2.namedWindow("Lightning Detection", cv2.WINDOW_NORMAL)

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break  # end of video or camera disconnected

        # (4) Optional: gamma correction (if you trained with gamma!=1.0)
        if APPLY_GAMMA and GAMMA != 1.0:
            # If your `adjust_gamma` expects BGR, feed frame_bgr directly
            frame_gc = adjust_gamma(frame_bgr, GAMMA)
        else:
            frame_gc = frame_bgr

        # (5) Convert BGR→RGB for YOLOv5
        frame_rgb = cv2.cvtColor(frame_gc, cv2.COLOR_BGR2RGB)

        # (6) Run inference (YOLOv5 handles letterbox→640×640 internally)
        #     Results are returned with boxes mapped back to original frame size.
        results = model(frame_rgb, size=640)  # size must match training size

        # (7) Parse detections and draw on the original BGR frame
        detections = results.xyxy[0].cpu().numpy()
        # detections shape: (N, 6) where each row = [x1, y1, x2, y2, conf, cls]

        for *box, conf, cls in detections:
            x1, y1, x2, y2 = map(int, box)
            class_list = ['lightning-bolt', 'tree', 'building']
            class_colors = {'lightning-bolt': (199, 204, 55), 
                       'tree': (26, 173, 70), 
                       'building': (92, 83, 83)}
            # Draw a green rectangle around the predicted lightning
            class_detected = int(cls)
            # Put a label with confidence
            class_name = class_list[class_detected]
            class_color = class_colors[class_name]
            label = f"{class_name} {conf:.2f}"
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), class_color, 2)
            
            
            cv2.putText(
                frame_bgr,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                class_color,
                3,
            )

        # (8) Display FPS (optional)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cv2.putText(
            frame_bgr,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
        )

        # (9) Show the frame
        cv2.imshow("Lightning Detection", frame_bgr)

        # (10) Break on 'q' key
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

