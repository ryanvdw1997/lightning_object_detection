from yolov5 import train

if __name__ == "__main__":
    train.run(
        img_size=640,
        batch_size=16,
        epochs=20,
        data="lightning.yaml",
        weights="yolov5s.pt",
        name="lightning_run"
    )