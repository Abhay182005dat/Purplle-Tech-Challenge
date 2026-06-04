import torch

_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load

from ultralytics import YOLO
import cv2

VIDEO_PATH = "data/Store 1/CAM 1 - zone.mp4"
OUTPUT_PATH = "tracked_output.mp4"

model = YOLO("yolov8s.pt")

results = model.track(
    source=VIDEO_PATH,
    tracker="bytetrack.yaml",
    persist=True,
    classes=[0],      # person only
    conf=0.3,
    stream=True
)

cap = cv2.VideoCapture(VIDEO_PATH)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

writer = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

for result in results:

    frame = result.orig_img.copy()

    if result.boxes is not None:

        boxes = result.boxes.xyxy.cpu().numpy()

        if result.boxes.id is not None:
            track_ids = result.boxes.id.cpu().numpy().astype(int)
        else:
            track_ids = [-1] * len(boxes)

        confs = result.boxes.conf.cpu().numpy()

        for box, tid, conf in zip(boxes, track_ids, confs):

            x1, y1, x2, y2 = map(int, box)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"ID:{tid}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    writer.write(frame)

writer.release()
cap.release()

print(f"Saved: {OUTPUT_PATH}")