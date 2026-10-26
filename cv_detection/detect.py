"""
Real-time object detection with YOLOv8 + OpenCV.

Why YOLOv8 for tonight: it's pretrained on COCO (80 everyday object
classes -- person, car, bottle, chair, etc.), runs fine on CPU for a
demo, and the ultralytics API wraps model loading + inference +
drawing boxes in a few lines, so you spend your time understanding the
pipeline instead of fighting boilerplate.

Pipeline (this is the part to be able to explain in an interview):
  1. Capture a frame from the camera (or read an image/video file)
  2. Resize/normalize happens internally in the model's preprocessing
  3. Forward pass through the CNN -> raw predictions (boxes, class
     scores, objectness)
  4. Non-max suppression collapses overlapping boxes into one per object
  5. Draw the surviving boxes + labels + confidence back onto the frame

Usage:
    python detect.py                  # webcam, live
    python detect.py --source photo.jpg
    python detect.py --source video.mp4
    python detect.py --source 0 --save   # save annotated output to runs/

Install first (see ../README.md):
    pip install ultralytics opencv-python
"""

import argparse
import time

import cv2
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description='YOLOv8 object detection demo')
    parser.add_argument('--source', default='0',
                         help="'0' for webcam, or a path to an image/video file")
    parser.add_argument('--model', default='yolov8n.pt',
                         help='model weights (yolov8n.pt = smallest/fastest, downloads automatically on first run)')
    parser.add_argument('--conf', type=float, default=0.4,
                         help='confidence threshold for keeping a detection')
    parser.add_argument('--save', action='store_true',
                         help='save annotated frames instead of (or in addition to) showing them')
    args = parser.parse_args()

    # Webcam index needs to be an int; file paths stay as strings
    source = int(args.source) if args.source.isdigit() else args.source

    print(f'[detect] loading model: {args.model}')
    model = YOLO(args.model)

    print(f'[detect] opening source: {source}')
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f'[detect] ERROR: could not open source {source}. '
              f'If running inside WSL2, webcam passthrough can be finicky -- '
              f'try `python detect.py --source some_photo.jpg` first to confirm '
              f'the pipeline works, then sort out camera access separately.')
        return

    prev_time = time.time()
    frame_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print('[detect] end of stream / no frame, stopping.')
            break

        # Run inference. results is a list (one per frame); we only pass one frame.
        results = model(frame, conf=args.conf, verbose=False)

        # .plot() draws boxes/labels/confidences onto a copy of the frame
        annotated = results[0].plot()

        # Simple FPS counter so you can talk about real-time performance
        frame_count += 1
        now = time.time()
        if now - prev_time >= 1.0:
            fps = frame_count / (now - prev_time)
            print(f'[detect] ~{fps:.1f} FPS')
            prev_time = now
            frame_count = 0

        # Print what was detected this frame (useful when there's no display,
        # e.g. running inside WSL2 without X forwarding)
        names = results[0].names
        detections = results[0].boxes
        if detections is not None and len(detections) > 0:
            labels = [names[int(c)] for c in detections.cls]
            print(f'[detect] found: {labels}')

        if args.save:
            cv2.imwrite('last_frame_annotated.jpg', annotated)
        else:
            cv2.imshow('YOLOv8 Detection', annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # If source is a single image, process once and exit
        if isinstance(source, str) and source.lower().endswith(('.jpg', '.jpeg', '.png')):
            cv2.imwrite('annotated_output.jpg', annotated)
            print('[detect] saved annotated_output.jpg')
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
