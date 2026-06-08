import cv2
import os
import argparse
from ultralytics import YOLO


DEFAULT_WEIGHTS = "yolo11n.pt"
DEFAULT_CLASS_ID = 0  # person in COCO
DEFAULT_CONF = 0.4
DEFAULT_FRAME_STEP = 1  # process every N frames


def crop_detections(
    video_path: str,
    output_dir: str,
    weights: str = DEFAULT_WEIGHTS,
    class_id: int = DEFAULT_CLASS_ID,
    conf: float = DEFAULT_CONF,
    frame_step: int = DEFAULT_FRAME_STEP,
):
    os.makedirs(output_dir, exist_ok=True)

    model = YOLO(weights)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return

    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[{video_stem}] {total_frames} frames @ {fps:.1f} fps")

    frame_idx = 0
    total_crops = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step != 0:
            frame_idx += 1
            continue

        results = model(frame, verbose=False, conf=conf, classes=[class_id])
        boxes = results[0].boxes

        for det_idx, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            h_img, w_img = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x2), min(h_img, y2)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            out_name = f"{video_stem}_f{frame_idx:06d}_d{det_idx}.jpg"
            cv2.imwrite(os.path.join(output_dir, out_name), crop)
            total_crops += 1

        if frame_idx % 500 == 0:
            print(f"  frame {frame_idx}/{total_frames} — {total_crops} crops so far")

        frame_idx += 1

    cap.release()
    print(f"Done. {total_crops} crops saved to: {output_dir}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop YOLO detections from video.")
    parser.add_argument("video_path", nargs="+", help="Video file(s) to process")
    parser.add_argument(
        "--output_dir", default=None,
        help="Output folder (default: <video_dir>/crops_<video_stem>)"
    )
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--class_id", type=int, default=DEFAULT_CLASS_ID,
                        help=f"COCO class to crop (default: {DEFAULT_CLASS_ID} = person)")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--frame_step", type=int, default=DEFAULT_FRAME_STEP,
                        help="Process every N frames (default: 1 = all frames)")
    args = parser.parse_args()

    for video_path in args.video_path:
        video_stem = os.path.splitext(os.path.basename(video_path))[0]
        video_dir = os.path.dirname(os.path.abspath(video_path))
        out = args.output_dir or os.path.join(video_dir, f"crops_{video_stem}")
        crop_detections(
            video_path=video_path,
            output_dir=out,
            weights=args.weights,
            class_id=args.class_id,
            conf=args.conf,
            frame_step=args.frame_step,
        )
