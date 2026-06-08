import cv2
import os
import argparse


TARGET_CLASS = 0


def crop_class(input_dir: str, output_dir: str, target_class: int = TARGET_CLASS):
    os.makedirs(output_dir, exist_ok=True)
    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}

    images = [
        f for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in image_exts
    ]

    total_crops = 0

    for img_name in sorted(images):
        base = os.path.splitext(img_name)[0]
        label_path = os.path.join(input_dir, base + ".txt")

        if not os.path.exists(label_path):
            continue

        img_path = os.path.join(input_dir, img_name)
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"[WARN] Could not read: {img_path}")
            continue

        h_img, w_img = frame.shape[:2]

        with open(label_path) as f:
            lines = [l.strip() for l in f if l.strip()]

        class_count = 0
        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            if cls != target_class:
                continue

            cx, cy, bw, bh = map(float, parts[1:])
            x1 = int((cx - bw / 2) * w_img)
            y1 = int((cy - bh / 2) * h_img)
            x2 = int((cx + bw / 2) * w_img)
            y2 = int((cy + bh / 2) * h_img)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x2), min(h_img, y2)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            out_name = f"{base}_cls{target_class}_{class_count}.jpg"
            cv2.imwrite(os.path.join(output_dir, out_name), crop)
            class_count += 1

        if class_count:
            print(f"  {img_name} -> {class_count} crops")
            total_crops += class_count

    print(f"\nDone. {total_crops} crops saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop YOLO-labeled class from images.")
    parser.add_argument("input_dir", help="Folder with images and .txt labels")
    parser.add_argument(
        "--output_dir", default=None,
        help="Output folder (default: <input_dir>_crops_cls<N>)"
    )
    parser.add_argument(
        "--class_id", type=int, default=TARGET_CLASS,
        help=f"Class ID to crop (default: {TARGET_CLASS})"
    )
    args = parser.parse_args()

    stripped = args.input_dir.rstrip("/\\")
    out = args.output_dir or f"{stripped}_crops_cls{args.class_id}"
    crop_class(args.input_dir, out, args.class_id)
