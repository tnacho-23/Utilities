
import os
import argparse
import shutil
from ultralytics import YOLO


def auto_label(input_path, output_path, model_path, conf_threshold=0.25):
    """
    Loads a YOLO *classification* model, runs inference on a folder of images,
    and saves results as:
      - <output_path>/labels/<stem>.txt  →  "<class_index> <confidence>"
      - <output_path>/<class_name>/<img>  →  image copied to its predicted class folder

    Falls back gracefully when confidence is below threshold (saves as "uncertain").
    """
    model = YOLO(model_path)

    labels_dir = os.path.join(output_path, "labels")
    os.makedirs(labels_dir, exist_ok=True)
    print(f"Created output directory: {output_path}")

    img_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    images = [f for f in os.listdir(input_path) if f.lower().endswith(img_extensions)]

    if not images:
        print(f"No images found in: {input_path}")
        return

    print(f"Found {len(images)} images. Starting classification...")

    # Resolve class names from the model
    class_names = model.names  # dict {0: 'classA', 1: 'classB', ...}

    for img_name in images:
        img_path = os.path.join(input_path, img_name)
        results = model(img_path, verbose=False)
        result = results[0]

        if result.probs is None:
            print(f"  [WARN] No classification output for: {img_name} — skipping.")
            continue

        top1_idx  = int(result.probs.top1)
        top1_conf = float(result.probs.top1conf)
        class_name = class_names.get(top1_idx, str(top1_idx))

        # Apply confidence threshold
        if top1_conf < conf_threshold:
            class_name = "uncertain"
            top1_idx   = -1

        # Save label file: "<class_index> <confidence>"
        stem       = os.path.splitext(img_name)[0]
        label_path = os.path.join(labels_dir, stem + ".txt")
        with open(label_path, "w") as f:
            f.write(f"{top1_idx} {top1_conf:.4f}\n")

        # Copy image to per-class subfolder for easy review
        class_dir = os.path.join(output_path, class_name)
        os.makedirs(class_dir, exist_ok=True)
        shutil.copy2(img_path, os.path.join(class_dir, img_name))

        print(f"  {img_name}  →  {class_name} ({top1_conf:.2%})")

    print(f"\nClassification complete.")
    print(f"  Labels : {labels_dir}")
    print(f"  Images : {output_path}/<class_name>/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-label images using a YOLO classifier.")
    parser.add_argument("--input",   type=str, required=False,
                        default=r"C:/Users/ignac/Escritorio/DualVision/Utilities/Yolo/images/fup_patente_easton_p2",
                        help="Folder containing images to classify.")
    parser.add_argument("--output",  type=str, required=False,
                        default=r"C:/Users/ignac/Escritorio/DualVision/Utilities/Yolo/labels/fup_patente_easton_p2",
                        help="Folder to save labels and sorted images.")
    parser.add_argument("--weights", type=str, required=False,
                        default=r"C:/Users/ignac/Escritorio/DualVision/Utilities/Yolo/weights/FUP_mall_y26mV1.pt",
                        help="Path to YOLO classification .pt weights.")
    parser.add_argument("--conf",    type=float, default=0.25,
                        help="Confidence threshold (default: 0.25).")

    args = parser.parse_args()
    auto_label(args.input, args.output, args.weights, args.conf)