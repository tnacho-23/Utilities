import argparse
import cv2
import os
import shutil
import numpy as np

DEFAULT_INPUT_DIR = "workbench/images/input"
DEFAULT_OUTPUT_DIR = "workbench/images/output_filtered"
SIMILARITY_THRESHOLD = 10.0  # mean absolute pixel diff (0-255); below this → too similar
COMPARE_SIZE = (320, 240)     # resize for diff comparison (speed)
EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def mean_diff(img_a, img_b):
    a = cv2.resize(img_a, COMPARE_SIZE)
    b = cv2.resize(img_b, COMPARE_SIZE)
    a_gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    b_gray = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    return np.mean(cv2.absdiff(a_gray, b_gray))


def filter_folder(input_dir, output_dir, threshold=SIMILARITY_THRESHOLD):
    images = sorted(
        f for f in os.listdir(input_dir) if f.lower().endswith(EXTENSIONS)
    )
    if not images:
        print("No images found.")
        return

    os.makedirs(output_dir, exist_ok=True)

    kept = 0
    skipped = 0
    ref_img = None

    for fname in images:
        path = os.path.join(input_dir, fname)
        img = cv2.imread(path)
        if img is None:
            print(f"  [WARN] Could not read: {fname}")
            continue

        if ref_img is None:
            shutil.copy2(path, os.path.join(output_dir, fname))
            ref_img = img
            kept += 1
            print(f"  [KEEP] {fname}  (first image)")
            continue

        diff = mean_diff(ref_img, img)
        if diff < threshold:
            skipped += 1
            print(f"  [SKIP] {fname}  diff={diff:.2f}")
        else:
            shutil.copy2(path, os.path.join(output_dir, fname))
            ref_img = img
            kept += 1
            print(f"  [KEEP] {fname}  diff={diff:.2f}")

    print(f"\nDone. Kept {kept}, skipped {skipped} out of {kept + skipped} images.")
    print(f"Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Filter similar consecutive images.")
    parser.add_argument("--input", default=DEFAULT_INPUT_DIR, help="Input folder")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR, help="Output folder")
    parser.add_argument("--threshold", type=float, default=SIMILARITY_THRESHOLD, help="Mean abs diff threshold")
    args = parser.parse_args()

    filter_folder(args.input, args.output, args.threshold)


if __name__ == "__main__":
    main()
