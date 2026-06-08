import cv2
import json
import os
import numpy as np

INPUT_DIR = "workbench/images/DCP/timelapse_sample_mayo_1-18"
VERTICES_JSON = "workbench/jsons/DCP/door_vertices.json"
OUTPUT_DIR = "workbench/images/DCP/timelapse_sample_mayo_1-18_cropped"
PADDING = 0

FOLDER_TO_CAMERA = {
    "Acceso Bodega 1": "192_168_4_116_4",
    "Linea de Produccion": "192_168_4_100_3",
    "Porton Bodega J": "192_168_4_123_2",
}

CAMERA_TO_NAME = {
    "192_168_4_116_4": "bodega1",
    "192_168_4_100_3": "lineas_produccion",
    "192_168_4_123_2": "bodega_j",
}

# Resolution at which the vertices were originally annotated (from cam_capture JSONs)
REFERENCE_RESOLUTION = {
    "192_168_4_100_3": (1920, 1080),
    "192_168_4_116_4": (2688, 1520),
    "192_168_4_123_2": (2560, 1440),
}


def crop_zone_padded(frame, vertices, padding):
    pts = np.array(vertices, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(pts)
    h_img, w_img = frame.shape[:2]
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(w_img, x + w + padding)
    y2 = min(h_img, y + h + padding)
    return frame[y1:y2, x1:x2]


def main():
    with open(VERTICES_JSON) as f:
        vertices_data = json.load(f)

    cameras = {entry["camera_name"]: entry["zones"] for entry in vertices_data}

    for folder_name, camera_name in FOLDER_TO_CAMERA.items():
        folder_path = os.path.join(INPUT_DIR, folder_name)
        if not os.path.isdir(folder_path):
            print(f"[SKIP] Folder not found: {folder_path}")
            continue

        zones = cameras.get(camera_name)
        if not zones:
            print(f"[SKIP] No zones for camera: {camera_name}")
            continue

        images = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        print(f"[{folder_name}] {len(images)} images, {len(zones)} zones")

        folder_prefix = CAMERA_TO_NAME.get(camera_name, camera_name)

        for zone in zones:
            out_folder = os.path.join(OUTPUT_DIR, f"{folder_prefix}_{zone['name']}")
            os.makedirs(out_folder, exist_ok=True)

        for img_name in images:
            img_path = os.path.join(folder_path, img_name)
            frame = cv2.imread(img_path)
            if frame is None:
                print(f"  [WARN] Could not read: {img_path}")
                continue

            ref_w, ref_h = REFERENCE_RESOLUTION[camera_name]
            frame = cv2.resize(frame, (ref_w, ref_h), interpolation=cv2.INTER_LINEAR)

            for zone in zones:
                cropped = crop_zone_padded(frame, zone["vertices"], PADDING)
                out_folder = os.path.join(OUTPUT_DIR, f"{folder_prefix}_{zone['name']}")
                out_path = os.path.join(out_folder, img_name)
                cv2.imwrite(out_path, cropped)

            print(f"  Cropped: {img_name}")

    print("Done.")


if __name__ == "__main__":
    main()
