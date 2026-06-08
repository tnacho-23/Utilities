import os
import random
import shutil
from pathlib import Path

SOURCE_DIR = Path(__file__).parent.parent / "workbench/images/FUP_Mall/FUP_P4"
OUTPUT_DIR = Path(__file__).parent.parent / "workbench/images/FUP_Mall/FUP_P4_sampled"
TOTAL_SAMPLES = 3000
SEED = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def get_images(folder: Path) -> list[Path]:
    """Recursively collect all images under *folder*."""
    return [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]

def safe_dest(img: Path, output_dir: Path) -> Path:
    """Return a collision-free destination path inside output_dir."""
    dest = output_dir / img.name
    if not dest.exists():
        return dest
    # Append parent folder name(s) until unique
    stem, suffix = img.stem, img.suffix
    for parent in img.parents:
        dest = output_dir / f"{parent.name}_{stem}{suffix}"
        if not dest.exists():
            return dest
    # Fallback: numeric suffix
    i = 1
    while True:
        dest = output_dir / f"{stem}_{i}{suffix}"
        if not dest.exists():
            return dest
        i += 1

def main():
    folders = sorted(f for f in SOURCE_DIR.iterdir() if f.is_dir())

    if not folders:
        print(f"No subfolders found in {SOURCE_DIR}")
        return

    per_folder = TOTAL_SAMPLES // len(folders)
    remainder = TOTAL_SAMPLES % len(folders)

    print(f"Found {len(folders)} top-level folders, sampling {per_folder} images each "
          f"(+{remainder} extra for first folders) = {TOTAL_SAMPLES} total")

    random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- First pass: collect images and cap each folder at its fair share ---
    folder_images = [(f, get_images(f)) for f in folders]
    quotas = [per_folder + (1 if i < remainder else 0) for i in range(len(folders))]

    # Redistribute budget from capped folders to uncapped ones
    redistributed = True
    while redistributed:
        redistributed = False
        uncapped = [(i, imgs) for i, (_, imgs) in enumerate(folder_images) if len(imgs) >= quotas[i]]
        for i, (_, imgs) in enumerate(folder_images):
            if len(imgs) < quotas[i]:
                leftover = quotas[i] - len(imgs)
                quotas[i] = len(imgs)           # cap at what's available
                # Spread leftover evenly across uncapped folders
                if uncapped:
                    extra_each = leftover // len(uncapped)
                    extra_rem  = leftover % len(uncapped)
                    for j, (ui, _) in enumerate(uncapped):
                        quotas[ui] += extra_each + (1 if j < extra_rem else 0)
                    redistributed = True        # re-check in case new caps were hit

    # --- Second pass: sample and copy ---
    total_copied = 0
    for (folder, images), n in zip(folder_images, quotas):
        sampled = random.sample(images, n)
        for img in sampled:
            shutil.copy2(img, safe_dest(img, OUTPUT_DIR))
        print(f"  {folder.name}: {len(images)} available -> {n} sampled")
        total_copied += n

    print(f"\nDone. {total_copied} images saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
