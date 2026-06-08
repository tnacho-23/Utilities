import anthropic
import base64
import os
import shutil

INPUT_DIR = "workbench/images/DCP/bodega_door_mayo_1-18"
OUTPUT_OPEN = "workbench/images/DCP/bodega_door_mayo_1-18/door_open"
OUTPUT_CLOSED = "workbench/images/DCP/bodega_door_mayo_1-18/door_closed"

PROMPT = (
    "This is a cropped security camera image of a warehouse rolling door. "
    "Is the door open (rolled up, not visible or barely visible, dark interior) "
    "or closed (corrugated metal panels visible across the image)? "
    "Reply with exactly one word: 'open' or 'closed'."
)


def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def classify(client: anthropic.Anthropic, image_path: str) -> str:
    data = encode_image(image_path)
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=5,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": data},
                },
                {"type": "text", "text": PROMPT},
            ],
        }],
    )
    return response.content[0].text.strip().lower()


def main():
    client = anthropic.Anthropic()

    os.makedirs(OUTPUT_OPEN, exist_ok=True)
    os.makedirs(OUTPUT_CLOSED, exist_ok=True)

    images = sorted(
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png")) and os.path.isfile(os.path.join(INPUT_DIR, f))
    )
    print(f"Found {len(images)} images")

    open_count = 0
    closed_count = 0

    for i, fname in enumerate(images, 1):
        src = os.path.join(INPUT_DIR, fname)
        label = classify(client, src)

        if "open" in label:
            shutil.copy2(src, os.path.join(OUTPUT_OPEN, fname))
            open_count += 1
        else:
            shutil.copy2(src, os.path.join(OUTPUT_CLOSED, fname))
            closed_count += 1

        print(f"  [{i:03d}/{len(images)}] [{label.upper():6}] {fname}")

    print(f"\nDone.  open={open_count}  closed={closed_count}")


if __name__ == "__main__":
    main()
