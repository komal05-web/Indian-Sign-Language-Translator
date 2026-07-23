import os
import shutil
import random

# Your original massive dataset
SOURCE_DIR = r"C:\\Users\\kkani\\Downloads\\isl_word"

# The new mini dataset it will create
DEST_DIR = r"C:\\Users\\kkani\\Downloads\\isl_word_mini"

# Increased to 50 — enough for decent generalisation, still fast to train
IMAGES_PER_CLASS = 500

# Reproducible sampling — same 500 images every run
RANDOM_SEED = 42


def create_mini_dataset():
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Could not find {SOURCE_DIR}")
        return

    # Wipe old mini dataset so stale images don't mix with new ones
    if os.path.exists(DEST_DIR):
        shutil.rmtree(DEST_DIR)
        print(f"Removed old mini dataset at: {DEST_DIR}")

    os.makedirs(DEST_DIR, exist_ok=True)
    random.seed(RANDOM_SEED)

    # Only process single-letter A-Z folders (ignores junk folders)
    classes = sorted([
        d for d in os.listdir(SOURCE_DIR)
        if os.path.isdir(os.path.join(SOURCE_DIR, d))
        and len(d) == 1 and d.isalpha()
    ])

    if not classes:
        print("No A-Z class folders found. Check SOURCE_DIR.")
        return

    total_copied = 0

    for class_name in classes:
        src_class_path  = os.path.join(SOURCE_DIR, class_name)
        dest_class_path = os.path.join(DEST_DIR, class_name)
        os.makedirs(dest_class_path, exist_ok=True)

        # Only pick valid image files
        all_images = [
            f for f in os.listdir(src_class_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]

        if len(all_images) == 0:
            print(f"  WARNING: No images found for class '{class_name}' — skipping.")
            continue

        # Random sample (not just first N) — better variety for training
        n = min(IMAGES_PER_CLASS, len(all_images))
        selected = random.sample(all_images, n)

        for img in selected:
            shutil.copy2(
                os.path.join(src_class_path, img),
                os.path.join(dest_class_path, img)
            )

        total_copied += n
        status = "" if n == IMAGES_PER_CLASS else f" (only {n} available)"
        print(f"  [{class_name}]  {n} images copied{status}")

    print(f"\nDone!  {total_copied} images across {len(classes)} classes.")
    print(f"Mini dataset saved at: {DEST_DIR}")


if __name__ == "__main__":
    create_mini_dataset()
