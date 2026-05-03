"""
startup.py — downloads model files from Google Drive at Render build time.

Steps:
  1. Upload your 4 model files to Google Drive
  2. Share each as "Anyone with the link"
  3. Copy the file ID from the share URL
  4. Set these as Environment Variables on Render dashboard:
       WORD_MODEL_ID, LABEL_MAP_ID, SENT_MODEL_ID, SENT_LABEL_ID
"""

import os
import urllib.request

FILES = {
    "word/isl_best_model.keras":         os.getenv("WORD_MODEL_ID", ""),
    "word/label_map.json":               os.getenv("LABEL_MAP_ID",  ""),
    "sentence/isl_sentence_model.keras": os.getenv("SENT_MODEL_ID", ""),
    "sentence/sentence_label_map.json":  os.getenv("SENT_LABEL_ID", ""),
}


def download(file_id, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    url = f"https://drive.google.com/drive/folders/1-BFw0WUFJBvxHGwPpiZeLuVCgQAm7Pu4?usp=drive_link"
    print(f"[Startup] Downloading {dest} …")
    try:
        urllib.request.urlretrieve(url, dest)
        mb = os.path.getsize(dest) / 1024 / 1024
        print(f"[Startup] Done  ({mb:.1f} MB)")
    except Exception as e:
        print(f"[Startup] FAILED: {e}")


def main():
    for dest, fid in FILES.items():
        if not fid:
            print(f"[Startup] Skipping {dest} — env var not set")
            continue
        if os.path.exists(dest):
            print(f"[Startup] Already exists: {dest}")
            continue
        download(fid, dest)


if __name__ == "__main__":
    main()
