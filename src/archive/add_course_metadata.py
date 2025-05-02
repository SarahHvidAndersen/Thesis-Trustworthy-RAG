#!/usr/bin/env python3
import os
import json

# Root folder where each subfolder is a course
ROOT = "processed_syllabi"

for course in os.listdir(ROOT):
    course_dir = os.path.join(ROOT, course)
    if not os.path.isdir(course_dir):
        continue

    for fname in ("processed_chunks.json", "processed_chunks_with_embeddings.json"):
        path = os.path.join(course_dir, fname)
        if not os.path.exists(path):
            continue

        print(f"Updating {path}…")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Inject the course name into each chunk's metadata
        for chunk in data:
            chunk["course"] = course

        # Write back in-place
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

print(" Done injecting course metadata into all chunks.")
