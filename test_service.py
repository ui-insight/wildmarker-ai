"""Test script for the wildmarker detection + classification service."""

import json
import sys
import os
import requests

BASE_URL = os.environ.get("SERVICE_URL", "http://localhost:8000")
TEST_IMAGES_DIR = os.path.join(os.path.dirname(__file__), "test_images")


def print_json(data):
    print(json.dumps(data, indent=2))


def test_health():
    print("=== Health Check ===")
    r = requests.get(f"{BASE_URL}/health")
    r.raise_for_status()
    print_json(r.json())
    print()


def test_single_image(image_path: str):
    print(f"=== Single Image: {os.path.basename(image_path)} ===")
    with open(image_path, "rb") as f:
        r = requests.post(
            f"{BASE_URL}/predict",
            files=[("files", (os.path.basename(image_path), f, "image/jpeg"))],
        )
    r.raise_for_status()
    print_json(r.json())
    print()


def test_two_images(path1: str, path2: str):
    print(f"=== Two Images: {os.path.basename(path1)}, {os.path.basename(path2)} ===")
    with open(path1, "rb") as f1, open(path2, "rb") as f2:
        r = requests.post(
            f"{BASE_URL}/predict",
            files=[
                ("files", (os.path.basename(path1), f1, "image/jpeg")),
                ("files", (os.path.basename(path2), f2, "image/jpeg")),
            ],
        )
    r.raise_for_status()
    print_json(r.json())
    print()


def test_invalid_file():
    print("=== Invalid File (should return error in result) ===")
    r = requests.post(
        f"{BASE_URL}/predict",
        files=[("files", ("bad.txt", b"this is not an image", "text/plain"))],
    )
    r.raise_for_status()
    data = r.json()
    print_json(data)
    assert data["results"][0]["error"] is not None, "Expected error for invalid file"
    print("PASS: Error correctly reported for invalid file\n")


def test_no_files():
    print("=== No Files (should return 422) ===")
    r = requests.post(f"{BASE_URL}/predict")
    print(f"Status: {r.status_code}")
    assert r.status_code == 422, f"Expected 422, got {r.status_code}"
    print("PASS: 422 returned for missing files\n")


def main():
    # Find test images
    images = sorted([
        os.path.join(TEST_IMAGES_DIR, f)
        for f in os.listdir(TEST_IMAGES_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]) if os.path.isdir(TEST_IMAGES_DIR) else []

    test_health()

    if len(images) >= 1:
        test_single_image(images[0])
    if len(images) >= 2:
        test_two_images(images[0], images[1])

    # Also test with sample_200 images if available
    sample_dir = os.path.join(os.path.dirname(__file__), "..", "sample_200")
    if os.path.isdir(sample_dir):
        sample_images = sorted([
            os.path.join(sample_dir, d, f)
            for d in os.listdir(sample_dir)
            if os.path.isdir(os.path.join(sample_dir, d))
            for f in os.listdir(os.path.join(sample_dir, d))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])[:2]
        if len(sample_images) >= 1:
            print("=== Using sample_200 images ===")
            test_single_image(sample_images[0])
        if len(sample_images) >= 2:
            test_two_images(sample_images[0], sample_images[1])

    test_invalid_file()
    test_no_files()

    print("All tests passed!")


if __name__ == "__main__":
    main()
