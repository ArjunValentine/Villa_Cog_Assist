#!/usr/bin/env python3
"""
Test script for object detection and database functionality.

This script tests the YOLO object detection and database logging
without requiring a RealSense camera.
"""

import numpy as np
import cv2
from ultralytics import YOLO
import sqlite3
import json
import time
from cognitive_pipeline import DetectedObject, GazeObjectIntersection


def test_object_detection():
    """Test YOLO object detection on a sample image."""
    print("Testing YOLO Object Detection...")

    # Create a simple test image (black background with some shapes)
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)

    # Draw some simple shapes to simulate objects
    cv2.rectangle(test_image, (100, 100), (200, 200), (255, 0, 0), -1)  # Blue rectangle
    cv2.circle(test_image, (400, 300), 50, (0, 255, 0), -1)  # Green circle

    # Load YOLO model
    yolo = YOLO('yolov8n.pt')

    # Run detection
    results = yolo(test_image, conf=0.1)  # Lower confidence for testing

    print(f"Detection completed. Found {len(results[0].boxes)} potential detections")

    # Since our test image doesn't contain real objects, detections might be low confidence
    # But this tests that the pipeline works
    return True


def test_database_operations():
    """Test database logging functionality."""
    print("Testing Database Operations...")

    # Connect to database
    conn = sqlite3.connect('cognitive_observations.db')
    cursor = conn.cursor()

    # Test observation logging
    test_data = {
        'class': 'test_object',
        'confidence': 0.85,
        'position_3d': [0.5, 0.2, 1.0],
        'bbox_2d': [100, 100, 200, 200]
    }

    cursor.execute(
        'INSERT INTO observations (timestamp, event_type, data) VALUES (?, ?, ?)',
        (time.time(), 'object_detected', json.dumps(test_data))
    )

    # Test gaze event logging
    cursor.execute(
        'INSERT INTO gaze_events (timestamp, object_class, confidence, distance, duration) VALUES (?, ?, ?, ?, ?)',
        (time.time(), 'test_object', 0.85, 1.0, 2.5)
    )

    conn.commit()

    # Query back the data
    cursor.execute('SELECT COUNT(*) FROM observations')
    obs_count = cursor.fetchone()[0]

    cursor.execute('SELECT COUNT(*) FROM gaze_events')
    gaze_count = cursor.fetchone()[0]

    conn.close()

    print(f"Database test successful. Observations: {obs_count}, Gaze events: {gaze_count}")
    return True


def test_data_structures():
    """Test the data structures."""
    print("Testing Data Structures...")

    # Test DetectedObject
    obj = DetectedObject(
        class_name='bottle',
        confidence=0.92,
        bbox_2d=(50, 50, 150, 200),
        center_3d=np.array([0.3, 0.1, 0.8]),
        bbox_3d=np.array([[0.2, 0.0, 0.7], [0.4, 0.0, 0.7], [0.4, 0.2, 0.7], [0.2, 0.2, 0.7]])
    )

    # Test GazeObjectIntersection
    intersection = GazeObjectIntersection(
        object_class='bottle',
        intersection_point=np.array([0.3, 0.1, 0.8]),
        distance_to_object=0.8,
        gaze_vector=np.array([0.3, 0.1, 0.8]) / np.linalg.norm(np.array([0.3, 0.1, 0.8]))
    )

    print("Data structures created successfully")
    return True


def main():
    """Run all tests."""
    print("Cognitive Pipeline Component Tests")
    print("=" * 40)

    try:
        test_data_structures()
        test_database_operations()
        test_object_detection()

        print("\n" + "=" * 40)
        print("All tests passed! The cognitive pipeline components are working.")
        print("\nTo run the full pipeline with camera:")
        print("  python3 demo.py")
        print("\nTo query logged observations:")
        print("  python3 query_database.py")

    except Exception as e:
        print(f"Test failed with error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())