#!/usr/bin/env python3
"""
Demo script for RealSense Cognitive States Pipeline.

This script demonstrates how to use the RealSenseCognitivePipeline class
to capture and process frames from a RealSense D415 camera for cognitive state analysis.
"""

import sys
import time
from cognitive_pipeline import RealSenseCognitivePipeline


def main():
    """Main demo function."""
    print("RealSense Cognitive States Demo")
    print("===============================")

    # Initialize the pipeline
    pipeline = RealSenseCognitivePipeline(width=1280, height=720, fps=30)

    try:
        print("Starting camera pipeline...")
        pipeline.start()
        print("Camera started successfully!")

        print("Processing frames... (Press Ctrl+C to stop)")

        frame_count = 0
        start_time = time.time()

        while True:
            # Process a frame
            frame_data = pipeline.process_frame()

            if frame_data:
                frame_count += 1

                # Print basic info every 30 frames
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed

                    buffer_size = len(pipeline.get_buffered_data())

                    print(".1f"
                          f"Buffer: {buffer_size} frames")

                    # Print feature availability
                    gaze_status = "✓" if frame_data.gaze else "✗"
                    posture_status = "✓" if frame_data.posture else "✗"
                    print(f"  Gaze: {gaze_status} | Posture: {posture_status}")

                    if frame_data.posture:
                        print(".2f"
                              ".2f")

            # Small delay to prevent overwhelming output
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping demo...")

    except Exception as e:
        print(f"Error during demo: {e}")
        return 1

    finally:
        pipeline.stop()
        print("Camera pipeline stopped.")

    return 0


if __name__ == "__main__":
    sys.exit(main())