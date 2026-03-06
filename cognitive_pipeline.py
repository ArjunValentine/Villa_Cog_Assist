import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp
from collections import deque
from typing import List, Dict, Tuple, Optional
import time
from dataclasses import dataclass


@dataclass
class GazeData:
    """Data structure for gaze vectors relative to head pose."""
    left_eye_gaze: np.ndarray  # 3D vector
    right_eye_gaze: np.ndarray  # 3D vector
    head_pose: np.ndarray  # Rotation matrix or Euler angles


@dataclass
class PostureData:
    """Data structure for posture Z-distance data."""
    left_shoulder_z: float
    right_shoulder_z: float
    torso_center_z: float


@dataclass
class FrameData:
    """Combined data for a single frame."""
    timestamp: float
    gaze: Optional[GazeData]
    posture: Optional[PostureData]
    color_image: np.ndarray
    depth_image: np.ndarray


class RealSenseCognitivePipeline:
    """
    A modular pipeline for RealSense D415 camera to extract cognitive state features.

    Handles camera initialization, depth-color alignment, MediaPipe Holistic integration,
    feature extraction (gaze vectors, posture Z-data), and temporal buffering.
    """

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30):
        """
        Initialize the RealSense pipeline with specified resolution and FPS.

        Args:
            width: Frame width (default 1280)
            height: Frame height (default 720)
            fps: Frames per second (30 or 60, default 30)
        """
        self.width = width
        self.height = height
        self.fps = fps

        # RealSense pipeline and configuration
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        # Align depth to color
        self.align = rs.align(rs.stream.color)

        # MediaPipe Holistic
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            refine_face_landmarks=True
        )

        # Temporal buffer: sliding window of 5 seconds
        self.buffer_duration = 5.0  # seconds
        self.frame_buffer: deque[FrameData] = deque()

        # Camera intrinsics (set after start)
        self.color_intrinsics: Optional[rs.intrinsics] = None
        self.depth_intrinsics: Optional[rs.intrinsics] = None

        self.is_running = False

    def start(self) -> None:
        """Start the RealSense pipeline and initialize intrinsics."""
        self.pipeline.start(self.config)
        profile = self.pipeline.get_active_profile()

        # Get intrinsics
        color_stream = profile.get_stream(rs.stream.color)
        self.color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

        depth_stream = profile.get_stream(rs.stream.depth)
        self.depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()

        self.is_running = True

    def stop(self) -> None:
        """Stop the RealSense pipeline."""
        if self.is_running:
            self.pipeline.stop()
            self.is_running = False

    def _get_depth_at_pixel(self, depth_frame: rs.depth_frame, x: int, y: int) -> float:
        """
        Get depth value at a specific pixel coordinate.

        Args:
            depth_frame: The depth frame
            x: X coordinate
            y: Y coordinate

        Returns:
            Depth in meters
        """
        depth_value = depth_frame.get_distance(x, y)
        return depth_value

    def _extract_gaze_vectors(self, face_landmarks) -> Optional[GazeData]:
        """
        Extract gaze vectors relative to head pose from MediaPipe face landmarks.

        This is a simplified implementation. In practice, you'd need more sophisticated
        eye model fitting for accurate gaze estimation.

        Args:
            face_landmarks: MediaPipe face landmarks

        Returns:
            GazeData object or None if extraction fails
        """
        if not face_landmarks:
            return None

        # Simplified: Use eye corner landmarks for basic gaze direction
        # Left eye: landmarks 33 (left corner), 133 (right corner), 159 (top), 145 (bottom)
        # Right eye: 362 (left corner), 263 (right corner), 386 (top), 374 (bottom)

        # For head pose, use nose and eye positions
        nose_tip = face_landmarks.landmark[1]  # Nose tip
        left_eye_inner = face_landmarks.landmark[133]
        right_eye_inner = face_landmarks.landmark[362]

        # Simplified head pose: assume forward facing, gaze along camera Z-axis
        # In a full implementation, compute rotation matrix from facial landmarks
        head_pose = np.eye(3)  # Identity for now

        # Simplified gaze vectors (would need proper eye model)
        left_gaze = np.array([0, 0, 1])  # Forward
        right_gaze = np.array([0, 0, 1])

        return GazeData(left_gaze, right_gaze, head_pose)

    def _extract_posture_z(self, pose_landmarks, depth_frame: rs.depth_frame) -> Optional[PostureData]:
        """
        Extract Z-distance data for shoulders and torso from pose landmarks and depth map.

        Args:
            pose_landmarks: MediaPipe pose landmarks
            depth_frame: Aligned depth frame

        Returns:
            PostureData object or None if extraction fails
        """
        if not pose_landmarks:
            return None

        # Pose landmarks: 11=left_shoulder, 12=right_shoulder, 23=hip_left, 24=hip_right
        left_shoulder = pose_landmarks.landmark[11]
        right_shoulder = pose_landmarks.landmark[12]
        left_hip = pose_landmarks.landmark[23]
        right_hip = pose_landmarks.landmark[24]

        # Convert normalized coordinates to pixel coordinates
        img_h, img_w = self.height, self.width
        ls_x, ls_y = int(left_shoulder.x * img_w), int(left_shoulder.y * img_h)
        rs_x, rs_y = int(right_shoulder.x * img_w), int(right_shoulder.y * img_h)
        lh_x, lh_y = int(left_hip.x * img_w), int(left_hip.y * img_h)
        rh_x, rh_y = int(right_hip.x * img_w), int(right_hip.y * img_h)

        # Get depth values (in meters)
        ls_z = self._get_depth_at_pixel(depth_frame, ls_x, ls_y)
        rs_z = self._get_depth_at_pixel(depth_frame, rs_x, rs_y)
        lh_z = self._get_depth_at_pixel(depth_frame, lh_x, lh_y)
        rh_z = self._get_depth_at_pixel(depth_frame, rh_x, rh_y)

        # Torso center as average of shoulders and hips
        torso_z = (ls_z + rs_z + lh_z + rh_z) / 4.0

        return PostureData(ls_z, rs_z, torso_z)

    def process_frame(self) -> Optional[FrameData]:
        """
        Capture and process a single frame from the camera.

        Returns:
            FrameData object with extracted features, or None if processing fails
        """
        if not self.is_running:
            return None

        try:
            # Capture frames
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                return None

            # Convert to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Process with MediaPipe
            rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            results = self.holistic.process(rgb_image)

            # Extract features
            gaze_data = self._extract_gaze_vectors(results.face_landmarks) if results.face_landmarks else None
            posture_data = self._extract_posture_z(results.pose_landmarks, depth_frame) if results.pose_landmarks else None

            # Create frame data
            frame_data = FrameData(
                timestamp=time.time(),
                gaze=gaze_data,
                posture=posture_data,
                color_image=color_image,
                depth_image=depth_image
            )

            # Add to buffer
            self._update_buffer(frame_data)

            return frame_data

        except Exception as e:
            print(f"Error processing frame: {e}")
            return None

    def _update_buffer(self, frame_data: FrameData) -> None:
        """Update the temporal buffer with new frame data."""
        self.frame_buffer.append(frame_data)

        # Remove old frames outside the 5-second window
        current_time = time.time()
        while self.frame_buffer and (current_time - self.frame_buffer[0].timestamp) > self.buffer_duration:
            self.frame_buffer.popleft()

    def get_buffered_data(self) -> List[FrameData]:
        """
        Get the current buffered data within the sliding window.

        Returns:
            List of FrameData objects in the buffer
        """
        return list(self.frame_buffer)

    def __del__(self):
        """Cleanup: stop the pipeline if still running."""
        self.stop()