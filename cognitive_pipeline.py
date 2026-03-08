import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp
from collections import deque
from typing import List, Dict, Tuple, Optional
import time
from dataclasses import dataclass
from ultralytics import YOLO
import sqlite3
import json


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
class DetectedObject:
    """Data structure for detected objects."""
    class_name: str
    confidence: float
    bbox_2d: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center_3d: np.ndarray  # 3D position in camera coordinates (x, y, z)
    bbox_3d: np.ndarray  # 3D bounding box corners


@dataclass
class GazeObjectIntersection:
    """Data structure for gaze-object intersections."""
    object_class: str
    intersection_point: np.ndarray  # 3D intersection point
    distance_to_object: float
    gaze_vector: np.ndarray


@dataclass
class FrameData:
    """Combined data for a single frame."""
    timestamp: float
    gaze: Optional[GazeData]
    posture: Optional[PostureData]
    detected_objects: List[DetectedObject]
    gaze_intersections: List[GazeObjectIntersection]
    color_image: np.ndarray
    depth_image: np.ndarray


class RealSenseCognitivePipeline:
    """
    A modular pipeline for RealSense D415 camera to extract cognitive state features.

    Handles camera initialization, depth-color alignment, MediaPipe Holistic integration,
    feature extraction (gaze vectors, posture Z-data), and temporal buffering.
    """

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30, yolo_model: str = 'yolov8n.pt'):
        """
        Initialize the RealSense pipeline with specified resolution and FPS.

        Args:
            width: Frame width (default 1280)
            height: Frame height (default 720)
            fps: Frames per second (30 or 60, default 30)
            yolo_model: YOLOv8 model to use (default 'yolov8n.pt' for nano model)
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

        # YOLO object detection
        self.yolo = YOLO(yolo_model)

        # SQLite database for logging observations
        self.db_conn = sqlite3.connect('cognitive_observations.db')
        self._init_database()

        # Temporal buffer: sliding window of 5 seconds
        self.buffer_duration = 5.0  # seconds
        self.frame_buffer: deque[FrameData] = deque()

        # Camera intrinsics (set after start)
        self.color_intrinsics: Optional[rs.intrinsics] = None
        self.depth_intrinsics: Optional[rs.intrinsics] = None

        self.is_running = False

    def _init_database(self):
        """Initialize the SQLite database schema."""
        cursor = self.db_conn.cursor()

        # Create observations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                event_type TEXT,
                data TEXT
            )
        ''')

        # Create gaze_events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS gaze_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                object_class TEXT,
                confidence REAL,
                distance REAL,
                duration REAL
            )
        ''')

        self.db_conn.commit()

    def log_observation(self, event_type: str, data: Dict):
        """Log an observation to the database."""
        cursor = self.db_conn.cursor()
        cursor.execute(
            'INSERT INTO observations (timestamp, event_type, data) VALUES (?, ?, ?)',
            (time.time(), event_type, json.dumps(data))
        )
        self.db_conn.commit()

    def log_gaze_event(self, object_class: str, confidence: float, distance: float, duration: float = 0.0):
        """Log a gaze event to the database."""
        cursor = self.db_conn.cursor()
        cursor.execute(
            'INSERT INTO gaze_events (timestamp, object_class, confidence, distance, duration) VALUES (?, ?, ?, ?, ?)',
            (time.time(), object_class, confidence, distance, duration)
        )
        self.db_conn.commit()

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

    def _detect_objects(self, color_image: np.ndarray, depth_frame: rs.depth_frame) -> List[DetectedObject]:
        """
        Detect objects in the color image using YOLOv8 and calculate their 3D positions.

        Args:
            color_image: RGB color image
            depth_frame: Aligned depth frame

        Returns:
            List of DetectedObject instances
        """
        detected_objects = []

        # Run YOLO detection
        results = self.yolo(color_image, conf=0.5)  # Confidence threshold

        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                class_name = self.yolo.names[class_id]

                # Calculate center point of bounding box
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # Get depth at center
                depth = self._get_depth_at_pixel(depth_frame, center_x, center_y)

                if depth > 0:  # Valid depth
                    # Convert 2D pixel coordinates to 3D camera coordinates
                    center_3d = self._pixel_to_3d(center_x, center_y, depth)

                    # Create simple 3D bounding box (approximate)
                    # Get depth at corners for more accurate bbox
                    corners_2d = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                    corners_3d = []
                    for cx, cy in corners_2d:
                        c_depth = self._get_depth_at_pixel(depth_frame, cx, cy)
                        if c_depth > 0:
                            corners_3d.append(self._pixel_to_3d(cx, cy, c_depth))
                        else:
                            corners_3d.append(center_3d)  # Fallback to center

                    bbox_3d = np.array(corners_3d)

                    detected_object = DetectedObject(
                        class_name=class_name,
                        confidence=float(confidence),
                        bbox_2d=(x1, y1, x2, y2),
                        center_3d=center_3d,
                        bbox_3d=bbox_3d
                    )
                    detected_objects.append(detected_object)

        return detected_objects

    def _pixel_to_3d(self, x: int, y: int, depth: float) -> np.ndarray:
        """
        Convert 2D pixel coordinates and depth to 3D camera coordinates.

        Args:
            x: Pixel x coordinate
            y: Pixel y coordinate
            depth: Depth in meters

        Returns:
            3D point in camera coordinates (x, y, z)
        """
        if not self.color_intrinsics:
            return np.array([0, 0, depth])

        # Use RealSense intrinsics to deproject pixel to 3D
        point_3d = rs.rs2_deproject_pixel_to_point(self.color_intrinsics, [x, y], depth)
        return np.array(point_3d)

    def _check_gaze_object_intersection(self, gaze_data: GazeData, detected_objects: List[DetectedObject]) -> List[GazeObjectIntersection]:
        """
        Check if gaze vectors intersect with detected object bounding boxes.

        Args:
            gaze_data: Current gaze data
            detected_objects: List of detected objects

        Returns:
            List of intersections found
        """
        intersections = []

        # Use average of left and right eye gaze for simplicity
        gaze_vector = (gaze_data.left_eye_gaze + gaze_data.right_eye_gaze) / 2.0
        gaze_vector = gaze_vector / np.linalg.norm(gaze_vector)  # Normalize

        # Camera origin (assuming gaze is relative to camera)
        camera_origin = np.array([0, 0, 0])

        for obj in detected_objects:
            # Simple intersection check: check if gaze ray passes near object center
            # In a full implementation, you'd check intersection with the 3D bounding box

            # Vector from camera to object center
            to_object = obj.center_3d - camera_origin
            distance_to_object = np.linalg.norm(to_object)

            # Normalize direction to object
            obj_direction = to_object / distance_to_object

            # Angle between gaze and object direction
            cos_angle = np.dot(gaze_vector, obj_direction)

            # If angle is small (cos_angle close to 1), consider it an intersection
            if cos_angle > 0.95:  # About 18 degrees tolerance
                intersection_point = camera_origin + gaze_vector * distance_to_object

                intersection = GazeObjectIntersection(
                    object_class=obj.class_name,
                    intersection_point=intersection_point,
                    distance_to_object=distance_to_object,
                    gaze_vector=gaze_vector
                )
                intersections.append(intersection)

        return intersections

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

            # Detect objects
            detected_objects = self._detect_objects(color_image, depth_frame)

            # Check gaze-object intersections
            gaze_intersections = []
            if gaze_data and detected_objects:
                gaze_intersections = self._check_gaze_object_intersection(gaze_data, detected_objects)

                # Log gaze events to database
                for intersection in gaze_intersections:
                    self.log_gaze_event(
                        intersection.object_class,
                        0.0,  # We don't have object confidence here, could be improved
                        intersection.distance_to_object
                    )

            # Log observations
            if detected_objects:
                for obj in detected_objects:
                    self.log_observation('object_detected', {
                        'class': obj.class_name,
                        'confidence': obj.confidence,
                        'position_3d': obj.center_3d.tolist(),
                        'bbox_2d': list(obj.bbox_2d)
                    })

            # Create frame data
            frame_data = FrameData(
                timestamp=time.time(),
                gaze=gaze_data,
                posture=posture_data,
                detected_objects=detected_objects,
                gaze_intersections=gaze_intersections,
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
        """Cleanup: stop the pipeline if still running and close database."""
        self.stop()
        if hasattr(self, 'db_conn'):
            self.db_conn.close()