# Villova Cognitive Assistant Pipeline

A real-time cognitive state analysis pipeline using Intel RealSense D415 camera, combining gaze tracking, posture analysis, object detection, and contextual logging for AI companions.  

Designed primarily for monitoring cognitive and consciousness states in vulnerable populations (e.g. Alzheimer’s patients), the system provides insights into attention, posture and interactions with the environment to aid caregivers or autonomous assistants.

## Features

### Core Pipeline
- **Gaze Tracking**: MediaPipe Holistic for facial landmark detection and gaze vector estimation
- **Posture Analysis**: 3D posture analysis using pose landmarks and depth data
- **Object Detection**: YOLOv8 integration for real-time object recognition
- **Gaze-Object Intersection**: Determines when user gaze intersects with detected objects
- **Temporal Buffering**: 5-second sliding window for cognitive state analysis

### Database Logging
- **Observations Table**: Logs all detected objects with 3D positions and confidence scores
- **Gaze Events Table**: Records when user gaze focuses on specific objects
- **Context Summary**: Provides AI companions with user activity patterns

## Installation

```bash
pip install pyrealsense2 numpy opencv-python mediapipe ultralytics
```

## Usage

### Executable build

A standalone binary can be generated with [PyInstaller](https://www.pyinstaller.org/):

```bash
pip install pyinstaller
./build_executable.sh   # see script below
```

*Note:* PyInstaller requires a Python runtime built with a shared library (`--enable-shared`). On some build containers (including this dev environment) the binary creation may fail with an error about the missing shared library – this is an environment issue rather than a problem with the source code.

The resulting `dist/villa_cog_assist` (or `.exe` on Windows) can be run directly without a Python interpreter.

Alternatively, install the package with a console entry point:

```bash
pip install .
villa-cog-assist  # runs the demo
```


## Usage

### Basic Demo
```bash
python3 demo.py
```

This will start the camera pipeline and display:
- Gaze detection status
- Posture data (shoulder and torso Z-distances)
- Detected objects with confidence scores
- Gaze-object intersections

### Database Query
```bash
python3 query_database.py
```

Shows recent observations and provides context summaries for AI integration.

### Component Testing
```bash
python3 test_components.py
```

Tests individual components without requiring camera hardware.

## Architecture

### Data Flow
1. **Camera Capture**: RealSense D415 captures synchronized color and depth frames
2. **Feature Extraction**:
   - MediaPipe processes RGB frames for gaze and posture
   - YOLOv8 detects objects in color frames
   - Depth data converts 2D detections to 3D positions
3. **Intersection Analysis**: Checks if gaze vectors intersect with object bounding boxes
4. **Logging**: All observations stored in SQLite database for persistence

### Database Schema

#### observations table
```sql
CREATE TABLE observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL,
    event_type TEXT,
    data TEXT
);
```

#### gaze_events table
```sql
CREATE TABLE gaze_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL,
    object_class TEXT,
    confidence REAL,
    distance REAL,
    duration REAL
);
```

## MCP Integration

The database serves as the foundation for Model Context Protocol (MCP) integration:

1. **Sensory Data**: Camera observations continuously logged to SQLite
2. **MCP Server**: Background service exposes database via tool APIs
3. **AI Client**: Chat interfaces access real-time context through MCP tools

Example system prompt for AI companion:
```
You are an AI companion. You have access to my real-time visual environment via the get_current_vision_context tool and my historical behavior via the get_user_profile tool.
```

## Minimum Hardware Requirements

To ensure reliable operation the following minimum specifications are recommended:

* **Camera**: Intel RealSense D415 (USB 3.0) – required for depth and color
* **CPU**: Quad‑core Intel i5 or equivalent (8‑thread) at 2.5 GHz or higher
* **Memory**: 8 GB RAM (16 GB preferred when running additional services)
* **Storage**: 1 GB free disk space for logs and database
* **GPU**: Optional NVIDIA GPU (CUDA‑capable) for accelerated YOLOv8 inference; CPU‑only mode supported with reduced frame rate
* **OS**: Linux (Ubuntu 20.04+/Debian) or Windows 10/11 with Python 3.8+

## Production Optimization

For deployment in vehicles or production environments:

- **Frame Rate**: Cap at 10-15 FPS to prevent overheating
- **Model Selection**: Use YOLOv8n (nano) for CPU efficiency
- **Background Processing**: Run pipeline as background service
- **Data Retention**: Implement log rotation for long-term usage

## API Reference

### RealSenseCognitivePipeline

#### Constructor
```python
pipeline = RealSenseCognitivePipeline(width=1280, height=720, fps=15, yolo_model='yolov8n.pt')
```

#### Methods
- `start()`: Initialize camera and start processing
- `stop()`: Stop camera pipeline
- `process_frame()`: Process single frame and return FrameData
- `log_observation(event_type, data)`: Log observation to database
- `log_gaze_event(object_class, confidence, distance, duration)`: Log gaze event

### Data Structures

#### FrameData
```python
@dataclass
class FrameData:
    timestamp: float
    gaze: Optional[GazeData]
    posture: Optional[PostureData]
    detected_objects: List[DetectedObject]
    gaze_intersections: List[GazeObjectIntersection]
    color_image: np.ndarray
    depth_image: np.ndarray
```

#### DetectedObject
```python
@dataclass
class DetectedObject:
    class_name: str
    confidence: float
    bbox_2d: Tuple[int, int, int, int]
    center_3d: np.ndarray
    bbox_3d: np.ndarray
```

## Testing Setup

### Bench Test
1. Place test objects (can of beans, oatmeal box) on desk
2. Run `python3 demo.py`
3. Verify object detection and gaze intersection logging
4. Check database with `python3 query_database.py`

### Performance Metrics
- **CPU Usage**: Monitor with `top` during operation
- **Frame Rate**: Verify consistent 10-15 FPS
- **Detection Accuracy**: Test with known objects at various distances

## Future Enhancements

- **Improved Gaze Estimation**: Replace simplified gaze with proper eye model fitting
- **3D Bounding Box Intersection**: More accurate intersection testing with full 3D boxes
- **Object Tracking**: Temporal consistency across frames
- **Semantic Context**: Higher-level understanding of user activities
- **MCP Server Implementation**: Complete MCP server for AI integration</content>
<parameter name="filePath">/workspaces/Villa_Cog_Assist/README.md