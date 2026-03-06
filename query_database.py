#!/usr/bin/env python3
"""
Database Query Script for Cognitive Observations.

This script demonstrates how to query the SQLite database for logged observations
and gaze events, similar to how an MCP server would expose this data.
"""

import sqlite3
import json
from datetime import datetime
import time


def query_recent_observations(hours: int = 1):
    """Query recent observations from the database."""
    conn = sqlite3.connect('cognitive_observations.db')
    cursor = conn.cursor()

    # Get observations from the last N hours
    since_time = time.time() - (hours * 3600)

    cursor.execute(
        'SELECT timestamp, event_type, data FROM observations WHERE timestamp > ? ORDER BY timestamp DESC',
        (since_time,)
    )

    observations = cursor.fetchall()
    conn.close()

    return observations


def query_gaze_events(hours: int = 1):
    """Query recent gaze events from the database."""
    conn = sqlite3.connect('cognitive_observations.db')
    cursor = conn.cursor()

    # Get gaze events from the last N hours
    since_time = time.time() - (hours * 3600)

    cursor.execute(
        'SELECT timestamp, object_class, confidence, distance, duration FROM gaze_events WHERE timestamp > ? ORDER BY timestamp DESC',
        (since_time,)
    )

    events = cursor.fetchall()
    conn.close()

    return events


def get_user_context_summary():
    """Generate a summary of user context from recent observations."""
    observations = query_recent_observations(hours=1)
    gaze_events = query_gaze_events(hours=1)

    summary = {
        'time_range': 'last hour',
        'total_observations': len(observations),
        'total_gaze_events': len(gaze_events),
        'detected_objects': {},
        'gaze_focus': {}
    }

    # Count object detections
    for obs in observations:
        timestamp, event_type, data = obs
        if event_type == 'object_detected':
            data_dict = json.loads(data)
            obj_class = data_dict['class']
            if obj_class not in summary['detected_objects']:
                summary['detected_objects'][obj_class] = 0
            summary['detected_objects'][obj_class] += 1

    # Count gaze events
    for event in gaze_events:
        timestamp, obj_class, confidence, distance, duration = event
        if obj_class not in summary['gaze_focus']:
            summary['gaze_focus'][obj_class] = 0
        summary['gaze_focus'][obj_class] += 1

    return summary


def init_database():
    """Initialize the database schema if it doesn't exist."""
    conn = sqlite3.connect('cognitive_observations.db')
    cursor = conn.cursor()

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

    conn.commit()
    conn.close()
    print("Database initialized.")


def main():
    """Main function to demonstrate database querying."""
    print("Cognitive Observations Database Query")
    print("=" * 40)

    # Initialize database
    init_database()

    # Get recent observations
    print("\nRecent Observations (last hour):")
    observations = query_recent_observations(hours=1)
    for obs in observations[:10]:  # Show last 10
        timestamp, event_type, data = obs
        time_str = datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')
        print(f"[{time_str}] {event_type}: {data}")

    # Get gaze events
    print("\nRecent Gaze Events (last hour):")
    gaze_events = query_gaze_events(hours=1)
    for event in gaze_events[:10]:  # Show last 10
        timestamp, obj_class, confidence, distance, duration = event
        time_str = datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')
        print(".2f")

    # Get context summary
    print("\nUser Context Summary (last hour):")
    summary = get_user_context_summary()
    print(f"Total observations: {summary['total_observations']}")
    print(f"Total gaze events: {summary['total_gaze_events']}")

    if summary['detected_objects']:
        print("Detected objects:")
        for obj_class, count in summary['detected_objects'].items():
            print(f"  {obj_class}: {count} times")

    if summary['gaze_focus']:
        print("Gaze focus:")
        for obj_class, count in summary['gaze_focus'].items():
            print(f"  {obj_class}: {count} times")

    print("\nThis data can be exposed via MCP for AI companions to understand user context.")


if __name__ == "__main__":
    main()