import cv2
import numpy as np
from threading import Thread
import tensorflow as tf
from collections import deque
import time
import os
import logging

class CombinedTracker:
    def __init__(self, model_path, labels_path, buffer_size=64, min_area=500, threshold=0.5, resolution=(640, 480)):
        """
        Initialize the combined TFLite detection and optical flow tracker
        """
        self.buffer_size = buffer_size
        self.min_area = min_area
        self.threshold = threshold
        self.resolution = resolution
        
        # Initialize TFLite
        self.setup_tflite(model_path, labels_path)
        
        # Initialize optical flow parameters
        self.track_points = None
        self.prev_gray = None
        self.tracks = []
        self.track_len = 10
        self.detect_interval = 5
        self.frame_idx = 0
        
        # Parameters for Shi-Tomasi corner detection
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )
        
        # Parameters for Lucas-Kanade optical flow
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Motion history
        self.motion_history = deque(maxlen=buffer_size)
        
        # Video stream
        self.videostream = None
        self.stopped = False
        self.frame = None
        
    def setup_tflite(self, model_path, labels_path):
        """Setup TFLite model and labels"""
        # Load labels
        with open(labels_path, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]
            if self.labels[0] == '???':
                del(self.labels[0])
        
        # Initialize TFLite interpreter
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get model details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]
        
        # Check if model is floating point
        self.floating_model = (self.input_details[0]['dtype'] == np.float32)
        
    def start_video_stream(self):
        """Initialize and start the video stream thread"""
        self.videostream = cv2.VideoCapture(0)
        ret = self.videostream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.videostream.set(3, self.resolution[0])
        ret = self.videostream.set(4, self.resolution[1])
        
        Thread(target=self.update, args=()).start()
        return self
    
    def update(self):
        """Update function for the video stream thread"""
        while True:
            if self.stopped:
                self.videostream.release()
                return
            
            _, self.frame = self.videostream.read()
    
    def detect_objects(self, frame):
        """Detect objects using TFLite model"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (self.width, self.height))
        input_data = np.expand_dims(frame_resized, axis=0)
        
        if self.floating_model:
            input_data = (np.float32(input_data) - 127.5) / 127.5
        
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        # Get detection results
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]
        
        detections = []
        for i in range(len(scores)):
            if scores[i] > self.threshold:
                ymin = int(max(1, (boxes[i][0] * self.resolution[1])))
                xmin = int(max(1, (boxes[i][1] * self.resolution[0])))
                ymax = int(min(self.resolution[1], (boxes[i][2] * self.resolution[1])))
                xmax = int(min(self.resolution[0], (boxes[i][3] * self.resolution[0])))
                
                detections.append({
                    'box': [xmin, ymin, xmax, ymax],
                    'class': self.labels[int(classes[i])],
                    'score': float(scores[i])
                })
        
        return detections
    
    def calculate_optical_flow(self, prev_gray, gray, points):
        """Calculate optical flow with sub-pixel precision"""
        if points is None or len(points) == 0:
            return None, None
        
        points = np.float32(points).reshape(-1, 1, 2)
        new_points, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, points, None, **self.lk_params
        )
        
        if new_points is None:
            return None, None
        
        # Filter points within frame boundaries
        good_new = []
        good_old = []
        for i, (new, old) in enumerate(zip(new_points, points)):
            a, b = new.ravel()
            c, d = old.ravel()
            if (0 <= a < self.resolution[0] and 0 <= b < self.resolution[1] and
                0 <= c < self.resolution[0] and 0 <= d < self.resolution[1] and
                status[i]):
                good_new.append([a, b])
                good_old.append([c, d])
        
        return np.array(good_new), np.array(good_old)
    
    def process_frame(self, frame):
        """Process a single frame with both TFLite detection and optical flow"""
        if frame is None:
            return frame, [], []
        
        # Convert frame to grayscale for optical flow
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Initialize or update tracking points
        if self.track_points is None or len(self.track_points) < 10:
            self.track_points = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
            self.prev_gray = gray.copy()
            
        # Calculate optical flow
        if self.prev_gray is not None and self.track_points is not None:
            good_new, good_old = self.calculate_optical_flow(self.prev_gray, gray, self.track_points)
            
            # Draw optical flow tracks
            if good_new is not None and good_old is not None:
                for new, old in zip(good_new, good_old):
                    a, b = new.astype(int)
                    c, d = old.astype(int)
                    frame = cv2.line(frame, (a, b), (c, d), (0, 255, 0), 2)
                    frame = cv2.circle(frame, (a, b), 5, (0, 0, 255), -1)
                
                self.track_points = good_new.reshape(-1, 1, 2)
        
        # Perform object detection at specified intervals
        detections = []
        if self.frame_idx % self.detect_interval == 0:
            detections = self.detect_objects(frame)
            
            # Draw detection boxes
            for det in detections:
                xmin, ymin, xmax, ymax = det['box']
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
                
                label = f"{det['class']}: {int(det['score']*100)}%"
                cv2.putText(frame, label, (xmin, ymin-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 255, 0), 2)
        
        # Update frame index and previous frame
        self.frame_idx += 1
        self.prev_gray = gray.copy()
        
        return frame, self.track_points, detections
    
    def stop(self):
        """Stop the video stream thread"""
        self.stopped = True

def main():
    # Initialize tracker
    tracker = CombinedTracker(
        model_path='detect.tflite',
        labels_path='labelmap.txt',
        threshold=0.5
    )
    
    # Start video stream
    tracker.start_video_stream()
    time.sleep(1)  # Allow camera to warm up
    
    try:
        while True:
            frame = tracker.frame
            if frame is not None:
                # Process frame
                processed_frame, tracks, detections = tracker.process_frame(frame)
                
                # Display FPS
                cv2.putText(processed_frame, f'Objects: {len(detections)}',
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                # Display frame
                cv2.imshow('Combined Tracker', processed_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        tracker.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
