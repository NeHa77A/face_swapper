import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import threading
import queue
from collections import deque
import time
import os

class FaceSwapProcessor:
    def __init__(self, source_image_paths, display_width=480, display_height=360):
        self.frame_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)
        self.running = False
        self.source_faces = {}
        self.selected_face_key = None
        self.display_size = (display_width, display_height)
        self.last_processed_frame = None
        self.last_detected_face = None
        self.swap_active = False  # New flag to control face swapping
        
        # Initialize face analysis
        self.app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=-1, det_size=(640, 640))
        
        model_path = 'inswapper_128_fp16.onnx'
        if not os.path.exists(model_path):
            print("Please ensure inswapper_128.onnx is in the current directory")
            return

        try:
            self.swapper = insightface.model_zoo.get_model(model_path, providers=['CPUExecutionProvider'])
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return
        
        # Load source faces
        for key, path in source_image_paths.items():
            try:
                if not os.path.exists(path):
                    print(f"Image not found: {path}")
                    continue
                    
                img = cv2.imread(path)
                if img is None:
                    print(f"Could not read image: {path}")
                    continue
                
                faces = self.app.get(img)
                if len(faces) > 0:
                    self.source_faces[key] = faces[0]
                    print(f"Successfully loaded face {key}")
                else:
                    print(f"No face detected in {path}")
            except Exception as e:
                print(f"Error loading {path}: {str(e)}")

        print(f"Loaded {len(self.source_faces)} faces")
        
        self.fps_deque = deque(maxlen=10)
        self.last_fps_time = time.time()
        
    def process_frame(self, frame):
        # Only process if swapping is active
        if not self.swap_active:
            return None  # Return None when no swapping should occur

        try:
            # Detect faces in current frame
            target_faces = self.app.get(frame)
            
            if target_faces:
                self.last_detected_face = target_faces[0]
            elif self.last_detected_face is not None:
                target_faces = [self.last_detected_face]
            else:
                return None

            # Perform face swap
            result = self.swapper.get(frame, 
                                    target_faces[0], 
                                    self.source_faces[self.selected_face_key], 
                                    paste_back=True)
            
            self.last_processed_frame = result
            return cv2.resize(result, self.display_size)
            
        except Exception as e:
            print(f"Error in face swap: {str(e)}")
            if self.last_processed_frame is not None:
                return cv2.resize(self.last_processed_frame, self.display_size)
            return None

    def frame_processor_worker(self):
        last_process_time = 0
        min_process_interval = 1.0 / 30

        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
                
                current_time = time.time()
                if current_time - last_process_time < min_process_interval:
                    continue
                
                processed = self.process_frame(frame)
                last_process_time = current_time
                
                # Only update result if processing occurred
                if processed is not None:
                    while not self.result_queue.empty():
                        try:
                            self.result_queue.get_nowait()
                        except queue.Empty:
                            break
                    self.result_queue.put(processed)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Worker error: {str(e)}")
                continue

    def start(self):
        self.running = True
        self.worker = threading.Thread(target=self.frame_processor_worker)
        self.worker.daemon = True
        self.worker.start()

    def stop(self):
        self.running = False
        if hasattr(self, 'worker'):
            self.worker.join()

    def update_fps(self):
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_fps_time)
        self.fps_deque.append(fps)
        self.last_fps_time = current_time
        return sum(self.fps_deque) / len(self.fps_deque)

def main():
    DISPLAY_WIDTH = 480
    DISPLAY_HEIGHT = 360
    
    source_image_paths = {
        "1": "Saket_Srivastava_sir.png",
        "2": "ajay.jpg"
    }
    
    processor = FaceSwapProcessor(source_image_paths, DISPLAY_WIDTH, DISPLAY_HEIGHT)
    if not processor.source_faces:
        print("No source faces found! Please check your image paths and image content.")
        return
        
    processor.start()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    cv2.namedWindow("Live Face Swap", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Live Face Swap", DISPLAY_WIDTH, DISPLAY_HEIGHT)
    
    print("\nPress 1/2 to start face swap, 'esc' to stop swap, 'q' to quit")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Update frame queue
            if processor.frame_queue.empty():
                processor.frame_queue.put(frame)

            # Display logic
            if processor.swap_active:
                try:
                    display_frame = processor.result_queue.get_nowait()
                    if display_frame is None:
                        continue
                except queue.Empty:
                    continue
            else:
                display_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

            # Add FPS counter
            fps = processor.update_fps()
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow("Live Face Swap", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == 27:  # ESC key
                processor.swap_active = False
                processor.selected_face_key = None
                print("Face swap stopped")
            elif chr(key) in processor.source_faces:
                processor.selected_face_key = chr(key)
                processor.swap_active = True
                print(f"Selected face: {processor.selected_face_key}")

    finally:
        processor.stop()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()