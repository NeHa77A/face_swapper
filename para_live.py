# # import cv2
# # import numpy as np
# # import insightface
# # from insightface.app import FaceAnalysis
# # from gfpgan import GFPGANer
# # from concurrent.futures import ThreadPoolExecutor
# # import queue
# # import threading
# # from collections import deque
# # import time

# # class FaceSwapProcessor:
# #     def __init__(self, source_image_paths):
# #         self.frame_queue = queue.Queue(maxsize=3)
# #         self.result_queue = queue.Queue()
# #         self.running = False
# #         self.source_images = {}
# #         self.source_faces = {}
# #         self.selected_face_key = None
        
# #         # Initialize face analysis
# #         self.app = FaceAnalysis(name='buffalo_l')
# #         self.app.prepare(ctx_id=0, det_size=(320, 320))
        
# #         # Load the face swapping model
# #         self.swapper = insightface.model_zoo.get_model('inswapper_128_fp16.onnx',
# #                                                       download=True,
# #                                                       download_zip=True)
        
# #         # Load source images and extract faces
# #         for key, path in source_image_paths.items():
# #             try:
# #                 self.source_images[key] = cv2.imread(path)
# #                 if self.source_images[key] is None:
# #                     print(f"Failed to load image: {path}")
# #                     continue
# #                 faces = self.app.get(self.source_images[key])
# #                 if len(faces) > 0:
# #                     self.source_faces[key] = faces[0]
# #                     print(f"Successfully loaded face {key}")
# #                 else:
# #                     print(f"No face detected in image: {path}")
# #             except Exception as e:
# #                 print(f"Error loading image {path}: {e}")

# #         # Frame processing stats
# #         self.fps_deque = deque(maxlen=30)
# #         self.last_fps_time = time.time()
        
# #     def process_frame(self, frame):
# #         """Process a single frame in the worker thread"""
# #         if self.selected_face_key not in self.source_faces:
# #             return frame

# #         # Detect faces in current frame
# #         target_faces = self.app.get(frame)
# #         if not target_faces:
# #             return frame

# #         # Process each detected face
# #         result = frame.copy()
# #         for face in target_faces:
# #             try:
# #                 result = self.swapper.get(result, face, 
# #                                         self.source_faces[self.selected_face_key],
# #                                         paste_back=True)
# #             except Exception as e:
# #                 print(f"Face swap error: {e}")
# #                 continue
                
# #         return result

# #     def frame_processor_worker(self):
# #         """Worker thread for processing frames"""
# #         while self.running:
# #             try:
# #                 frame = self.frame_queue.get(timeout=0.1)
# #                 processed_frame = self.process_frame(frame)
# #                 self.result_queue.put(processed_frame)
# #                 self.frame_queue.task_done()
# #             except queue.Empty:
# #                 continue
# #             except Exception as e:
# #                 print(f"Worker error: {e}")
# #                 continue

# #     def start(self):
# #         """Start the processing threads"""
# #         self.running = True
# #         self.workers = []
# #         for _ in range(2):  # Number of worker threads
# #             worker = threading.Thread(target=self.frame_processor_worker)
# #             worker.daemon = True
# #             worker.start()
# #             self.workers.append(worker)

# #     def stop(self):
# #         """Stop all processing threads"""
# #         self.running = False
# #         for worker in self.workers:
# #             worker.join()

# #     def update_fps(self):
# #         """Calculate and return current FPS"""
# #         current_time = time.time()
# #         self.fps_deque.append(current_time - self.last_fps_time)
# #         self.last_fps_time = current_time
# #         if len(self.fps_deque) > 0:
# #             return 1.0 / (sum(self.fps_deque) / len(self.fps_deque))
# #         return 0

# # def main():
# #     # Initialize with source image paths
# #     source_image_paths = {
# #         "1": "Saket_Srivastava_sir.png",
# #         "2": "ajay.jpg"
# #     }
    
# #     processor = FaceSwapProcessor(source_image_paths)
# #     if not processor.source_faces:
# #         print("No valid source faces found. Please check your image paths and try again.")
# #         return
        
# #     processor.start()

# #     # Initialize webcam with optimized settings
# #     cap = cv2.VideoCapture(0)
# #     if not cap.isOpened():
# #         print("Failed to open webcam")
# #         processor.stop()
# #         return
        
# #     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# #     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# #     cap.set(cv2.CAP_PROP_FPS, 30)
    
# #     print("Press number keys (1/2) to select source face. Press 'q' to quit.")
    
# #     display_frame = None  # Initialize display_frame

# #     try:
# #         while True:
# #             ret, frame = cap.read()
# #             if not ret:
# #                 print("Failed to read frame from webcam")
# #                 break

# #             # Skip frame if queue is full
# #             if not processor.frame_queue.full():
# #                 processor.frame_queue.put(frame)

# #             # Get the most recent processed frame
# #             try:
# #                 while not processor.result_queue.empty():
# #                     display_frame = processor.result_queue.get_nowait()
# #             except queue.Empty:
# #                 pass

# #             # If we have a frame to display
# #             if display_frame is not None:
# #                 # Add FPS counter
# #                 fps = processor.update_fps()
# #                 fps_frame = display_frame.copy()
# #                 cv2.putText(fps_frame, f"FPS: {fps:.1f}", (10, 30),
# #                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
# #                 cv2.imshow("Live Face Swap", fps_frame)
# #             else:
# #                 # Show original frame if no processed frame is available
# #                 cv2.imshow("Live Face Swap", frame)

# #             # Handle key presses
# #             key = cv2.waitKey(1) & 0xFF
# #             if key == ord('q'):
# #                 break
# #             elif chr(key) in processor.source_faces:
# #                 processor.selected_face_key = chr(key)
# #                 print(f"Selected face: {processor.selected_face_key}")

# #     except Exception as e:
# #         print(f"Error in main loop: {e}")

# #     finally:
# #         processor.stop()
# #         cap.release()
# #         cv2.destroyAllWindows()

# # if __name__ == '__main__':
# #     main()
# import cv2
# import numpy as np
# import insightface
# from insightface.app import FaceAnalysis
# import threading
# import queue
# from collections import deque
# import time
# import os

# class FaceSwapProcessor:
#     def __init__(self, source_image_paths, display_width=480, display_height=360):
#         self.frame_queue = queue.Queue(maxsize=1)
#         self.result_queue = queue.Queue(maxsize=1)
#         self.running = False
#         self.source_faces = {}
#         self.selected_face_key = None
#         self.display_size = (display_width, display_height)
#         self.last_processed_frame = None
#         self.last_detected_face = None
#         self.swap_active = False  # New flag to control face swapping
        
#         # Initialize face analysis
#         self.app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
#         self.app.prepare(ctx_id=-1, det_size=(640, 640))
        
#         model_path = 'inswapper_128_fp16.onnx'
#         if not os.path.exists(model_path):
#             print("Please ensure inswapper_128.onnx is in the current directory")
#             return

#         try:
#             self.swapper = insightface.model_zoo.get_model(model_path, providers=['CPUExecutionProvider'])
#         except Exception as e:
#             print(f"Error loading model: {str(e)}")
#             return
        
#         # Load source faces
#         for key, path in source_image_paths.items():
#             try:
#                 if not os.path.exists(path):
#                     print(f"Image not found: {path}")
#                     continue
                    
#                 img = cv2.imread(path)
#                 if img is None:
#                     print(f"Could not read image: {path}")
#                     continue
                
#                 faces = self.app.get(img)
#                 if len(faces) > 0:
#                     self.source_faces[key] = faces[0]
#                     print(f"Successfully loaded face {key}")
#                 else:
#                     print(f"No face detected in {path}")
#             except Exception as e:
#                 print(f"Error loading {path}: {str(e)}")

#         print(f"Loaded {len(self.source_faces)} faces")
        
#         self.fps_deque = deque(maxlen=10)
#         self.last_fps_time = time.time()
        
#     def process_frame(self, frame):
#         # Only process if swapping is active
#         if not self.swap_active:
#             return None  # Return None when no swapping should occur

#         try:
#             # Detect faces in current frame
#             target_faces = self.app.get(frame)
            
#             if target_faces:
#                 self.last_detected_face = target_faces[0]
#             elif self.last_detected_face is not None:
#                 target_faces = [self.last_detected_face]
#             else:
#                 return None

#             # Perform face swap
#             result = self.swapper.get(frame, 
#                                     target_faces[0], 
#                                     self.source_faces[self.selected_face_key], 
#                                     paste_back=True)
            
#             self.last_processed_frame = result
#             return cv2.resize(result, self.display_size)
            
#         except Exception as e:
#             print(f"Error in face swap: {str(e)}")
#             if self.last_processed_frame is not None:
#                 return cv2.resize(self.last_processed_frame, self.display_size)
#             return None

#     def frame_processor_worker(self):
#         last_process_time = 0
#         min_process_interval = 1.0 / 30

#         while self.running:
#             try:
#                 frame = self.frame_queue.get(timeout=0.1)
                
#                 current_time = time.time()
#                 if current_time - last_process_time < min_process_interval:
#                     continue
                
#                 processed = self.process_frame(frame)
#                 last_process_time = current_time
                
#                 # Only update result if processing occurred
#                 if processed is not None:
#                     while not self.result_queue.empty():
#                         try:
#                             self.result_queue.get_nowait()
#                         except queue.Empty:
#                             break
#                     self.result_queue.put(processed)
                
#             except queue.Empty:
#                 continue
#             except Exception as e:
#                 print(f"Worker error: {str(e)}")
#                 continue

#     def start(self):
#         self.running = True
#         self.worker = threading.Thread(target=self.frame_processor_worker)
#         self.worker.daemon = True
#         self.worker.start()

#     def stop(self):
#         self.running = False
#         if hasattr(self, 'worker'):
#             self.worker.join()

#     def update_fps(self):
#         current_time = time.time()
#         fps = 1.0 / (current_time - self.last_fps_time)
#         self.fps_deque.append(fps)
#         self.last_fps_time = current_time
#         return sum(self.fps_deque) / len(self.fps_deque)

# def main():
#     DISPLAY_WIDTH = 480
#     DISPLAY_HEIGHT = 360
    
#     source_image_paths = {
#         "1": "Saket_Srivastava_sir.png",
#         "2": "ajay.jpg"
#     }
    
#     processor = FaceSwapProcessor(source_image_paths, DISPLAY_WIDTH, DISPLAY_HEIGHT)
#     if not processor.source_faces:
#         print("No source faces found! Please check your image paths and image content.")
#         return
        
#     processor.start()

#     cap = cv2.VideoCapture(0)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#     cap.set(cv2.CAP_PROP_FPS, 30)
#     cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
#     cv2.namedWindow("Live Face Swap", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("Live Face Swap", DISPLAY_WIDTH, DISPLAY_HEIGHT)
    
#     print("\nPress 1/2 to start face swap, 'esc' to stop swap, 'q' to quit")
    
#     try:
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             # Update frame queue
#             if processor.frame_queue.empty():
#                 processor.frame_queue.put(frame)

#             # Display logic
#             if processor.swap_active:
#                 try:
#                     display_frame = processor.result_queue.get_nowait()
#                     if display_frame is None:
#                         continue
#                 except queue.Empty:
#                     continue
#             else:
#                 display_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

#             # Add FPS counter
#             fps = processor.update_fps()
#             cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 20),
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
#             cv2.imshow("Live Face Swap", display_frame)

#             key = cv2.waitKey(1) & 0xFF
#             if key == ord('q'):
#                 break
#             elif key == 27:  # ESC key
#                 processor.swap_active = False
#                 processor.selected_face_key = None
#                 print("Face swap stopped")
#             elif chr(key) in processor.source_faces:
#                 processor.selected_face_key = chr(key)
#                 processor.swap_active = True
#                 print(f"Selected face: {processor.selected_face_key}")

#     finally:
#         processor.stop()
#         cap.release()
#         cv2.destroyAllWindows()

# if __name__ == '__main__':
#     main()
import cv2
import queue
import time
import threading
import numpy as np
import os
from insightface.app import FaceAnalysis
from insightface.model_zoo import model_zoo

class FaceSwapProcessor:
    def __init__(self, source_image_paths, display_width, display_height):
        self.source_image_paths = source_image_paths
        self.display_size = (display_width, display_height)
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
        self.running = False
        self.swap_active = False
        self.selected_face_key = None
        self.last_detected_face = None
        self.last_processed_frame = None
        self.source_faces = {}
        self.fps = 0

        # Initialize face analysis and loading the swap model
        self.app = FaceAnalysis(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

        model_path = "inswapper_128_fp16.onnx"
        if not os.path.exists(model_path):
            print("Please ensure the model file 'inswapper_128.onnx' is in the current directory.")
            return

        try:
            self.swapper = model_zoo.get_model(model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            return

        # Load source faces
        for key, image_path in self.source_image_paths.items():
            if os.path.exists(image_path):
                image = cv2.imread(image_path)
                faces = self.app.get(image)
                if faces:
                    self.source_faces[key] = faces[0]
                    print(f"Successfully loaded face {key}")
                else:
                    print(f"Failed to detect face in image: {image_path}")
            else:
                print(f"Image path does not exist: {image_path}")

    def start(self):
        if not self.running:
            self.running = True
            self.worker_thread = threading.Thread(target=self.frame_processor_worker, daemon=True)
            self.worker_thread.start()

    def stop(self):
        self.running = False
        if self.worker_thread.is_alive():
            self.worker_thread.join()

    def process_frame(self, frame):
        if not self.swap_active or self.selected_face_key is None:
            print("Face swapping is inactive or no face selected.")
            return None

        try:
            # Detect faces in the current frame
            target_faces = self.app.get(frame)

            if target_faces:
                self.last_detected_face = target_faces[0]
            elif self.last_detected_face is not None:
                target_faces = [self.last_detected_face]
            else:
                print("No face detected.")
                return None

            # Perform face swap
            print("Performing face swap...")
            result = self.swapper.get(frame, target_faces[0], self.source_faces[self.selected_face_key], paste_back=True)
            self.last_processed_frame = result
            print("Face swap successful.")
            return cv2.resize(result, self.display_size)

        except Exception as e:
            print(f"Error during face swap: {str(e)}")
            if self.last_processed_frame is not None:
                return cv2.resize(self.last_processed_frame, self.display_size)
            return None

    def frame_processor_worker(self):
        last_process_time = 0
        min_process_interval = 1.0 / 30  # Max processing rate: 30 FPS

        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.1)

                # Process the frame if enough time has passed
                current_time = time.time()
                if current_time - last_process_time < min_process_interval:
                    continue

                processed_frame = self.process_frame(frame)
                last_process_time = current_time

                # Only update result if processing occurred
                if processed_frame is not None:
                    while not self.result_queue.empty():
                        try:
                            self.result_queue.get_nowait()
                        except queue.Empty:
                            break
                    self.result_queue.put(processed_frame)
                    print("Processed frame added to result queue.")

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Worker error: {str(e)}")
                continue

    def update_fps(self):
        current_time = time.time()
        if not hasattr(self, "_last_frame_time"):  # Initialize for the first frame
            self._last_frame_time = current_time
            self.fps = 0.0  # Set initial FPS to 0
        else:
            self.fps = 1 / (current_time - self._last_frame_time)
        self._last_frame_time = current_time
        return self.fps

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

    cv2.namedWindow("Live Face Swap", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Live Face Swap", DISPLAY_WIDTH, DISPLAY_HEIGHT)

    print("\nPress 1/2 to start face swap, 'esc' to stop swap, 'q' to quit")

    try:
        last_displayed_frame = None  # Cache last processed frame

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to read from camera.")
                break

            # Add the current frame to the processing queue if empty
            if processor.frame_queue.empty():
                processor.frame_queue.put(frame)

            # Fetch the processed frame or fallback to the last frame
            if processor.swap_active:
                try:
                    display_frame = processor.result_queue.get_nowait()
                    last_displayed_frame = display_frame  # Cache successfully swapped frame
                    print("Displaying swapped frame...")
                except queue.Empty:
                    if last_displayed_frame is not None:
                        display_frame = last_displayed_frame
                        print("Using last processed frame.")
                    else:
                        display_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
                        print("Result queue empty. Showing original frame.")
            else:
                # If swapping is inactive, show the original frame
                display_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
                last_displayed_frame = None  # Clear the cache when not swapping

            # Add FPS counter to the display
            fps = processor.update_fps()
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Show the frame
            cv2.imshow("Live Face Swap", display_frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # Quit
                break
            elif key == 27:  # ESC key to stop swapping
                processor.swap_active = False
                processor.selected_face_key = None
                print("Face swap stopped.")
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