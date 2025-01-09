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