# # import cv2
# # import numpy as np
# # import insightface
# # from insightface.app import FaceAnalysis
# # from gfpgan import GFPGANer

# # def swap_faces(source_image, target_image):
# #     # Initialize the FaceAnalysis app for detection
# #     app = FaceAnalysis(name='buffalo_l')
# #     app.prepare(ctx_id=-1, det_size=(640, 640))  # Use GPU if available; ctx_id=-1 for CPU.

# #     # Load the face swapping model
# #     swapper = insightface.model_zoo.get_model('inswapper_128_fp16.onnx', download=True, download_zip=True)

# #     # Detect faces in the source image
# #     source_faces = app.get(source_image)
# #     if len(source_faces) == 0:
# #         raise ValueError("No faces detected in the source image!")

# #     # Detect faces in the target image
# #     target_faces = app.get(target_image)
# #     if len(target_faces) == 0:
# #         raise ValueError("No faces detected in the target image!")

# #     # Use the first detected face in the source image
# #     source_face = source_faces[0]

# #     # Perform face swapping for each face in the target image
# #     result = target_image.copy()
# #     for face in target_faces:
# #         result = swapper.get(result, face, source_face, paste_back=True)

# #     return result

# # def restore_image(image):
# #     # Initialize the GFPGAN model for restoration
# #     gfpgan = GFPGANer(
# #         model_path='GFPGANv1.4.pth',  # Model will be downloaded automatically
# #         upscale=2,  # Upscaling factor
# #         arch='clean',  # Use the clean architecture
# #         channel_multiplier=2,  # Channel multiplier for quality adjustment
# #         bg_upsampler=None  # Use default background upsampler
# #     )

# #     # Perform restoration
# #     _, _, restored_image = gfpgan.enhance(image, has_aligned=False, only_center_face=False, paste_back=True)
# #     return restored_image

# # if __name__ == '__main__':
# #     # Directly load images
# #     source_image_path = 'Saket_Srivastava_sir.png'  # Replace with your source image path
# #     target_image_path = 'khadus.jpg'  # Replace with your target image path

# #     # Read images
# #     source_image = cv2.imread(source_image_path)
# #     target_image = cv2.imread(target_image_path)

# #     if source_image is None:
# #         raise ValueError(f"Source image could not be loaded from {source_image_path}")
# #     if target_image is None:
# #         raise ValueError(f"Target image could not be loaded from {target_image_path}")

# #     # Swap faces and get the result
# #     swapped_image = swap_faces(source_image, target_image)

# #     # Save the swapped image
# #     swapped_image_path = "swapped_result1.jpg"
# #     cv2.imwrite(swapped_image_path, swapped_image)
# #     print(f"Face-swapped image saved as '{swapped_image_path}'")

# #     # Restore the swapped image
# #     restored_image = restore_image(swapped_image)

# #     # Save the restored image
# #     restored_image_path = "restored_result1.jpg"
# #     cv2.imwrite(restored_image_path, restored_image)
# #     print(f"Restored image saved as '{restored_image_path}'")

# import cv2
# import queue
# import time
# import threading
# import numpy as np
# import os
# from insightface.app import FaceAnalysis
# from insightface.model_zoo import model_zoo

# class FaceSwapProcessor:
#     def __init__(self, source_image_paths, display_width, display_height):
#         self.source_image_paths = source_image_paths
#         self.display_size = (display_width, display_height)
#         self.frame_queue = queue.Queue(maxsize=2)
#         self.result_queue = queue.Queue(maxsize=2)
#         self.running = False
#         self.swap_active = False
#         self.selected_face_key = None
#         self.last_detected_face = None
#         self.last_processed_frame = None
#         self.source_faces = {}
#         self.fps = 0

#         # Initialize face analysis and loading the swap model
#         self.app = FaceAnalysis(providers=["CPUExecutionProvider"])  # Ensure CPU usage
#         self.app.prepare(ctx_id=-1, det_size=(640, 640))  # Force CPU mode

#         model_path = "inswapper_128_fp16.onnx"
#         if not os.path.exists(model_path):
#             print("Please ensure the model file 'inswapper_128.onnx' is in the current directory.")
#             return

#         try:
#             self.swapper = model_zoo.get_model(model_path)
#         except Exception as e:
#             print(f"Error loading model: {e}")
#             return

#         # Load source faces
#         for key, image_path in self.source_image_paths.items():
#             if os.path.exists(image_path):
#                 image = cv2.imread(image_path)
#                 faces = self.app.get(image)
#                 if faces:
#                     self.source_faces[key] = faces[0]
#                     print(f"Successfully loaded face {key}")
#                 else:
#                     print(f"Failed to detect face in image: {image_path}")
#             else:
#                 print(f"Image path does not exist: {image_path}")

#     def start(self):
#         if not self.running:
#             self.running = True
#             self.worker_thread = threading.Thread(target=self.frame_processor_worker, daemon=True)
#             self.worker_thread.start()

#     def stop(self):
#         self.running = False
#         if self.worker_thread.is_alive():
#             self.worker_thread.join()

#     def process_frame(self, frame):
#         if not self.swap_active or self.selected_face_key is None:
#             print("Face swapping is inactive or no face selected.")
#             return None

#         try:
#             # Detect faces in the current frame
#             target_faces = self.app.get(frame)

#             if target_faces:
#                 self.last_detected_face = target_faces[0]
#             elif self.last_detected_face is not None:
#                 target_faces = [self.last_detected_face]
#             else:
#                 print("No face detected.")
#                 return None

#             # Perform face swap
#             print("Performing face swap...")
#             result = self.swapper.get(frame, target_faces[0], self.source_faces[self.selected_face_key], paste_back=True)
#             self.last_processed_frame = result
#             print("Face swap successful.")
#             return cv2.resize(result, self.display_size)

#         except Exception as e:
#             print(f"Error during face swap: {str(e)}")
#             if self.last_processed_frame is not None:
#                 return cv2.resize(self.last_processed_frame, self.display_size)
#             return None

#     def frame_processor_worker(self):
#         last_process_time = 0
#         min_process_interval = 1.0 / 30  # Max processing rate: 30 FPS

#         while self.running:
#             try:
#                 frame = self.frame_queue.get(timeout=0.1)

#                 # Process the frame if enough time has passed
#                 current_time = time.time()
#                 if current_time - last_process_time < min_process_interval:
#                     continue

#                 processed_frame = self.process_frame(frame)
#                 last_process_time = current_time

#                 # Only update result if processing occurred
#                 if processed_frame is not None:
#                     while not self.result_queue.empty():
#                         try:
#                             self.result_queue.get_nowait()
#                         except queue.Empty:
#                             break
#                     self.result_queue.put(processed_frame)
#                     print("Processed frame added to result queue.")

#             except queue.Empty:
#                 continue
#             except Exception as e:
#                 print(f"Worker error: {str(e)}")
#                 continue

#     def update_fps(self):
#         current_time = time.time()
#         if not hasattr(self, "_last_frame_time"):  # Initialize for the first frame
#             self._last_frame_time = current_time
#             self.fps = 0.0  # Set initial FPS to 0
#         else:
#             self.fps = 1 / (current_time - self._last_frame_time)
#         self._last_frame_time = current_time
#         return self.fps

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

#     # For virtual webcam, the output needs to be sent to v4l2loopback (Linux) or OBS virtual camera (Windows/macOS).
#     # On Linux, create a virtual webcam using v4l2loopback (e.g., /dev/video0).
#     # Replace cv2.VideoWriter with actual virtual webcam output if needed.

#     virtual_cam = cv2.VideoWriter("/dev/video0", cv2.VideoWriter_fourcc(*"MJPG"), 30, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

#     print("\nPress 1/2 to start face swap, 'esc' to stop swap, 'q' to quit")

#     try:
#         last_displayed_frame = None  # Cache last processed frame

#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 print("Error: Unable to read from camera.")
#                 break

#             # Add the current frame to the processing queue if empty
#             if processor.frame_queue.empty():
#                 processor.frame_queue.put(frame)

#             # Fetch the processed frame or fallback to the last frame
#             if processor.swap_active:
#                 try:
#                     display_frame = processor.result_queue.get_nowait()
#                     last_displayed_frame = display_frame  # Cache successfully swapped frame
#                     print("Displaying swapped frame...")
#                 except queue.Empty:
#                     if last_displayed_frame is not None:
#                         display_frame = last_displayed_frame
#                         print("Using last processed frame.")
#                     else:
#                         display_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
#                         print("Result queue empty. Showing original frame.")
#             else:
#                 # If swapping is inactive, show the original frame
#                 display_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
#                 last_displayed_frame = None  # Clear the cache when not swapping

#             # Send the frame to virtual webcam (like /dev/video0 on Linux or virtual camera on Windows/macOS)
#             virtual_cam.write(display_frame)

#             # Show the frame for debugging (optional)
#             cv2.imshow("Live Face Swap", display_frame)

#             # Handle keyboard input
#             key = cv2.waitKey(1) & 0xFF
#             if key == ord('q'):  # Quit
#                 break
#             elif key == 27:  # ESC key to stop swapping
#                 processor.swap_active = False
#                 processor.selected_face_key = None
#                 print("Face swap stopped.")
#             elif chr(key) in processor.source_faces:
#                 processor.selected_face_key = chr(key)
#                 processor.swap_active = True
#                 print(f"Selected face: {processor.selected_face_key}")

#     finally:
#         processor.stop()
#         cap.release()
#         virtual_cam.release()  # Release the virtual webcam resource
#         cv2.destroyAllWindows()

# if __name__ == '__main__':
#     main()
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import pyvirtualcam
import logging
import tkinter as tk
from tkinter import filedialog, ttk
import threading
from datetime import datetime
import os
from pathlib import Path
import sys

class FaceSwapGUI:
    def __init__(self):
        # Setup logging
        self.setup_logging()
        
        # Initialize variables
        self.DISPLAY_WIDTH = 1280
        self.DISPLAY_HEIGHT = 720
        self.FPS = 30
        self.running = False
        self.source_faces = {}
        self.selected_face = None
        self.swap_active = False
        
        try:
            # Initialize face detection
            self.init_face_analysis()
            
            # Create GUI
            self.setup_gui()
            
            self.logger.info("FaceSwapGUI initialized successfully")
        except Exception as e:
            self.logger.error(f"Initialization error: {str(e)}")
            raise

    def setup_logging(self):
        # Create logs directory in AppData
        appdata_path = os.path.join(os.getenv('APPDATA'), 'FaceSwapGUI')
        log_dir = Path(appdata_path) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"faceswap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def init_face_analysis(self):
        self.logger.info("Initializing face analysis...")
        self.face_analyzer = FaceAnalysis(name='buffalo_l')
        self.face_analyzer.prepare(ctx_id=0, det_size=(320, 320))
        
        model_path = 'inswapper_128_fp16.onnx'
        try:
            self.swapper = insightface.model_zoo.get_model(model_path)
            self.logger.info("Face swap model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading face swap model: {str(e)}")
            raise

    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Face Swap Virtual Camera")
        self.root.geometry("500x400")
        
        # Create main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Face management section
        face_frame = ttk.LabelFrame(main_frame, text="Face Management", padding="5")
        face_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(face_frame, text="Add New Face", command=self.load_face).pack(side=tk.LEFT, padx=5)
        ttk.Button(face_frame, text="Clear All Faces", command=self.clear_faces).pack(side=tk.LEFT, padx=5)
        
        # Face selection section
        select_frame = ttk.LabelFrame(main_frame, text="Face Selection", padding="5")
        select_frame.pack(fill=tk.X, pady=5)
        
        self.face_listbox = tk.Listbox(select_frame, height=5)
        self.face_listbox.pack(fill=tk.X, pady=5)
        self.face_listbox.bind('<<ListboxSelect>>', self.on_face_select)
        
        # Camera control section
        camera_frame = ttk.LabelFrame(main_frame, text="Camera Control", padding="5")
        camera_frame.pack(fill=tk.X, pady=5)
        
        self.start_button = ttk.Button(camera_frame, text="Start Camera", command=self.toggle_camera)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.swap_button = ttk.Button(camera_frame, text="Enable Swap", command=self.toggle_swap, state=tk.DISABLED)
        self.swap_button.pack(side=tk.LEFT, padx=5)
        
        # Status section
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding="5")
        status_frame.pack(fill=tk.X, pady=5)
        
        self.status_label = ttk.Label(status_frame, text="Ready to start")
        self.status_label.pack(fill=tk.X)

    def load_face(self):
        file_path = filedialog.askopenfilename(
            title="Select Face Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        
        if file_path:
            try:
                img = cv2.imread(file_path)
                faces = self.face_analyzer.get(img)
                
                if faces:
                    face_id = len(self.source_faces)
                    face_name = os.path.basename(file_path)
                    self.source_faces[face_id] = {
                        'face': faces[0],
                        'name': face_name
                    }
                    self.face_listbox.insert(tk.END, face_name)
                    self.status_label.config(text=f"Loaded face: {face_name}")
                    self.logger.info(f"Added new face: {face_name}")
                else:
                    self.status_label.config(text="No face detected in image")
            except Exception as e:
                self.status_label.config(text=f"Error loading face: {str(e)}")
                self.logger.error(f"Face loading error: {str(e)}")

    def clear_faces(self):
        self.source_faces.clear()
        self.face_listbox.delete(0, tk.END)
        self.selected_face = None
        self.swap_button.config(state=tk.DISABLED)
        self.status_label.config(text="All faces cleared")
        self.logger.info("Cleared all faces")

    def on_face_select(self, event):
        selection = self.face_listbox.curselection()
        if selection:
            self.selected_face = self.source_faces[selection[0]]['face']
            self.swap_button.config(state=tk.NORMAL)
            self.status_label.config(text=f"Selected face: {self.source_faces[selection[0]]['name']}")

    def toggle_swap(self):
        self.swap_active = not self.swap_active
        self.swap_button.config(text="Disable Swap" if self.swap_active else "Enable Swap")
        status = "enabled" if self.swap_active else "disabled"
        self.status_label.config(text=f"Face swap {status}")
        self.logger.info(f"Face swap {status}")

    # Relevant section that needs to be modified in the camera_worker method

    def camera_worker(self):
        try:
            cap = cv2.VideoCapture(0)
            
            # First get the actual camera resolution
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Try to set the desired resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.DISPLAY_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.DISPLAY_HEIGHT)
            
            # Check what resolution we actually got
            final_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            final_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            self.logger.info(f"Camera resolution: {final_width}x{final_height}")
            
            # Create virtual camera with the same resolution as the actual camera
            with pyvirtualcam.Camera(width=self.DISPLAY_WIDTH, height=self.DISPLAY_HEIGHT, fps=self.FPS) as cam:
                self.logger.info("Virtual camera started")
                
                while self.running:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Resize the frame to match virtual camera resolution
                    frame = cv2.resize(frame, (self.DISPLAY_WIDTH, self.DISPLAY_HEIGHT))

                    if self.swap_active and self.selected_face is not None:
                        try:
                            faces = self.face_analyzer.get(frame)
                            if faces:
                                frame = self.swapper.get(frame, faces[0], self.selected_face, paste_back=True)
                        except Exception as e:
                            self.logger.error(f"Face swap error: {str(e)}")

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    cam.send(frame_rgb)
                    cam.sleep_until_next_frame()

            cap.release()
            self.logger.info("Camera stopped")
            
        except Exception as e:
            self.logger.error(f"Camera worker error: {str(e)}")
            self.root.after(0, self.stop_camera)

    def toggle_camera(self):
        if not self.running:
            self.running = True
            self.start_button.config(text="Stop Camera")
            threading.Thread(target=self.camera_worker, daemon=True).start()
            self.status_label.config(text="Camera started")
        else:
            self.stop_camera()

    def stop_camera(self):
        self.running = False
        self.start_button.config(text="Start Camera")
        self.status_label.config(text="Camera stopped")

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

    def on_closing(self):
        self.stop_camera()
        self.root.destroy()

def main():
    try:
        app = FaceSwapGUI()
        app.run()
    except Exception as e:
        logging.error(f"Application error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()