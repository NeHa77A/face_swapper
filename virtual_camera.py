import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import pyvirtualcam
import logging
import tkinter as tk
from tkinter import filedialog, ttk
import threading

class FaceSwapApp:
    def __init__(self):
        # Initialize face detection and swap models
        self.face_analyzer = FaceAnalysis(name='buffalo_l')
        self.face_analyzer.prepare(ctx_id=0, det_size=(320, 320))
        self.swapper = insightface.model_zoo.get_model('inswapper_128_fp16.onnx', download=True)
        
        # Initialize variables
        self.source_face = None
        self.running = False
        self.source_img_path = None
        
        # Create GUI
        self.setup_gui()

    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Face Swap Virtual Camera")
        self.root.geometry("400x300")

        # Create frames
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(fill=tk.X)

        status_frame = ttk.Frame(self.root, padding="10")
        status_frame.pack(fill=tk.X)

        # Create buttons
        ttk.Button(control_frame, text="Select Source Face", command=self.load_source_face).pack(pady=5)
        self.toggle_btn = ttk.Button(control_frame, text="Start", command=self.toggle_camera)
        self.toggle_btn.pack(pady=5)

        # Status label
        self.status_label = ttk.Label(status_frame, text="Select a source face to begin")
        self.status_label.pack(pady=5)

    def load_source_face(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")]
        )
        if file_path:
            try:
                img = cv2.imread(file_path)
                faces = self.face_analyzer.get(img)
                if faces:
                    self.source_face = faces[0]
                    self.source_img_path = file_path
                    self.status_label.config(text="Source face loaded successfully")
                else:
                    self.status_label.config(text="No face detected in the source image")
            except Exception as e:
                self.status_label.config(text=f"Error loading image: {str(e)}")

    def start_camera(self):
        try:
            # Initialize camera
            cap = cv2.VideoCapture(0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            # Initialize virtual camera
            with pyvirtualcam.Camera(width=width, height=height, fps=fps) as cam:
                self.status_label.config(text="Virtual camera started")
                
                while self.running:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Process frame if source face is loaded
                    if self.source_face is not None:
                        # Detect faces in current frame
                        faces = self.face_analyzer.get(frame)
                        
                        # Swap faces
                        if faces:
                            for face in faces:
                                frame = self.swapper.get(frame, face, self.source_face, paste_back=True)

                    # Convert BGR to RGB for virtual camera
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    cam.send(frame_rgb)
                    cam.sleep_until_next_frame()

            cap.release()
            self.status_label.config(text="Camera stopped")

        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}")
            self.running = False
            self.toggle_btn.config(text="Start")

    def toggle_camera(self):
        if not self.source_face:
            self.status_label.config(text="Please select a source face first")
            return

        if not self.running:
            self.running = True
            self.toggle_btn.config(text="Stop")
            # Start camera in a separate thread
            threading.Thread(target=self.start_camera, daemon=True).start()
        else:
            self.running = False
            self.toggle_btn.config(text="Start")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = FaceSwapApp()
    app.run()