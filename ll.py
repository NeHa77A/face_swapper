# import cv2
# import numpy as np
# import insightface
# from insightface.app import FaceAnalysis
# import pyvirtualcam
# from threading import Thread, Lock
# from queue import Queue
# import time
# import command

# command.run_sudo_commands()


# def check_face_validity(face):
#     required_attrs = ['embedding', 'bbox', 'kps']
#     return all(hasattr(face, attr) for attr in required_attrs) and \
#            all(getattr(face, attr) is not None for attr in required_attrs)

# def process_source_image(img, app):
#     if len(img.shape) != 3 or img.shape[2] != 3:
#         raise ValueError("Image must be in BGR format")
    
#     faces = app.get(img)
#     if not faces:
#         raise ValueError("No faces detected in source image")
        
#     face = faces[0]
#     if not check_face_validity(face):
#         raise ValueError("Invalid face detection in source image")
        
#     return face

# def swap_faces_live(source_face, frame, app, swapper):
#     try:
#         if not check_face_validity(source_face):
#             return frame

#         faces = app.get(frame)
#         if not faces:
#             return frame

#         result = frame.copy()
#         for face in faces[:1]:
#             try:
#                 if not check_face_validity(face):
#                     continue

#                 if face.embedding is not None and source_face.embedding is not None:
#                     result = swapper.get(
#                         result, 
#                         face, 
#                         source_face, 
#                         paste_back=True
#                     )
#                     result = np.clip(result, 0, 255).astype(np.uint8)
#             except Exception as e:
#                 print(f"Swap error: {str(e)}")
#                 continue
                
#         return result
#     except Exception as e:
#         print(f"Processing error: {str(e)}")
#         return frame

# class VideoStreamProcessor:
#     def __init__(self, source_face):
#         self.frame_queue = Queue(maxsize=4)
#         self.display_queue = Queue(maxsize=4)
#         self.virtual_cam_queue = Queue(maxsize=4)
#         self.stopped = False
#         self.lock = Lock()
#         self.fps = 0
#         self.last_fps_update = time.time()
#         self.frame_times = []
#         self.source_face = source_face

#     def start(self):
#         Thread(target=self.capture_frames, args=(), daemon=True).start()
#         Thread(target=self.process_frames, args=(), daemon=True).start()
#         Thread(target=self.virtual_cam_output, args=(), daemon=True).start()
#         return self

#     def capture_frames(self):
#         while not self.stopped:
#             if not self.frame_queue.full():
#                 ret, frame = self.cap.read()
#                 if ret:
#                     self.frame_queue.put(frame)
#             else:
#                 time.sleep(0.001)

#     def process_frames(self):
#         while not self.stopped:
#             if self.frame_queue.empty():
#                 time.sleep(0.001)
#                 continue

#             try:
#                 frame = self.frame_queue.get()
#                 start_time = time.time()

#                 result = swap_faces_live(self.source_face, frame, app, swapper)

#                 process_time = time.time() - start_time
#                 with self.lock:
#                     self.frame_times.append(process_time)
#                     if len(self.frame_times) > 30:
#                         self.frame_times.pop(0)
                    
#                     current_time = time.time()
#                     if current_time - self.last_fps_update >= 1.0:
#                         self.fps = len(self.frame_times) / sum(self.frame_times)
#                         self.last_fps_update = current_time

#                 preview = result.copy()
#                 cv2.putText(preview, f"FPS: {self.fps:.1f}", 
#                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#                 if not self.display_queue.full():
#                     self.display_queue.put(preview)
#                 if not self.virtual_cam_queue.full():
#                     self.virtual_cam_queue.put(result)

#             except Exception as e:
#                 print(f"Processing error: {str(e)}")
#                 continue

#     def virtual_cam_output(self):
#         try:
#             with pyvirtualcam.Camera(width=640, height=480, fps=30) as cam:
#                 print(f'Virtual camera created: {cam.device}')
#                 while not self.stopped:
#                     if self.virtual_cam_queue.empty():
#                         time.sleep(0.001)
#                         continue
                    
#                     frame = self.virtual_cam_queue.get()
#                     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                     cam.send(frame_rgb)
#                     cam.sleep_until_next_frame()
#         except Exception as e:
#             print(f"Virtual camera error: {str(e)}")
#             print("Please ensure v4l2loopback is properly installed and loaded")

#     def set_cap(self, cap):
#         self.cap = cap

#     def stop(self):
#         self.stopped = True

# if __name__ == '__main__':
#     try:
#         # Initialize face detection and swap models
#         app = FaceAnalysis(name='buffalo_l', allowed_modules=['detection', 'recognition'])
#         app.prepare(ctx_id=0 if cv2.cuda.getCudaEnabledDeviceCount() > 0 else -1, det_size=(640, 640))
        
#         model_path = 'inswapper_128_fp162.onnx'
#         swapper = insightface.model_zoo.get_model(model_path)
#         if swapper is None:
#             raise ValueError("Failed to load face swap model")

#         # Load Ajay Devgn's image
#         img_path = "/home/suyodhan/Desktop/noob/demo/testing_video/code/pra/images/Ajay_Devgn_at_the_launch_of_MTV_Super_Fight_League.jpg"
#         img = cv2.imread(img_path)
#         if img is None:
#             raise ValueError(f"Failed to load image: {img_path}")
            
#         source_face = process_source_image(img, app)
#         print("Successfully loaded source face")

#         # Initialize video capture
#         cap = cv2.VideoCapture(0)
#         cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#         cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#         cap.set(cv2.CAP_PROP_FPS, 30)
#         cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        
#         if not cap.isOpened():
#             raise ValueError("Unable to access webcam")

#         video_processor = VideoStreamProcessor(source_face)
#         video_processor.set_cap(cap)
#         video_processor.start()
        
#         print("\nPress 'q' to quit")
#         print("Virtual camera initialized - ready for use in Google Meet")
        
#         while True:
#             if video_processor.display_queue.empty():
#                 time.sleep(0.001)
#                 continue

#             frame = video_processor.display_queue.get()
#             cv2.imshow("Preview (Local Only)", frame)

#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#     except Exception as e:
#         print(f"An error occurred: {str(e)}")
    
#     finally:
#         if 'video_processor' in locals():
#             video_processor.stop()
#         if 'cap' in locals():
#             cap.release()
#         cv2.destroyAllWindows()

import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import pyvirtualcam
from threading import Thread, Lock
from queue import Queue
import time
import os

def check_face_validity(face):
    required_attrs = ['embedding', 'bbox', 'kps']
    return all(hasattr(face, attr) for attr in required_attrs) and \
           all(getattr(face, attr) is not None for attr in required_attrs)

def process_source_image(img, app):
    if len(img.shape) != 3 or img.shape[2] != 3:
        raise ValueError("Image must be in BGR format")
    
    faces = app.get(img)
    if not faces:
        raise ValueError("No faces detected in source image")
        
    face = faces[0]
    if not check_face_validity(face):
        raise ValueError("Invalid face detection in source image")
        
    return face

def swap_faces_live(source_face, frame, app, swapper):
    try:
        if not check_face_validity(source_face):
            return frame

        faces = app.get(frame)
        if not faces:
            return frame

        result = frame.copy()
        for face in faces[:1]:
            try:
                if not check_face_validity(face):
                    continue

                if face.embedding is not None and source_face.embedding is not None:
                    result = swapper.get(
                        result, 
                        face, 
                        source_face, 
                        paste_back=True
                    )
                    result = np.clip(result, 0, 255).astype(np.uint8)
            except Exception as e:
                print(f"Swap error: {str(e)}")
                continue
                
        return result
    except Exception as e:
        print(f"Processing error: {str(e)}")
        return frame

class VideoStreamProcessor:
    def __init__(self, source_face):
        self.frame_queue = Queue(maxsize=4)
        self.display_queue = Queue(maxsize=4)
        self.virtual_cam_queue = Queue(maxsize=4)
        self.stopped = False
        self.lock = Lock()
        self.fps = 0
        self.last_fps_update = time.time()
        self.frame_times = []
        self.source_face = source_face

    def start(self):
        Thread(target=self.capture_frames, args=(), daemon=True).start()
        Thread(target=self.process_frames, args=(), daemon=True).start()
        Thread(target=self.virtual_cam_output, args=(), daemon=True).start()
        return self

    def capture_frames(self):
        while not self.stopped:
            if not self.frame_queue.full():
                ret, frame = self.cap.read()
                if ret:
                    self.frame_queue.put(frame)
            else:
                time.sleep(0.001)

    def process_frames(self):
        while not self.stopped:
            if self.frame_queue.empty():
                time.sleep(0.001)
                continue

            try:
                frame = self.frame_queue.get()
                start_time = time.time()

                result = swap_faces_live(self.source_face, frame, app, swapper)

                process_time = time.time() - start_time
                with self.lock:
                    self.frame_times.append(process_time)
                    if len(self.frame_times) > 30:
                        self.frame_times.pop(0)
                    
                    current_time = time.time()
                    if current_time - self.last_fps_update >= 1.0:
                        self.fps = len(self.frame_times) / sum(self.frame_times)
                        self.last_fps_update = current_time

                preview = result.copy()
                cv2.putText(preview, f"FPS: {self.fps:.1f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if not self.display_queue.full():
                    self.display_queue.put(preview)
                if not self.virtual_cam_queue.full():
                    self.virtual_cam_queue.put(result)

            except Exception as e:
                print(f"Processing error: {str(e)}")
                continue

    def virtual_cam_output(self):
        try:
            # For Windows, we'll use OBS Virtual Camera
            with pyvirtualcam.Camera(width=640, height=480, fps=30, device='OBS Virtual Camera') as cam:
                print(f'Virtual camera created: {cam.device}')
                while not self.stopped:
                    if self.virtual_cam_queue.empty():
                        time.sleep(0.001)
                        continue
                    
                    frame = self.virtual_cam_queue.get()
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    cam.send(frame_rgb)
                    cam.sleep_until_next_frame()
        except Exception as e:
            print(f"Virtual camera error: {str(e)}")
            print("Please ensure OBS Virtual Camera is installed")

    def set_cap(self, cap):
        self.cap = cap

    def stop(self):
        self.stopped = True

if __name__ == '__main__':
    try:
        # Initialize face detection and swap models
        app = FaceAnalysis(name='buffalo_l', allowed_modules=['detection', 'recognition'])
        # Use CPU by default on Windows as CUDA setup can be tricky
        app.prepare(ctx_id=-1, det_size=(640, 640))
        
        # Update model path for Windows
        model_path = os.path.join(os.path.dirname(__file__), 'inswapper_128_fp16.onnx')
        swapper = insightface.model_zoo.get_model(model_path)
        if swapper is None:
            raise ValueError("Failed to load face swap model")

        # Update image path for Windows
        img_path = os.path.join(os.path.dirname(__file__), 'images', 'khadus.jpg')
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
            
        source_face = process_source_image(img, app)
        print("Successfully loaded source face")

        # Initialize video capture
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        
        if not cap.isOpened():
            raise ValueError("Unable to access webcam")

        video_processor = VideoStreamProcessor(source_face)
        video_processor.set_cap(cap)
        video_processor.start()
        
        print("\nPress 'q' to quit")
        print("Virtual camera initialized - ready for use in video calls")
        
        while True:
            if video_processor.display_queue.empty():
                time.sleep(0.001)
                continue

            frame = video_processor.display_queue.get()
            cv2.imshow("Preview (Local Only)", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"An error occurred: {str(e)}")
    
    finally:
        if 'video_processor' in locals():
            video_processor.stop()
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()